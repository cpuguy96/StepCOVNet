"""Beat frontend encoders, condition projections, and heads for ITGPT placement."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.itgpt import constants
from stepcovnet.itgpt.transformer import ItgptRmsNorm, _init_linear_or_conv


class ItgptBeatCNNEncoder(nn.Module):
    """ITGPT ``BeatCNNEncoder`` on ``(32, 80, 3)`` log-mel beats.

    Accepts channels-last ``(BT, 32, 80, 3)`` to match
    :class:`stepcovnet.ddcl.datasets.DdclChart` directly, permutes internally
    for PyTorch ``Conv2d``, and permutes back before flattening in channels-last
    order for ``proj0``.

    Args:
        d_model: Projected output width.
        hidden: Base convolution channels.
    """

    def __init__(self, d_model: int, hidden: int = constants.CNN_HIDDEN) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.conv0 = nn.Conv2d(constants.N_CHANNELS, self.hidden, kernel_size=(7, 3))
        self.conv1 = nn.Conv2d(
            self.hidden,
            self.hidden,
            kernel_size=(3, 3),
            stride=(1, 3),
            padding=(1, 0),
        )
        self.conv2 = nn.Conv2d(self.hidden, self.hidden * 2, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(
            self.hidden * 2,
            self.hidden * 2,
            kernel_size=(3, 3),
            stride=(1, 3),
            padding=(1, 0),
        )
        self.proj0 = nn.Linear(self.hidden * 16, self.hidden * 16)
        self.proj1 = nn.Linear(self.hidden * 16, d_model)
        for layer in (
            self.conv0,
            self.conv1,
            self.conv2,
            self.conv3,
            self.proj0,
            self.proj1,
        ):
            _init_linear_or_conv(layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode each beat spectrogram.

        Args:
            inputs: ``(batch * beats, 32, 80, 3)``.

        Returns:
            ``(batch * beats, 24, d_model)``.
        """
        hidden = inputs.permute(0, 3, 1, 2).contiguous()
        hidden = nn_func.gelu(self.conv0(hidden))
        hidden = nn_func.gelu(self.conv1(hidden))
        hidden = nn_func.gelu(self.conv2(hidden))
        hidden = nn_func.gelu(self.conv3(hidden))
        # Permute (BT, 2H, 24, 8) -> (BT, 24, 8, 2H) for channels-last flatten.
        hidden = hidden.permute(0, 2, 3, 1).contiguous()
        hidden = hidden.reshape(-1, constants.CNN_FRAMES_OUT, self.hidden * 16)
        return self.proj1(nn_func.gelu(self.proj0(hidden)))


class ItgptBeatMertEncoder(nn.Module):
    """Beat-aligned MERT encoder producing ``(batch * beats, 24, d_model)``.

    Args:
        feature_dim: Per-frame MERT width (e.g. 1024).
        d_model: Projected width.
        dropout_rate: Dropout after the first linear projection.
        n_mert_layers: Stacked MERT layer count (``1`` disables layer mixing).
        mert_time_mask_max_frames: SpecAugment maximum contiguous MERT frames
            zeroed per beat during training (``0`` disables time masking).
        mert_channel_dropout: Channel-wise dropout probability across the MERT
            feature axis per beat during training (``0.0`` disables).
        mert_time_mask_preserve_downbeat: If True, sample time-mask starts in
            ``[1, n_frames)`` so frame ``0`` (beat downbeat) is never zeroed.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        dropout_rate: float = constants.DROPOUT_RATE,
        n_mert_layers: int = 1,
        mert_time_mask_max_frames: int = 0,
        mert_channel_dropout: float = 0.0,
        mert_time_mask_preserve_downbeat: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.dropout_rate = float(dropout_rate)
        self.n_mert_layers = int(n_mert_layers)
        self.mert_time_mask_max_frames = int(mert_time_mask_max_frames)
        self.mert_channel_dropout = float(mert_channel_dropout)
        self.mert_time_mask_preserve_downbeat = bool(mert_time_mask_preserve_downbeat)
        if self.n_mert_layers > 1:
            self.layer_logits: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.n_mert_layers, dtype=torch.float32)
            )
        else:
            self.register_parameter("layer_logits", None)
        self.norm = ItgptRmsNorm(self.feature_dim)
        self.proj0 = nn.Linear(self.feature_dim, self.d_model)
        self.drop = nn.Dropout(self.dropout_rate)
        self.conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)
        self.proj1 = nn.Linear(self.d_model, self.d_model)
        for layer in (self.proj0, self.conv, self.proj1):
            _init_linear_or_conv(layer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode each beat's MERT frame sequence.

        Args:
            inputs: ``(N, L, feature_dim)`` when ``n_mert_layers == 1``, or
                ``(N, L, n_mert_layers, feature_dim)`` when ``n_mert_layers > 1``.

        Returns:
            ``(N, L, d_model)``.
        """
        hidden = inputs
        if self.n_mert_layers > 1 and self.layer_logits is not None:
            weights = torch.softmax(self.layer_logits.float(), dim=0).to(hidden.dtype)
            hidden = torch.sum(
                hidden * weights.view(1, 1, self.n_mert_layers, 1),
                dim=-2,
            )
        hidden = self.norm(hidden)
        if self.training:
            if self.mert_time_mask_max_frames > 0:
                n_items = hidden.shape[0]
                n_frames = hidden.shape[1]
                start_low = (
                    1 if self.mert_time_mask_preserve_downbeat and n_frames > 1 else 0
                )
                max_w = min(self.mert_time_mask_max_frames, n_frames - start_low)
                widths = torch.randint(0, max_w + 1, (n_items, 1), device=hidden.device)
                starts = torch.randint(
                    start_low, n_frames, (n_items, 1), device=hidden.device
                )
                pos = torch.arange(n_frames, device=hidden.device).unsqueeze(0)
                time_keep = (pos < starts) | (pos >= (starts + widths))
                hidden = hidden * time_keep.unsqueeze(-1).to(hidden.dtype)
            if self.mert_channel_dropout > 0.0:
                keep_prob = 1.0 - self.mert_channel_dropout
                chan_mask = (
                    torch.rand(
                        (hidden.shape[0], 1, hidden.shape[2]),
                        device=hidden.device,
                        dtype=torch.float32,
                    )
                    < keep_prob
                ).to(hidden.dtype) / keep_prob
                hidden = hidden * chan_mask
        hidden = self.drop(nn_func.gelu(self.proj0(hidden)))
        hidden = nn_func.gelu(self.conv(hidden.transpose(1, 2))).transpose(1, 2)
        return self.proj1(hidden)


class ItgptConvHead(nn.Module):
    """ITGPT ``ConvHead`` residual depthwise/pointwise conv over beats.

    Args:
        d_model: Channel width.
        dropout: Dropout after the pointwise conv.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = ItgptRmsNorm(d_model)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        _init_linear_or_conv(self.dw)
        _init_linear_or_conv(self.pw)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add a local depthwise/pointwise conv residual.

        Args:
            inputs: ``(batch, beats, d_model)``.

        Returns:
            Same shape as ``inputs``.
        """
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = self.pw(nn_func.gelu(self.dw(hidden))).transpose(1, 2)
        return inputs + self.drop(hidden)


class ItgptConditionProj(nn.Module):
    """Min-max normalize a scalar and map to a condition token.

    Args:
        d_model: Token width.
        min_val: Lower clip bound.
        max_val: Upper clip bound.
    """

    def __init__(self, d_model: int, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.dense0 = nn.Linear(1, d_model // 2)
        self.dense1 = nn.Linear(d_model // 2, d_model)
        _init_linear_or_conv(self.dense0)
        _init_linear_or_conv(self.dense1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project ``(batch, 1)`` to ``(batch, 1, d_model)``.

        Args:
            inputs: Scalar condition per chart.

        Returns:
            Token sequence of length 1.
        """
        scaled = torch.clamp(
            (inputs - self.min_val) / (self.max_val - self.min_val),
            0.0,
            1.0,
        )
        return self.dense1(nn_func.gelu(self.dense0(scaled))).unsqueeze(1)


class ItgptBeatAuxHead(nn.Module):
    """Auxiliary per-beat activity/density head with zero-init slot modulation.

    Predicts per-beat activity logits and ``log1p(step_count)`` from
    post-``conv_head`` beat representations while adding a zero-initialized
    residual logit modulation into the 48 slot logits so step-0 output matches
    the baseline exactly.

    Args:
        d_model: Incoming beat feature width.
        n_slots: Number of intra-beat slots (default 48).
        hidden_dim: Optional hidden width (defaults to ``max(32, d_model // 2)``).

    Raises:
        ValueError: If dimensions are non-positive.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = constants.N_SLOTS,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_slots < 1:
            raise ValueError(f"n_slots must be positive, got {n_slots}")
        resolved_hidden = (
            max(32, int(d_model) // 2) if hidden_dim is None else int(hidden_dim)
        )
        if resolved_hidden < 1:
            raise ValueError(f"hidden_dim must be positive, got {resolved_hidden}")
        self.d_model = int(d_model)
        self.n_slots = int(n_slots)
        self.hidden_dim = resolved_hidden
        self.norm = ItgptRmsNorm(self.d_model)
        self.hidden = nn.Linear(self.d_model, self.hidden_dim)
        self.aux_proj = nn.Linear(self.hidden_dim, 2)
        self.gate_proj = nn.Linear(self.hidden_dim, self.n_slots)
        _init_linear_or_conv(self.hidden)
        _init_linear_or_conv(self.aux_proj)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self, global_h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute beat auxiliary predictions and zero-init slot residual.

        Args:
            global_h: Post-``conv_head`` beat features ``(B, T, d_model)``.

        Returns:
            Tuple ``(beat_active_logit, beat_count_pred, slot_delta)`` in
            float32 with shapes ``(B, T)``, ``(B, T)``, and ``(B, T, n_slots)``.
        """
        feat = nn_func.gelu(self.hidden(self.norm(global_h)))
        feat_fp32 = feat.float()
        aux = nn_func.linear(
            feat_fp32,
            self.aux_proj.weight.float(),
            self.aux_proj.bias.float() if self.aux_proj.bias is not None else None,
        )
        slot_delta = nn_func.linear(
            feat_fp32,
            self.gate_proj.weight.float(),
            self.gate_proj.bias.float() if self.gate_proj.bias is not None else None,
        )
        return aux[..., 0], aux[..., 1], slot_delta
