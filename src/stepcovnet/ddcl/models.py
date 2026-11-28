"""PyTorch DDCL branched ConvLSTM and MERT placement models (`omalley2025ddcl`)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.ddcl import constants


def _init_xavier_uniform(module: nn.Module) -> None:
    """Apply Glorot/Xavier uniform weights and zero bias.

    Args:
        module: Layer to initialize if linear or conv.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvLSTM2d(nn.Module):
    """2D Convolutional LSTM with valid spatial convolutions.

    Applies valid spatial convolution to inputs ``x_t`` and same-padded spatial
    convolution to the recurrent hidden state ``h_{t-1}``.

    Args:
        in_channels: Input channel count.
        out_channels: Output / hidden channel count.
        kernel_size: Spatial kernel ``(kh, kw)``.
        go_backwards: If True, iterate time in reverse order.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        go_backwards: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.go_backwards = bool(go_backwards)
        self.input_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * 4,
            kernel_size=self.kernel_size,
            padding=0,
        )
        pad_h = self.kernel_size[0] // 2
        pad_w = self.kernel_size[1] // 2
        self.recurrent_conv = nn.Conv2d(
            self.out_channels,
            self.out_channels * 4,
            kernel_size=self.kernel_size,
            padding=(pad_h, pad_w),
            bias=False,
        )
        _init_xavier_uniform(self.input_conv)
        nn.init.orthogonal_(self.recurrent_conv.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run ConvLSTM2D across ``(batch, time, channels, height, width)``.

        Args:
            inputs: Sequence of feature maps ``(B, T, C, H, W)``.

        Returns:
            Sequence of hidden maps ``(B, T, out_channels, H_out, W_out)``.
        """
        batch_size, seq_len = int(inputs.shape[0]), int(inputs.shape[1])
        steps = range(seq_len - 1, -1, -1) if self.go_backwards else range(seq_len)
        h_state: torch.Tensor | None = None
        c_state: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for t_idx in steps:
            x_conv = self.input_conv(inputs[:, t_idx])
            if h_state is None or c_state is None:
                h_out, w_out = int(x_conv.shape[2]), int(x_conv.shape[3])
                h_state = torch.zeros(
                    (batch_size, self.out_channels, h_out, w_out),
                    dtype=x_conv.dtype,
                    device=x_conv.device,
                )
                c_state = torch.zeros_like(h_state)
            gates = x_conv + self.recurrent_conv(h_state)
            i_gate, f_gate, c_cand, o_gate = gates.chunk(4, dim=1)
            c_state = torch.sigmoid(f_gate) * c_state + torch.sigmoid(
                i_gate
            ) * torch.tanh(c_cand)
            h_state = torch.sigmoid(o_gate) * torch.tanh(c_state)
            outputs.append(h_state)
        return torch.stack(outputs, dim=1)


class _DdclLogmelBranch(nn.Module):
    """Single forward or backward ConvLSTM2D + MaxPool branch.

    Args:
        go_backwards: Whether the first ConvLSTM2D runs in reverse time order.
    """

    def __init__(self, *, go_backwards: bool) -> None:
        super().__init__()
        self.convlstm0 = ConvLSTM2d(
            constants.N_CHANNELS,
            constants.CONV1_FILTERS,
            constants.CONV1_KERNEL,
            go_backwards=go_backwards,
        )
        self.pool0 = nn.MaxPool2d(
            kernel_size=(constants.POOL_SIZE[1], constants.POOL_SIZE[2]),
            stride=(constants.POOL_SIZE[1], constants.POOL_SIZE[2]),
        )
        self.convlstm1 = ConvLSTM2d(
            constants.CONV1_FILTERS,
            constants.CONV2_FILTERS,
            constants.CONV2_KERNEL,
            go_backwards=False,
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=(constants.POOL_SIZE[1], constants.POOL_SIZE[2]),
            stride=(constants.POOL_SIZE[1], constants.POOL_SIZE[2]),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, n_frames, 80, 3)`` into ``(B, T, flat_dim)``.

        Args:
            audio: Channels-last beat audio windows.

        Returns:
            Flattened per-beat features ``(B, T, flat_dim)``.
        """
        batch_size, seq_len = int(audio.shape[0]), int(audio.shape[1])
        x = audio.permute(0, 1, 4, 2, 3).contiguous()
        x = self.convlstm0(x)
        x = self.pool0(x.reshape(batch_size * seq_len, *x.shape[2:])).reshape(
            batch_size, seq_len, constants.CONV1_FILTERS, -1, 26
        )
        x = self.convlstm1(x)
        x = self.pool1(x.reshape(batch_size * seq_len, *x.shape[2:])).reshape(
            batch_size, seq_len, constants.CONV2_FILTERS, -1, 8
        )
        # Permute (B, T, C, H, W) -> (B, T, H, W, C) for channels-last flatten.
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x.reshape(batch_size, seq_len, -1)


class _DdclHead(nn.Module):
    """Shared dual-branch LSTM + LeakyReLU DNN + 48-slot head.

    Args:
        branch_in_dim: Per-beat encoded audio feature dimension.
        lstm_units: Hidden units in each LSTM layer.
        dropout_rate: Dropout probability between layers.
        dense_sizes: Widths of the post-LSTM LeakyReLU dense layers.
        late_stream_merge: Whether to concatenate stream features after LSTM0.
    """

    def __init__(
        self,
        *,
        branch_in_dim: int,
        lstm_units: int,
        dropout_rate: float,
        dense_sizes: tuple[int, ...],
        late_stream_merge: bool = False,
    ) -> None:
        super().__init__()
        self.late_stream_merge = bool(late_stream_merge)
        self.in_drop = nn.Dropout(dropout_rate)
        if self.late_stream_merge:
            self.fwd_lstm0 = nn.LSTM(branch_in_dim, lstm_units, batch_first=True)
            self.bwd_lstm0 = nn.LSTM(branch_in_dim, lstm_units, batch_first=True)
            lstm1_in = lstm_units + constants.STREAM_DIM
        else:
            lstm0_in = branch_in_dim + constants.STREAM_DIM
            self.fwd_lstm0 = nn.LSTM(lstm0_in, lstm_units, batch_first=True)
            self.bwd_lstm0 = nn.LSTM(lstm0_in, lstm_units, batch_first=True)
            lstm1_in = lstm_units
        self.mid_drop = nn.Dropout(dropout_rate)
        self.fwd_lstm1 = nn.LSTM(lstm1_in, lstm_units, batch_first=True)
        self.bwd_lstm1 = nn.LSTM(lstm1_in, lstm_units, batch_first=True)

        dnn_layers: list[nn.Module] = []
        in_dim = lstm_units * 2
        for width in dense_sizes:
            lin = nn.Linear(in_dim, width)
            _init_xavier_uniform(lin)
            dnn_layers.extend(
                [lin, nn.LeakyReLU(negative_slope=0.2), nn.Dropout(dropout_rate)]
            )
            in_dim = width
        self.dnn = nn.Sequential(*dnn_layers)
        self.slots = nn.Linear(in_dim, constants.N_SLOTS)
        _init_xavier_uniform(self.slots)

    def forward_logits(
        self,
        audio_fwd: torch.Tensor,
        audio_bwd: torch.Tensor,
        stream_fwd: torch.Tensor,
        stream_bwd: torch.Tensor,
    ) -> torch.Tensor:
        """Run dual LSTM branches and output float32 48-slot logits.

        Args:
            audio_fwd: Encoded forward sequence ``(B, T, D)``.
            audio_bwd: Encoded backward sequence ``(B, T, D)``.
            stream_fwd: Forward ``[meter, bpm]`` stream ``(B, T, 2)``.
            stream_bwd: Backward ``[meter, bpm]`` stream ``(B, T, 2)``.

        Returns:
            Float32 logits tensor of shape ``(B, 48)``.
        """
        sf = stream_fwd.to(audio_fwd.dtype)
        sb = stream_bwd.to(audio_bwd.dtype)
        if self.late_stream_merge:
            fwd_seq, _ = self.fwd_lstm0(self.in_drop(audio_fwd))
            bwd_seq, _ = self.bwd_lstm0(torch.flip(self.in_drop(audio_bwd), dims=[1]))
            fwd_in = torch.cat([fwd_seq, sf], dim=-1)
            bwd_in = torch.cat([bwd_seq, sb], dim=-1)
        else:
            fwd_seq, _ = self.fwd_lstm0(
                self.in_drop(torch.cat([audio_fwd, sf], dim=-1))
            )
            bwd_seq, _ = self.bwd_lstm0(
                self.in_drop(torch.cat([audio_bwd, sb], dim=-1))
            )
            fwd_in = fwd_seq
            bwd_in = bwd_seq
        fwd_out, _ = self.fwd_lstm1(self.mid_drop(fwd_in))
        bwd_out, _ = self.bwd_lstm1(self.mid_drop(bwd_in))
        combined = torch.cat([fwd_out[:, -1, :], bwd_out[:, -1, :]], dim=-1)
        hidden = self.dnn(combined)
        return nn_func.linear(
            hidden.float(),
            self.slots.weight.float(),
            self.slots.bias.float() if self.slots.bias is not None else None,
        )


class DdclConvLstmPlacementModel(nn.Module):
    """PyTorch DDCL branched ConvLSTM placement model (`omalley2025ddcl`).

    Args:
        memlen: Context beats besides the current one (paper: 15).
        n_frames: Audio frames per beat (paper: 32).
        lstm_units: LSTM width (paper: 200).
        dropout_rate: Dropout on LSTM and dense layers (paper: 0.2).
        dense_sizes: Hidden widths (paper: 512, 256).

    Raises:
        ValueError: If sizes are invalid.
    """

    def __init__(
        self,
        *,
        memlen: int = constants.MEMLEN,
        n_frames: int = constants.N_FRAMES_PER_BEAT,
        lstm_units: int = constants.LSTM_UNITS,
        dropout_rate: float = constants.DROPOUT_RATE,
        dense_sizes: tuple[int, ...] = constants.DENSE_SIZES,
    ) -> None:
        super().__init__()
        if memlen < 0:
            raise ValueError(f"memlen must be >= 0, got {memlen}")
        if n_frames < 9:
            raise ValueError(
                f"n_frames must be >= 9 for valid (7,3)+(3,3) ConvLSTM, got {n_frames}"
            )
        if lstm_units < 1:
            raise ValueError(f"lstm_units must be at least 1, got {lstm_units}")
        self.memlen = int(memlen)
        self.n_frames = int(n_frames)
        self.lstm_units = int(lstm_units)
        self.dropout_rate = float(dropout_rate)
        self.dense_sizes = tuple(int(w) for w in dense_sizes)

        self.fwd_branch = _DdclLogmelBranch(go_backwards=False)
        self.bwd_branch = _DdclLogmelBranch(go_backwards=True)
        flat_dim = (self.n_frames - 8) * 8 * constants.CONV2_FILTERS
        self.head = _DdclHead(
            branch_in_dim=flat_dim,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            dense_sizes=self.dense_sizes,
            late_stream_merge=False,
        )

    def forward_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute float32 48-slot logits ``(batch, 48)``.

        Args:
            inputs: Dict with ``audio_fwd``, ``audio_bwd``, ``stream_fwd``, ``stream_bwd``.

        Returns:
            Float32 logits ``(batch, 48)``.
        """
        out_fwd = self.fwd_branch(inputs["audio_fwd"])
        out_bwd = self.bwd_branch(inputs["audio_bwd"])
        return self.head.forward_logits(
            out_fwd, out_bwd, inputs["stream_fwd"], inputs["stream_bwd"]
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute 48-slot probabilities ``(batch, 48)``.

        Args:
            inputs: DDCL input dictionary.

        Returns:
            Float32 probabilities ``(batch, 48)``.
        """
        return torch.sigmoid(self.forward_logits(inputs))

    def model_config(self) -> dict[str, object]:
        """Return constructor kwargs for checkpoint serialization.

        Returns:
            JSON-serializable constructor kwargs mapping.
        """
        return {
            "arch": "convlstm",
            "memlen": self.memlen,
            "n_frames": self.n_frames,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "dense_sizes": list(self.dense_sizes),
        }


class DdclMertPlacementModel(nn.Module):
    """PyTorch beat-aligned MERT DDCL placement model.

    Args:
        memlen: Context beats besides the current one (paper: 15).
        n_frames: Sub-beat MERT frames per beat (default: 16).
        feature_dim: MERT embedding width (default: 1024).
        proj_dim: Per-frame projection width (default: 128).
        conv_filters: Sub-beat 1D conv filters (default: 64).
        noise_std: Optional train-time Gaussian noise stddev after LayerNorm.
        lstm_units: Hidden width per LSTM layer (paper: 200).
        dropout_rate: Dropout rate.
        dense_sizes: Fully-connected hidden widths (paper: 512, 256).

    Raises:
        ValueError: If any dimension hyperparameter is invalid.
    """

    def __init__(
        self,
        *,
        memlen: int = constants.MEMLEN,
        n_frames: int = constants.MERT_FRAMES_PER_BEAT,
        feature_dim: int = constants.MERT_FEATURE_DIM,
        proj_dim: int = constants.MERT_PROJ_DIM,
        conv_filters: int = constants.MERT_CONV_FILTERS,
        noise_std: float = 0.0,
        lstm_units: int = constants.LSTM_UNITS,
        dropout_rate: float = constants.DROPOUT_RATE,
        dense_sizes: tuple[int, ...] = constants.DENSE_SIZES,
    ) -> None:
        super().__init__()
        if memlen < 0:
            raise ValueError(f"memlen must be >= 0, got {memlen}")
        if n_frames < 1:
            raise ValueError(f"n_frames must be at least 1, got {n_frames}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be at least 1, got {feature_dim}")
        if proj_dim < 1:
            raise ValueError(f"proj_dim must be at least 1, got {proj_dim}")
        if conv_filters < 1:
            raise ValueError(f"conv_filters must be at least 1, got {conv_filters}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        if lstm_units < 1:
            raise ValueError(f"lstm_units must be at least 1, got {lstm_units}")

        self.memlen = int(memlen)
        self.n_frames = int(n_frames)
        self.feature_dim = int(feature_dim)
        self.proj_dim = int(proj_dim)
        self.conv_filters = int(conv_filters)
        self.noise_std = float(noise_std)
        self.lstm_units = int(lstm_units)
        self.dropout_rate = float(dropout_rate)
        self.dense_sizes = tuple(int(w) for w in dense_sizes)

        self.mert_ln = nn.LayerNorm(self.feature_dim, eps=1e-3)
        self.mert_proj = nn.Linear(self.feature_dim, self.proj_dim)
        self.mert_drop = nn.Dropout(self.dropout_rate)
        self.mert_conv = nn.Conv1d(
            self.proj_dim,
            self.conv_filters,
            kernel_size=3,
            padding=1,
        )
        self.mert_pool = (
            nn.AvgPool1d(kernel_size=2, stride=2) if self.n_frames >= 2 else None
        )
        _init_xavier_uniform(self.mert_proj)
        _init_xavier_uniform(self.mert_conv)

        pooled_frames = self.n_frames // 2 if self.n_frames >= 2 else self.n_frames
        flat_dim = pooled_frames * self.conv_filters
        self.head = _DdclHead(
            branch_in_dim=flat_dim,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate,
            dense_sizes=self.dense_sizes,
            late_stream_merge=True,
        )

    def _encode_mert(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, n_frames, feature_dim)`` into ``(B, T, flat_dim)``.

        Args:
            audio: Beat-aligned MERT embeddings ``(B, T, n_frames, feature_dim)``.

        Returns:
            Flattened per-beat features ``(B, T, flat_dim)``.
        """
        batch_size, seq_len = int(audio.shape[0]), int(audio.shape[1])
        x = self.mert_ln(audio)
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        x = self.mert_drop(nn_func.gelu(self.mert_proj(x)))
        x = x.reshape(batch_size * seq_len, self.n_frames, self.proj_dim).transpose(
            1, 2
        )
        x = nn_func.gelu(self.mert_conv(x))
        if self.mert_pool is not None:
            x = self.mert_pool(x)
        # (BT, conv_filters, pooled_frames) -> (B, T, pooled_frames, conv_filters)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, seq_len, -1)

    def forward_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute float32 48-slot logits ``(batch, 48)``.

        Args:
            inputs: Dict with ``audio_fwd``, ``audio_bwd``, ``stream_fwd``, ``stream_bwd``.

        Returns:
            Float32 logits ``(batch, 48)``.
        """
        out_fwd = self._encode_mert(inputs["audio_fwd"])
        out_bwd = self._encode_mert(inputs["audio_bwd"])
        return self.head.forward_logits(
            out_fwd, out_bwd, inputs["stream_fwd"], inputs["stream_bwd"]
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute 48-slot probabilities ``(batch, 48)``.

        Args:
            inputs: DDCL input dictionary.

        Returns:
            Float32 probabilities ``(batch, 48)``.
        """
        return torch.sigmoid(self.forward_logits(inputs))

    def model_config(self) -> dict[str, object]:
        """Return constructor kwargs for checkpoint serialization.

        Returns:
            JSON-serializable constructor kwargs mapping.
        """
        return {
            "arch": "mert",
            "memlen": self.memlen,
            "n_frames": self.n_frames,
            "feature_dim": self.feature_dim,
            "proj_dim": self.proj_dim,
            "conv_filters": self.conv_filters,
            "noise_std": self.noise_std,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "dense_sizes": list(self.dense_sizes),
        }


class BinaryFocalWithLogitsLoss(nn.Module):
    """Binary focal cross-entropy with logits.

    Args:
        gamma: Focusing parameter (default: ``2.0``).
        alpha: Class balance weight when ``apply_class_balancing`` is True.
        apply_class_balancing: Whether to apply alpha balancing (default False).
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float = 0.25,
        apply_class_balancing: bool = False,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.apply_class_balancing = bool(apply_class_balancing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute scalar mean binary focal loss from float32 logits.

        Args:
            logits: Raw logits ``(batch, 48)``.
            targets: Binary targets ``(batch, 48)``.

        Returns:
            Scalar float32 loss tensor.
        """
        logits_f = logits.float()
        targets_f = targets.float()
        bce = nn_func.binary_cross_entropy_with_logits(
            logits_f, targets_f, reduction="none"
        )
        probs = torch.sigmoid(logits_f)
        p_t = targets_f * probs + (1.0 - targets_f) * (1.0 - probs)
        modulating = torch.pow(torch.clamp(1.0 - p_t, min=1e-7), self.gamma)
        loss = modulating * bce
        if self.apply_class_balancing:
            alpha_t = targets_f * self.alpha + (1.0 - targets_f) * (1.0 - self.alpha)
            loss = alpha_t * loss
        return loss.mean()


def build_convlstm_placement_model(
    *,
    memlen: int = constants.MEMLEN,
    n_frames: int = constants.N_FRAMES_PER_BEAT,
    lstm_units: int = constants.LSTM_UNITS,
    dropout_rate: float = constants.DROPOUT_RATE,
    dense_sizes: tuple[int, ...] = constants.DENSE_SIZES,
) -> DdclConvLstmPlacementModel:
    """Build the PyTorch DDCL log-mel ConvLSTM model.

    Args:
        memlen: Context beats besides the current one.
        n_frames: Audio frames per beat.
        lstm_units: LSTM width.
        dropout_rate: Dropout rate.
        dense_sizes: Fully-connected hidden widths.

    Returns:
        Instantiated :class:`DdclConvLstmPlacementModel`.
    """
    return DdclConvLstmPlacementModel(
        memlen=memlen,
        n_frames=n_frames,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_sizes=tuple(dense_sizes),
    )


def build_mert_placement_model(
    *,
    memlen: int = constants.MEMLEN,
    n_frames: int = constants.MERT_FRAMES_PER_BEAT,
    feature_dim: int = constants.MERT_FEATURE_DIM,
    proj_dim: int = constants.MERT_PROJ_DIM,
    conv_filters: int = constants.MERT_CONV_FILTERS,
    noise_std: float = 0.0,
    lstm_units: int = constants.LSTM_UNITS,
    dropout_rate: float = constants.DROPOUT_RATE,
    dense_sizes: tuple[int, ...] = constants.DENSE_SIZES,
) -> DdclMertPlacementModel:
    """Build the PyTorch DDCL beat-aligned MERT model.

    Args:
        memlen: Context beats besides the current one.
        n_frames: Sub-beat MERT frames per beat.
        feature_dim: MERT embedding width.
        proj_dim: Per-frame projection width.
        conv_filters: Sub-beat conv filters.
        noise_std: Train-time Gaussian noise stddev.
        lstm_units: LSTM width.
        dropout_rate: Dropout rate.
        dense_sizes: Fully-connected hidden widths.

    Returns:
        Instantiated :class:`DdclMertPlacementModel`.
    """
    return DdclMertPlacementModel(
        memlen=memlen,
        n_frames=n_frames,
        feature_dim=feature_dim,
        proj_dim=proj_dim,
        conv_filters=conv_filters,
        noise_std=noise_std,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_sizes=tuple(dense_sizes),
    )
