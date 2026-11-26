"""PyTorch hierarchical transformer placement from ITGPT (`omalley2026itgpt`).

Architecture matches :mod:`stepcovnet.itgpt.models` and upstream
https://github.com/miguelomalley/ITGPT/blob/main/onset.py, while using
:func:`torch.nn.functional.scaled_dot_product_attention` (FlashAttention /
cuDNN SDPA on CUDA) and Keras-compatible default weight initialization.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.itgpt import constants


def grid_importance_weights(
    n_slots: int = constants.N_SLOTS,
    *,
    weight_24th: float = constants.GRID_WEIGHT_24TH,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the ITGPT 48-slot BCE weights.

    Args:
        n_slots: Slot count (paper: 48).
        weight_24th: Loss weight for 24th/triplet slots (paper: 1.0).
        device: Optional tensor device.

    Returns:
        Float32 tensor of shape ``(n_slots,)``.
    """
    weights = [constants.GRID_WEIGHT_MICRO] * n_slots
    for index in constants.INDICES_16TH:
        weights[index] = constants.GRID_WEIGHT_16TH
    for index in constants.INDICES_24TH:
        weights[index] = float(weight_24th)
    for index in constants.INDICES_32ND:
        weights[index] = 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _init_linear_or_conv(module: nn.Module) -> None:
    """Apply Glorot/Xavier uniform + zero bias matching Keras defaults.

    Args:
        module: Layer to initialize if it is ``Linear``, ``Conv1d``, or ``Conv2d``.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ItgptRmsNorm(nn.Module):
    """RMSNorm matching ITGPT ``nn.RMSNorm`` and :class:`stepcovnet.itgpt.models.ItgptRmsNorm`.

    Args:
        d_model: Feature width along the last dimension.
        epsilon: Stabilizer added inside the square root.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.epsilon = float(epsilon)
        self.scale = nn.Parameter(torch.ones(self.d_model, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension.

        Args:
            inputs: Activation tensor ``(..., d_model)``.

        Returns:
            Normalized tensor with the same shape and dtype as ``inputs``.
        """
        in_dtype = inputs.dtype
        x_fp32 = inputs.float()
        rms = torch.sqrt(
            torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.epsilon
        )
        out = (x_fp32 / rms) * self.scale.float()
        return out.to(in_dtype)


class ItgptFFN(nn.Module):
    """GELU MLP with residual dropout (ITGPT ``FFN``).

    Args:
        d_model: Model width.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden = 4 * d_model
        self.dense1 = nn.Linear(d_model, hidden)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden, d_model)
        self.drop2 = nn.Dropout(dropout)
        _init_linear_or_conv(self.dense1)
        _init_linear_or_conv(self.dense2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the GELU MLP.

        Args:
            inputs: ``(..., d_model)``.

        Returns:
            Same shape as ``inputs``.
        """
        hidden = self.drop1(nn_func.gelu(self.dense1(inputs)))
        return self.drop2(self.dense2(hidden))


class ItgptMultiHeadSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention backed by PyTorch SDPA.

    Uses separate Q/K/V/O linear projections to match Keras
    ``MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)``
    1-to-1 for exact weight parity while enabling FlashAttention / cuDNN SDPA.

    Args:
        d_model: Model width.
        n_heads: Attention head count dividing ``d_model``.
        dropout: Attention weight dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _init_linear_or_conv(proj)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional self-attention.

        Args:
            inputs: ``(batch, seq_len, d_model)``.

        Returns:
            Tensor with shape ``(batch, seq_len, d_model)``.
        """
        batch_size, seq_len, _ = inputs.shape
        q = (
            self.q_proj(inputs)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(inputs)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(inputs)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        dropout_p = self.dropout if self.training else 0.0
        attn_out = nn_func.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=False,
        )
        merged = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(merged)


class ItgptEncoderBlock(nn.Module):
    """Pre-norm bidirectional self-attention block (ITGPT ``EncoderBlock``).

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = ItgptRmsNorm(d_model)
        self.attn = ItgptMultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = ItgptRmsNorm(d_model)
        self.ffn = ItgptFFN(d_model, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention + FFN.

        Args:
            inputs: ``(batch, time, d_model)``.

        Returns:
            Same shape as ``inputs``.
        """
        hidden = inputs + self.attn(self.ln1(inputs))
        return hidden + self.ffn(self.ln2(hidden))


class ItgptNormlessEncoderBlock(nn.Module):
    """ITGPT ``NormlessEncoderBlock`` (attention then RMSNorm FFN).

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = ItgptMultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = ItgptFFN(d_model, dropout)
        self.norm = ItgptRmsNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply attention then RMSNorm + FFN.

        Args:
            inputs: ``(batch, time, d_model)``.

        Returns:
            Same shape as ``inputs``.
        """
        attended = inputs + self.attn(inputs)
        return attended + self.ffn(self.norm(attended))


class ItgptGatedCompression(nn.Module):
    """ITGPT ``GatedCompression`` (value * sigmoid(gate) + RMSNorm).

    Args:
        in_features: Incoming feature width.
        out_features: Compressed output width.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.proj = nn.Linear(self.in_features, self.out_features * 2)
        self.norm = ItgptRmsNorm(self.out_features)
        _init_linear_or_conv(self.proj)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compress the last axis.

        Args:
            inputs: ``(..., in_features)``.

        Returns:
            ``(..., out_features)``.
        """
        value, gate = self.proj(inputs).chunk(2, dim=-1)
        return self.norm(value * torch.sigmoid(gate))


class ItgptHierarchicalAttnBlock(nn.Module):
    """ITGPT ``HierarchicalAttnBlock`` over ``beats_per_chunk`` groups.

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        dropout: Dropout rate.
        frames_per_beat: Frames kept per beat (24 or 6).
        beats_per_chunk: Beats attending together in one chunk.
        max_chunks: Chunk positional embedding table size.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        frames_per_beat: int,
        beats_per_chunk: int,
        max_chunks: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.frames_per_beat = int(frames_per_beat)
        self.beats_per_chunk = int(beats_per_chunk)
        total_frames = self.beats_per_chunk * self.frames_per_beat
        self.pos_emb = nn.Parameter(
            torch.empty(1, total_frames, self.d_model, dtype=torch.float32)
        )
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.chunk_pos_emb = nn.Embedding(max_chunks, self.d_model)
        nn.init.trunc_normal_(
            self.chunk_pos_emb.weight, mean=0.0, std=0.02, a=-0.04, b=0.04
        )
        self.attn_block = ItgptNormlessEncoderBlock(self.d_model, n_heads, dropout)
        self.norm1 = ItgptRmsNorm(self.d_model)
        self.norm2 = ItgptRmsNorm(self.d_model)

    def forward(
        self,
        h_beat: torch.Tensor,
        *,
        batch_size: int,
        n_beats: int,
    ) -> torch.Tensor:
        """Chunk frames and apply intra-chunk attention.

        Args:
            h_beat: ``(B * T, frames_per_beat, d_model)``.
            batch_size: Batch size ``B``.
            n_beats: Padded beat count ``T`` (multiple of ``beats_per_chunk``).

        Returns:
            Tensor with shape ``(B * T, frames_per_beat, d_model)``.
        """
        hidden = self.norm1(h_beat)
        chunk = self.beats_per_chunk
        frames = self.frames_per_beat
        num_chunks = n_beats // chunk
        n_chunk_rows = batch_size * num_chunks
        n_beat_rows = batch_size * n_beats
        hidden = hidden.reshape(
            batch_size, num_chunks, chunk * frames, self.d_model
        ).reshape(n_chunk_rows, chunk * frames, self.d_model)
        hidden = hidden + self.pos_emb.to(hidden.dtype)
        chunk_idx = torch.arange(num_chunks, dtype=torch.long, device=hidden.device)
        chunk_emb = self.chunk_pos_emb(chunk_idx).to(hidden.dtype)
        chunk_emb = (
            chunk_emb.unsqueeze(0)
            .expand(batch_size, num_chunks, self.d_model)
            .reshape(n_chunk_rows, 1, self.d_model)
        )
        hidden = hidden + chunk_emb
        hidden = self.attn_block(hidden)
        hidden = hidden.reshape(
            batch_size, num_chunks, chunk, frames, self.d_model
        ).reshape(n_beat_rows, frames, self.d_model)
        return self.norm2(hidden)


class ItgptBeatCNNEncoder(nn.Module):
    """ITGPT ``BeatCNNEncoder`` on ``(32, 80, 3)`` log-mel beats.

    Accepts channels-last ``(BT, 32, 80, 3)`` to match
    :class:`stepcovnet.ddcl.datasets.DdclChart` directly, permutes internally
    for PyTorch ``Conv2d``, and permutes back before flattening so ``proj0``
    feature ordering matches Keras 1-to-1.

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
        # Permute (BT, 2H, 24, 8) -> (BT, 24, 8, 2H) to match Keras flatten order.
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
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        dropout_rate: float = constants.DROPOUT_RATE,
        n_mert_layers: int = 1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.dropout_rate = float(dropout_rate)
        self.n_mert_layers = int(n_mert_layers)
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
        hidden = self.drop(nn_func.gelu(self.proj0(self.norm(hidden))))
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


class ItgptConditionFilm(nn.Module):
    """Zero-initialized FiLM modulation ``(1 + gamma) * h + beta``.

    Args:
        d_model: Feature channel width to modulate.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden = nn.Linear(self.d_model * 2, self.d_model)
        self.gamma_beta = nn.Linear(self.d_model, self.d_model * 2)
        _init_linear_or_conv(self.hidden)
        nn.init.zeros_(self.gamma_beta.weight)
        nn.init.zeros_(self.gamma_beta.bias)

    def forward(
        self,
        features: torch.Tensor,
        cond_vec: torch.Tensor,
        *,
        batch_size: int,
        n_beats: int,
        frames_per_beat: int | None = None,
    ) -> torch.Tensor:
        """Modulate beat or sub-beat features with chart-level FiLM parameters.

        Args:
            features: ``(B * T, F, d_model)`` when ``frames_per_beat`` is set,
                or ``(B, T, d_model)`` when ``frames_per_beat`` is None.
            cond_vec: Condition embedding ``(B, 2 * d_model)``.
            batch_size: Batch size ``B``.
            n_beats: Padded beat count ``T``.
            frames_per_beat: Optional sub-beat frame count ``F``.

        Returns:
            Modulated tensor with the same shape as ``features``.
        """
        params = self.gamma_beta(nn_func.gelu(self.hidden(cond_vec)))
        delta_gamma, beta = params.chunk(2, dim=-1)
        scale = 1.0 + delta_gamma
        if frames_per_beat is None:
            return features * scale.unsqueeze(1) + beta.unsqueeze(1)
        reshaped = features.reshape(batch_size, n_beats, frames_per_beat, self.d_model)
        scale_b = scale.reshape(batch_size, 1, 1, self.d_model)
        beta_b = beta.reshape(batch_size, 1, 1, self.d_model)
        modulated = reshaped * scale_b + beta_b
        return modulated.reshape(batch_size * n_beats, frames_per_beat, self.d_model)


class ItgptPlacementModel(nn.Module):
    """PyTorch ITGPT hierarchical placement network (dynamic beat length).

    Args:
        d_model: Transformer width.
        n_heads: Attention heads.
        n_enc_layers: Global encoder depth.
        cnn_hidden: Beat CNN base channels.
        dropout_rate: Dropout.
        max_beats: Positional table length.
        audio_feature_type: Beat audio frontend (``"logmel"`` or ``"mert"``).
        n_frames_per_beat: Input frames per beat (``32`` for log-mel, ``24``
            for MERT).
        feature_dim: Per-frame feature width when ``audio_feature_type == "mert"``.
        n_mert_layers: Stacked MERT layer count (``1`` disables layer mixing).
        early_film: If True, apply zero-initialized FiLM modulation from
            ``[difficulty, bpm]`` before the sub-beat and global stacks.
        cyclic_measure_pos: If True, add zero-initialized cyclic ``beat % 4``
            and ``beat % 16`` positional embeddings before ``level6``.
        continuous_beat_conv: If True, run ``beat_mert`` over the continuous
            ``(batch, beats * n_frames_per_beat, feature_dim)`` stream.

    Raises:
        ValueError: If frontend or width arguments are invalid.
    """

    def __init__(
        self,
        *,
        d_model: int = constants.D_MODEL,
        n_heads: int = constants.N_HEADS,
        n_enc_layers: int = constants.N_ENC_LAYERS,
        cnn_hidden: int = constants.CNN_HIDDEN,
        dropout_rate: float = constants.DROPOUT_RATE,
        max_beats: int = constants.MAX_BEATS,
        audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL,
        n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
        feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
        n_mert_layers: int = 1,
        early_film: bool = False,
        cyclic_measure_pos: bool = False,
        continuous_beat_conv: bool = False,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads != 0:
            raise ValueError(f"n_heads must divide d_model, got {n_heads}, {d_model}")
        if n_enc_layers < 1:
            raise ValueError(f"n_enc_layers must be at least 1, got {n_enc_layers}")
        if audio_feature_type not in ddcl_constants.AUDIO_FEATURE_TYPES:
            raise ValueError(
                "audio_feature_type must be one of "
                f"{ddcl_constants.AUDIO_FEATURE_TYPES}, got {audio_feature_type!r}"
            )
        if (
            audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT
            and n_frames_per_beat != constants.CNN_FRAMES_OUT
        ):
            raise ValueError(
                "MERT ITGPT requires n_frames_per_beat == "
                f"{constants.CNN_FRAMES_OUT}, got {n_frames_per_beat}"
            )
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if n_mert_layers < 1:
            raise ValueError(f"n_mert_layers must be positive, got {n_mert_layers}")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_enc_layers = int(n_enc_layers)
        self.cnn_hidden = int(cnn_hidden)
        self.dropout_rate = float(dropout_rate)
        self.max_beats = int(max_beats)
        self.audio_feature_type = str(audio_feature_type)
        self.n_frames_per_beat = int(n_frames_per_beat)
        self.feature_dim = int(feature_dim)
        self.n_mert_layers = int(n_mert_layers)
        self.early_film = bool(early_film)
        self.cyclic_measure_pos = bool(cyclic_measure_pos)
        self.continuous_beat_conv = bool(continuous_beat_conv)

        self.beat_cnn: ItgptBeatCNNEncoder | None = None
        self.beat_mert: ItgptBeatMertEncoder | None = None
        if self.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
            self.beat_mert = ItgptBeatMertEncoder(
                self.feature_dim,
                self.d_model,
                dropout_rate=self.dropout_rate,
                n_mert_layers=self.n_mert_layers,
            )
        else:
            self.beat_cnn = ItgptBeatCNNEncoder(self.d_model, self.cnn_hidden)

        self.diff_proj = ItgptConditionProj(
            self.d_model,
            constants.MIN_DIFFICULTY,
            constants.MAX_DIFFICULTY,
        )
        self.bpm_proj = ItgptConditionProj(
            self.d_model,
            constants.MIN_BPM,
            constants.MAX_BPM,
        )

        self.early_film_subbeat: ItgptConditionFilm | None = None
        self.early_film_beat: ItgptConditionFilm | None = None
        if self.early_film:
            self.early_film_subbeat = ItgptConditionFilm(self.d_model)
            self.early_film_beat = ItgptConditionFilm(self.d_model)

        self.level1 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=1,
        )
        self.level2 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=4,
        )
        self.level3 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=16,
        )
        self.level4 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=32,
        )
        self.frame_compress = ItgptGatedCompression(4 * self.d_model, self.d_model)
        self.level5 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=6,
            beats_per_chunk=64,
        )
        self.beat_compress = ItgptGatedCompression(6 * self.d_model, self.d_model)
        self.level6_pos = nn.Embedding(self.max_beats + 200, self.d_model)
        nn.init.normal_(self.level6_pos.weight, mean=0.0, std=0.02)

        self.bar_pos_emb: nn.Embedding | None = None
        self.phrase_pos_emb: nn.Embedding | None = None
        if self.cyclic_measure_pos:
            self.bar_pos_emb = nn.Embedding(4, self.d_model)
            self.phrase_pos_emb = nn.Embedding(16, self.d_model)
            nn.init.zeros_(self.bar_pos_emb.weight)
            nn.init.zeros_(self.phrase_pos_emb.weight)

        self.level6 = nn.ModuleList(
            [
                ItgptEncoderBlock(self.d_model, self.n_heads, self.dropout_rate)
                for _ in range(self.n_enc_layers)
            ]
        )
        self.conv_head = ItgptConvHead(self.d_model, self.dropout_rate)
        self.onset_head = nn.Linear(self.d_model, constants.N_SLOTS)
        _init_linear_or_conv(self.onset_head)

    def forward_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the hierarchical encoder and return float32 slot logits.

        Args:
            inputs: Dict with ``audio``, ``bpm``, and ``difficulty`` tensors.

        Returns:
            Float32 logits tensor of shape ``(batch, beats, 48)``.

        Raises:
            RuntimeError: If the active beat encoder was not initialized.
        """
        audio = inputs["audio"]
        batch_size = audio.shape[0]
        n_beats = audio.shape[1]
        n_beat_rows = batch_size * n_beats

        if self.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
            if self.beat_mert is None:
                raise RuntimeError("beat_mert encoder is not initialized")
            tail = (
                (self.n_mert_layers, self.feature_dim)
                if self.n_mert_layers > 1
                else (self.feature_dim,)
            )
            if self.continuous_beat_conv:
                mert_in = audio.reshape(
                    batch_size, n_beats * self.n_frames_per_beat, *tail
                )
                hidden = self.beat_mert(mert_in)
                hidden = hidden.reshape(
                    n_beat_rows, self.n_frames_per_beat, self.d_model
                )
            else:
                mert_in = audio.reshape(n_beat_rows, self.n_frames_per_beat, *tail)
                hidden = self.beat_mert(mert_in)
        else:
            if self.beat_cnn is None:
                raise RuntimeError("beat_cnn encoder is not initialized")
            conv_in = audio.reshape(
                n_beat_rows,
                constants.N_FRAMES_PER_BEAT,
                constants.N_MELS,
                constants.N_CHANNELS,
            )
            hidden = self.beat_cnn(conv_in)

        diff_tok = self.diff_proj(inputs["difficulty"])
        bpm_tok = self.bpm_proj(inputs["bpm"])
        cond_vec = torch.cat(
            [diff_tok.squeeze(1), bpm_tok.squeeze(1)],
            dim=-1,
        )
        if self.early_film_subbeat is not None:
            hidden = self.early_film_subbeat(
                hidden,
                cond_vec,
                batch_size=batch_size,
                n_beats=n_beats,
                frames_per_beat=constants.CNN_FRAMES_OUT,
            )

        hidden = self.level1(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = self.level2(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = self.level3(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = self.level4(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = hidden.reshape(n_beat_rows, 6, 4 * self.d_model)
        hidden = self.frame_compress(hidden)
        hidden = self.level5(hidden, batch_size=batch_size, n_beats=n_beats)
        global_h = self.beat_compress(hidden.reshape(n_beat_rows, 6 * self.d_model))
        global_h = global_h.reshape(batch_size, n_beats, self.d_model)

        if self.early_film_beat is not None:
            global_h = self.early_film_beat(
                global_h,
                cond_vec,
                batch_size=batch_size,
                n_beats=n_beats,
            )

        beat_idx = torch.arange(n_beats, dtype=torch.long, device=global_h.device)
        pos = self.level6_pos(beat_idx)
        if self.bar_pos_emb is not None and self.phrase_pos_emb is not None:
            pos = (
                pos
                + self.bar_pos_emb(beat_idx % 4)
                + self.phrase_pos_emb(beat_idx % 16)
            )
        global_h = global_h + pos.to(global_h.dtype).unsqueeze(0)
        global_h = torch.cat(
            [diff_tok.to(global_h.dtype), bpm_tok.to(global_h.dtype), global_h],
            dim=1,
        )
        for block in self.level6:
            global_h = block(global_h)
        global_h = global_h[:, 2:, :]
        global_h = self.conv_head(global_h)
        # Keep onset head in fp32 for numerical stability under autocast.
        return nn_func.linear(
            global_h.float(),
            self.onset_head.weight.float(),
            self.onset_head.bias.float() if self.onset_head.bias is not None else None,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the hierarchical encoder and return float32 slot probabilities.

        Args:
            inputs: Dict with ``audio``, ``bpm``, and ``difficulty`` tensors.

        Returns:
            Float32 probabilities tensor of shape ``(batch, beats, 48)``.
        """
        return torch.sigmoid(self.forward_logits(inputs))

    def model_config(self) -> dict[str, object]:
        """Return constructor kwargs for checkpoint serialization.

        Returns:
            JSON-serializable constructor kwargs mapping.
        """
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_enc_layers": self.n_enc_layers,
            "cnn_hidden": self.cnn_hidden,
            "dropout_rate": self.dropout_rate,
            "max_beats": self.max_beats,
            "audio_feature_type": self.audio_feature_type,
            "n_frames_per_beat": self.n_frames_per_beat,
            "feature_dim": self.feature_dim,
            "n_mert_layers": self.n_mert_layers,
            "early_film": self.early_film,
            "cyclic_measure_pos": self.cyclic_measure_pos,
            "continuous_beat_conv": self.continuous_beat_conv,
        }


def build_itgpt_placement_model(
    *,
    d_model: int = constants.D_MODEL,
    n_heads: int = constants.N_HEADS,
    n_enc_layers: int = constants.N_ENC_LAYERS,
    cnn_hidden: int = constants.CNN_HIDDEN,
    dropout_rate: float = constants.DROPOUT_RATE,
    max_beats: int = constants.MAX_BEATS,
    audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL,
    n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
    feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
    n_mert_layers: int = 1,
    early_film: bool = False,
    cyclic_measure_pos: bool = False,
    continuous_beat_conv: bool = False,
) -> ItgptPlacementModel:
    """Build the PyTorch ITGPT placement network.

    Args:
        d_model: Transformer width.
        n_heads: Attention heads.
        n_enc_layers: Global encoder depth.
        cnn_hidden: Beat CNN base channels.
        dropout_rate: Dropout.
        max_beats: Maximum padded beat length.
        audio_feature_type: Beat audio frontend (``"logmel"`` or ``"mert"``).
        n_frames_per_beat: Input frames per beat.
        feature_dim: Per-frame width when ``audio_feature_type == "mert"``.
        n_mert_layers: Stacked MERT layer count (``1`` disables layer mixing).
        early_film: If True, apply zero-initialized FiLM conditioning.
        cyclic_measure_pos: If True, add cyclic ``beat % 4`` and ``beat % 16``
            positional embeddings.
        continuous_beat_conv: If True, run ``beat_mert`` across beat boundaries.

    Returns:
        Instantiated :class:`ItgptPlacementModel`.
    """
    return ItgptPlacementModel(
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        cnn_hidden=cnn_hidden,
        dropout_rate=dropout_rate,
        max_beats=max_beats,
        audio_feature_type=audio_feature_type,
        n_frames_per_beat=n_frames_per_beat,
        feature_dim=feature_dim,
        n_mert_layers=n_mert_layers,
        early_film=early_film,
        cyclic_measure_pos=cyclic_measure_pos,
        continuous_beat_conv=continuous_beat_conv,
    )


class ItgptPlacementLoss(nn.Module):
    """Masked grid-weighted slot BCE-with-logits with optional focal modulation.

    Matches :class:`stepcovnet.itgpt.trainers.ItgptPlacementLoss` while taking
    raw float32 logits for numerical stability.

    Args:
        focal_gamma: Non-negative focal exponent ``(1 - p_t)^gamma``
            (``0.0`` disables focal modulation).
        grid_weight_24th: Loss weight for 24th/triplet slots (default ``1.0``).
        normalize_by_valid_beats: If True, divide masked beat losses by the
            number of real beats (``mask.sum()``). If False, divide by the
            total padded beat count (matching Keras ``sum_over_batch_size``).
    """

    def __init__(
        self,
        focal_gamma: float = 0.0,
        grid_weight_24th: float = constants.GRID_WEIGHT_24TH,
        *,
        normalize_by_valid_beats: bool = False,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.grid_weight_24th = float(grid_weight_24th)
        self.normalize_by_valid_beats = bool(normalize_by_valid_beats)
        self.register_buffer(
            "grid_weights",
            grid_importance_weights(weight_24th=self.grid_weight_24th),
            persistent=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        beat_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar masked slot-weighted BCE loss.

        Args:
            logits: Raw logits ``(batch, beats, 48)``.
            targets: Binary slot targets ``(batch, beats, 48)``.
            beat_mask: Real-beat mask ``(batch, beats)`` with 1 on real beats.

        Returns:
            Scalar float32 loss tensor.
        """
        logits_fp32 = logits.float()
        targets_fp32 = targets.float()
        mask_fp32 = beat_mask.float()
        per_slot = nn_func.binary_cross_entropy_with_logits(
            logits_fp32,
            targets_fp32,
            reduction="none",
        )
        if self.focal_gamma > 0.0:
            probs = torch.sigmoid(logits_fp32)
            p_t = targets_fp32 * probs + (1.0 - targets_fp32) * (1.0 - probs)
            per_slot = per_slot * torch.pow(
                torch.clamp(1.0 - p_t, min=1e-7), self.focal_gamma
            )
        grid_weights: torch.Tensor = self.grid_weights  # type: ignore[assignment]
        per_beat = torch.mean(per_slot * grid_weights.to(logits_fp32.device), dim=-1)
        masked = per_beat * mask_fp32
        if self.normalize_by_valid_beats:
            denom = torch.clamp(mask_fp32.sum(), min=1.0)
            return masked.sum() / denom
        return masked.mean()
