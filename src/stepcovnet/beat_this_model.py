"""Self-contained pure-PyTorch inference implementation of Beat This! (ISMIR 2024).

Implements the Beat This! neural beat and downbeat tracker architecture,
log-mel spectrogram frontend, chunked sliding-window inference, and minimal
peak-picking postprocessor using only PyTorch, NumPy, and librosa (no external
``beat_this``, ``einops``, ``rotary-embedding-torch``, ``soxr``, or
``torchaudio`` dependencies).

Pretrained weights (e.g. ``final0.ckpt``) are downloaded on demand via
:func:`torch.hub.load_state_dict_from_url` and cached in the standard PyTorch
hub cache directory.

Attribution & Licenses:
    - Beat This! architecture, preprocessing, chunking, and postprocessing:
      Francesco Foscarin, Jan Schlueter, and Gerhard Widmer, *"Beat This!
      Accurate Beat Tracking Without DBN Postprocessing"*, ISMIR 2024
      (https://github.com/CPJKU/beat_this, Copyright 2024 Institute of
      Computational Perception, JKU Linz, Austria, MIT License).
    - Rotary transformer blocks adapted from Phil Wang's BS-RoFormer
      (https://github.com/lucidrains/BS-RoFormer, MIT License).
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import pathlib
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nn_func

CHECKPOINT_BASE_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
DEFAULT_CHECKPOINT = "final0"
BEAT_THIS_SR = 22050
BEAT_THIS_FPS = 50
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_BORDER_SIZE = 6


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent even/odd feature pairs ``(x0, x1) -> (-x1, x0)``.

    Args:
        x: Input tensor whose last dimension is even.

    Returns:
        Rotated tensor of the same shape as ``x``.
    """
    x_pairs = x.unflatten(-1, (-1, 2))
    x1, x2 = x_pairs.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class _RotaryEmbedding(nn.Module):
    """Minimal rotary positional embedding matching ``rotary-embedding-torch``.

    Args:
        dim: Head feature dimension (must be even).
        theta: Base frequency scaling constant (default ``10000.0``).
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            float(theta) ** (torch.arange(0, dim, 2, dtype=torch.float32) / float(dim))
        )
        self.freqs = nn.Parameter(inv_freq, requires_grad=False)

    def rotate_queries_or_keys(self, t: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings along sequence dimension ``-2``.

        Args:
            t: Query or key tensor of shape ``(..., seq_len, dim)``.

        Returns:
            Tensor of the same shape and dtype as ``t`` with rotary embeddings
            applied.
        """
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=self.freqs.dtype)
        angles = torch.outer(positions, self.freqs).repeat_interleave(2, dim=-1)
        cos = angles.cos()
        sin = angles.sin()
        t_fp32 = t.float()
        rotated = (t_fp32 * cos) + (_rotate_half(t_fp32) * sin)
        return rotated.to(t.dtype)


class _Permute(nn.Module):
    """Permute tensor dimensions inside an :class:`nn.Sequential` container.

    Args:
        dims: Target dimension ordering.
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x.permute(*self.dims)``.

        Args:
            x: Input tensor.

        Returns:
            Permuted tensor.
        """
        return x.permute(*self.dims)


class _Unsqueeze(nn.Module):
    """Insert a singleton dimension inside an :class:`nn.Sequential` container.

    Args:
        dim: Dimension index to insert.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x.unsqueeze(self.dim)``.

        Args:
            x: Input tensor.

        Returns:
            Unsqueezed tensor.
        """
        return x.unsqueeze(self.dim)


class _ConcatFreqChannels(nn.Module):
    """Reshape ``(batch, channels, freqs, time)`` to ``(batch, time, channels * freqs)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten channels and frequency bins per time step.

        Args:
            x: 4D feature map of shape ``(B, C, F, T)``.

        Returns:
            3D sequence tensor of shape ``(B, T, C * F)``.
        """
        batch, channels, freqs, time = x.shape
        return x.permute(0, 3, 1, 2).reshape(batch, time, channels * freqs)


class RMSNorm(nn.Module):
    """L2-normalized RMSNorm matching Beat This! / BS-RoFormer weights.

    Args:
        size: Feature dimension size along ``dim``.
        dim: Negative axis over which to normalize (default ``-1``).
    """

    def __init__(self, size: int, dim: int = -1) -> None:
        super().__init__()
        if dim >= 0:
            raise ValueError(f"dim must be negative, got {dim}")
        self.scale = float(size) ** 0.5
        self.gamma = nn.Parameter(torch.ones((size,) + (1,) * (abs(dim) - 1)))
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` along ``self.dim`` and scale by ``self.gamma``.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        return nn_func.normalize(x, dim=self.dim) * self.scale * self.gamma


class FeedForward(nn.Module):
    """Pre-norm GELU feed-forward network matching BS-RoFormer state keys.

    Args:
        dim: Input feature width.
        mult: Hidden-layer expansion multiplier.
        dropout: Dropout probability.
        dim_out: Output feature width (defaults to ``dim``).
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
        dim_out: int | None = None,
    ) -> None:
        super().__init__()
        out_dim = dim if dim_out is None else dim_out
        dim_inner = int(dim * mult)
        self.activation = nn.GELU()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_inner, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., dim_out)``.
        """
        return self.net(x)


class Attention(nn.Module):
    """Multi-head gated rotary self-attention matching Beat This! weights.

    Args:
        dim: Model width.
        heads: Number of attention heads.
        dim_head: Feature width per head.
        dropout: Attention and output dropout probability.
        rotary_embed: Optional shared :class:`_RotaryEmbedding` instance.
        gating: Whether to apply sigmoid per-head output gating.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotary_embed: _RotaryEmbedding | None = None,
        gating: bool = True,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout = float(dropout)
        dim_inner = self.heads * self.dim_head

        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates: nn.Linear | None = nn.Linear(dim, self.heads) if gating else None
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run gated rotary self-attention over ``x``.

        Args:
            x: Input sequence tensor of shape ``(batch, seq_len, dim)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, dim)``.
        """
        x = self.norm(x)
        batch, seq_len, _ = x.shape
        qkv = (
            self.to_qkv(x)
            .reshape(batch, seq_len, 3, self.heads, self.dim_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = nn_func.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )

        if self.to_gates is not None:
            gates = self.to_gates(x).transpose(1, 2).unsqueeze(-1).sigmoid()
            out = out * gates

        out = out.transpose(1, 2).reshape(batch, seq_len, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
    """Stack of rotary attention and feed-forward residual blocks.

    Args:
        dim: Transformer hidden dimension.
        depth: Number of layers.
        dim_head: Dimension per attention head.
        heads: Number of attention heads.
        attn_dropout: Attention dropout probability.
        ff_dropout: Feed-forward dropout probability.
        ff_mult: Feed-forward expansion multiplier.
        norm_output: Whether to apply a trailing :class:`RMSNorm`.
        rotary_embed: Shared :class:`_RotaryEmbedding` instance.
        gating: Whether to enable per-head attention gating.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 32,
        heads: int = 16,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        ff_mult: int = 4,
        norm_output: bool = True,
        rotary_embed: _RotaryEmbedding | None = None,
        gating: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            gating=gating,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm: nn.Module = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transformer layers sequentially.

        Args:
            x: Input sequence tensor of shape ``(batch, seq_len, dim)``.

        Returns:
            Transformed sequence tensor of shape ``(batch, seq_len, dim)``.
        """
        for layer in self.layers:
            attn, ff = layer  # type: ignore[misc]
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class PartialFTTransformer(nn.Module):
    """Alternating frequency-axis and time-axis rotary transformer block.

    Attribute names ``attnF``, ``ffF``, ``attnT``, and ``ffT`` match upstream
    Beat This! checkpoint keys verbatim.

    Args:
        dim: Channel dimension.
        dim_head: Dimension per attention head.
        n_head: Number of attention heads (must equal ``dim // dim_head``).
        rotary_embed: Shared :class:`_RotaryEmbedding` instance.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed: _RotaryEmbedding,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % dim_head != 0 or dim // dim_head != n_head:
            raise ValueError(
                f"Expected dim == n_head * dim_head, got dim={dim}, "
                f"n_head={n_head}, dim_head={dim_head}"
            )
        self.attnF = Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffF = FeedForward(dim, dropout=dropout)
        self.attnT = Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffT = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-axis then time-axis attention to ``(B, C, F, T)``.

        Args:
            x: Feature tensor of shape ``(batch, channels, freqs, time)``.

        Returns:
            Transformed tensor of the same shape ``(batch, channels, freqs, time)``.
        """
        batch, channels, freqs, time = x.shape
        x = x.permute(0, 3, 2, 1).reshape(batch * time, freqs, channels)
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        x = (
            x.reshape(batch, time, freqs, channels)
            .permute(0, 2, 1, 3)
            .reshape(batch * freqs, time, channels)
        )
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        return x.reshape(batch, freqs, time, channels).permute(0, 3, 1, 2)


class SumHead(nn.Module):
    """Beat + downbeat logit head where ``beat_logit = raw_beat + downbeat``.

    Args:
        input_dim: Transformer output feature width.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute framewise beat and downbeat logits.

        Args:
            x: Sequence tensor of shape ``(batch, time, input_dim)``.

        Returns:
            Dictionary with ``"beat"`` and ``"downbeat"`` tensors of shape
            ``(batch, time)``.
        """
        beat_downbeat = self.beat_downbeat_lin(x)
        beat = beat_downbeat[..., 0]
        downbeat = beat_downbeat[..., 1]
        if hasattr(
            torch.amp, "is_autocast_available"
        ) and not torch.amp.is_autocast_available(beat.device.type):
            disable_autocast: contextlib.AbstractContextManager[Any] = (
                contextlib.nullcontext()
            )
        else:
            disable_autocast = torch.autocast(beat.device.type, enabled=False)
        with disable_autocast:
            beat = beat.float() + downbeat.float()
        return {"beat": beat, "downbeat": downbeat}


class Head(nn.Module):
    """Independent beat and downbeat linear projection head.

    Args:
        input_dim: Transformer output feature width.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.beat_downbeat_lin = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute independent framewise beat and downbeat logits.

        Args:
            x: Sequence tensor of shape ``(batch, time, input_dim)``.

        Returns:
            Dictionary with ``"beat"`` and ``"downbeat"`` tensors of shape
            ``(batch, time)``.
        """
        beat_downbeat = self.beat_downbeat_lin(x)
        return {"beat": beat_downbeat[..., 0], "downbeat": beat_downbeat[..., 1]}


class BeatThis(nn.Module):
    """Beat This! neural beat and downbeat tracker (Foscarin et al., 2024).

    Args:
        spect_dim: Input mel-spectrogram bin count (default ``128``).
        transformer_dim: Hidden width of the main transformer stack.
        ff_mult: Feed-forward expansion factor.
        n_layers: Number of main transformer layers.
        head_dim: Per-head dimension shared across frontend and transformer.
        stem_dim: Output channel count of the stem convolution.
        dropout: Optional mapping with ``"frontend"`` and ``"transformer"``
            dropout rates.
        sum_head: If True, use :class:`SumHead`; else :class:`Head`.
        partial_transformers: If True, include alternating F/T partial
            transformers inside each frontend block.
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        dropout: dict[str, float] | None = None,
        sum_head: bool = True,
        partial_transformers: bool = True,
    ) -> None:
        super().__init__()
        dropout_cfg = (
            {"frontend": 0.1, "transformer": 0.2} if dropout is None else dict(dropout)
        )
        if transformer_dim % head_dim != 0:
            raise ValueError(
                f"transformer_dim ({transformer_dim}) must be divisible by "
                f"head_dim ({head_dim})"
            )
        rotary_embed = _RotaryEmbedding(head_dim)

        stem = self.make_stem(spect_dim, stem_dim)
        freq_dim = spect_dim // 4

        frontend_blocks_list: list[nn.Module] = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks_list.append(
                self.make_frontend_block(
                    dim,
                    dim * 2,
                    partial_transformers=partial_transformers,
                    head_dim=head_dim,
                    rotary_embed=rotary_embed,
                    dropout=float(dropout_cfg.get("frontend", 0.1)),
                )
            )
            dim *= 2
            freq_dim //= 2
        frontend_blocks = nn.Sequential(*frontend_blocks_list)
        concat = _ConcatFreqChannels()
        linear = nn.Linear(dim * freq_dim, transformer_dim)
        self.frontend = nn.Sequential(
            collections.OrderedDict(
                stem=stem,
                blocks=frontend_blocks,
                concat=concat,
                linear=linear,
            )
        )

        tf_dropout = float(dropout_cfg.get("transformer", 0.2))
        self.transformer_blocks = Transformer(
            dim=transformer_dim,
            depth=n_layers,
            heads=transformer_dim // head_dim,
            attn_dropout=tf_dropout,
            ff_dropout=tf_dropout,
            rotary_embed=rotary_embed,
            ff_mult=ff_mult,
            dim_head=head_dim,
            norm_output=True,
        )
        self.task_heads: nn.Module = (
            SumHead(transformer_dim) if sum_head else Head(transformer_dim)
        )

    @staticmethod
    def make_stem(spect_dim: int, stem_dim: int) -> nn.Sequential:
        """Construct the spectrogram stem (BatchNorm1d + stride-4 Conv2d).

        Args:
            spect_dim: Number of mel frequency bins.
            stem_dim: Output channel count.

        Returns:
            Sequential stem module matching upstream checkpoint keys.
        """
        return nn.Sequential(
            collections.OrderedDict(
                rearrange_tf=_Permute(0, 2, 1),
                bn1d=nn.BatchNorm1d(spect_dim),
                add_channel=_Unsqueeze(1),
                conv2d=nn.Conv2d(
                    in_channels=1,
                    out_channels=stem_dim,
                    kernel_size=(4, 3),
                    stride=(4, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                bn2d=nn.BatchNorm2d(stem_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def make_frontend_block(
        in_dim: int,
        out_dim: int,
        partial_transformers: bool = True,
        head_dim: int | None = 32,
        rotary_embed: _RotaryEmbedding | None = None,
        dropout: float = 0.1,
    ) -> nn.Sequential:
        """Construct one frontend partial-transformer + stride-2 conv block.

        Args:
            in_dim: Input channel count.
            out_dim: Output channel count.
            partial_transformers: Whether to prepend :class:`PartialFTTransformer`.
            head_dim: Attention head dimension.
            rotary_embed: Shared rotary embedding instance.
            dropout: Dropout rate.

        Returns:
            Sequential frontend block matching upstream checkpoint keys.

        Raises:
            ValueError: If ``partial_transformers`` is True without ``head_dim``
                or ``rotary_embed``.
        """
        if partial_transformers and (head_dim is None or rotary_embed is None):
            raise ValueError(
                "Must specify head_dim and rotary_embed when partial_transformers=True"
            )
        partial_mod: nn.Module = (
            PartialFTTransformer(
                dim=in_dim,
                dim_head=int(head_dim),  # type: ignore[arg-type]
                n_head=in_dim // int(head_dim),  # type: ignore[arg-type]
                rotary_embed=rotary_embed,  # type: ignore[arg-type]
                dropout=dropout,
            )
            if partial_transformers
            else nn.Identity()
        )
        return nn.Sequential(
            collections.OrderedDict(
                partial=partial_mod,
                conv2d=nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=(2, 3),
                    stride=(2, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                norm=nn.BatchNorm2d(out_dim),
                activation=nn.GELU(),
            )
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full Beat This! model on a batch of log-mel spectrograms.

        Args:
            x: Log-mel spectrogram tensor of shape ``(batch, time, spect_dim)``.

        Returns:
            Dictionary with ``"beat"`` and ``"downbeat"`` logit tensors of shape
            ``(batch, time)``.
        """
        x = self.frontend(x)
        x = self.transformer_blocks(x)
        return self.task_heads(x)


class LogMelSpectrogram(nn.Module):
    """Pure-PyTorch equivalent of Beat This!'s ``LogMelSpect`` frontend.

    Matches ``torchaudio.transforms.MelSpectrogram(sample_rate=22050,
    n_fft=1024, hop_length=441, f_min=30, f_max=11000, n_mels=128,
    mel_scale="slaney", normalized="frame_length", power=1)`` followed by
    ``torch.log1p(1000 * mel.T)``.

    Args:
        sample_rate: Audio sample rate in Hz (default ``22050``).
        n_fft: STFT FFT and window size (default ``1024``).
        hop_length: STFT hop size in samples (default ``441``, i.e. 50 FPS).
        f_min: Lowest mel filter frequency in Hz (default ``30.0``).
        f_max: Highest mel filter frequency in Hz (default ``11000.0``).
        n_mels: Number of mel filterbank bins (default ``128``).
        log_multiplier: Multiplier inside ``log1p(log_multiplier * mel)``.
    """

    window: torch.Tensor
    mel_fb: torch.Tensor

    def __init__(
        self,
        sample_rate: int = BEAT_THIS_SR,
        n_fft: int = 1024,
        hop_length: int = 441,
        f_min: float = 30.0,
        f_max: float = 11000.0,
        n_mels: int = 128,
        log_multiplier: float = 1000.0,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.log_multiplier = float(log_multiplier)
        self.register_buffer("window", torch.hann_window(self.n_fft))
        fb = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=int(n_mels),
            fmin=float(f_min),
            fmax=float(f_max),
            htk=False,
            norm=None,
        ).astype(np.float32)
        self.register_buffer("mel_fb", torch.from_numpy(fb))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a 1D waveform tensor to a ``(time, n_mels)`` log-mel tensor.

        Args:
            waveform: 1D float audio waveform tensor at ``self.sample_rate``.

        Returns:
            Float32 log-mel spectrogram tensor of shape ``(n_frames, n_mels)``.
        """
        spec_complex = torch.stft(
            waveform.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=True,
            onesided=True,
            return_complex=True,
        )
        mag = spec_complex.abs()
        mel = self.mel_fb @ mag
        return torch.log1p(self.log_multiplier * mel.T)


def _zeropad_time(spect: torch.Tensor, left: int = 0, right: int = 0) -> torch.Tensor:
    """Zero-pad a ``(time, bins)`` spectrogram along the time dimension."""
    if left == 0 and right == 0:
        return spect
    return nn_func.pad(spect, (0, 0, left, right), mode="constant", value=0.0)


def split_piece(
    spect: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    border_size: int = DEFAULT_BORDER_SIZE,
    avoid_short_end: bool = True,
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Split a ``(time, bins)`` spectrogram into overlapping model chunks.

    Args:
        spect: Input spectrogram of shape ``(time, bins)``.
        chunk_size: Chunk length in frames (default ``1500``, i.e. 30 s).
        border_size: Border frames discarded on each side of a chunk.
        avoid_short_end: Shift the final chunk left to avoid a short tail.

    Returns:
        Tuple ``(chunks, starts)`` where ``chunks`` is a list of 2D tensors and
        ``starts`` is a 1D integer array of chunk start frame indices.
    """
    n_frames = int(spect.shape[0])
    step = chunk_size - 2 * border_size
    starts = np.arange(-border_size, n_frames - border_size, step, dtype=np.int64)
    if avoid_short_end and n_frames > step and starts.size > 0:
        starts[-1] = n_frames - (chunk_size - border_size)
    chunks = [
        _zeropad_time(
            spect[max(int(start), 0) : min(int(start) + chunk_size, n_frames)],
            left=max(0, -int(start)),
            right=max(0, min(border_size, int(start) + chunk_size - n_frames)),
        )
        for start in starts
    ]
    return chunks, starts


def aggregate_prediction(
    pred_chunks: list[dict[str, torch.Tensor]],
    starts: np.ndarray,
    full_size: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    border_size: int = DEFAULT_BORDER_SIZE,
    overlap_mode: str = "keep_first",
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch chunked beat/downbeat logit predictions back to full piece length.

    Args:
        pred_chunks: List of ``{"beat": Tensor, "downbeat": Tensor}`` chunks.
        starts: Chunk start frame indices returned by :func:`split_piece`.
        full_size: Total number of frames in the full spectrogram.
        chunk_size: Chunk length in frames.
        border_size: Number of border frames trimmed from both sides of each
            chunk prediction.
        overlap_mode: Overlap resolution strategy (``"keep_first"`` or
            ``"keep_last"``).
        device: Target tensor device.

    Returns:
        Tuple ``(beat_logits, downbeat_logits)`` of shape ``(full_size,)``.
    """
    trimmed = (
        [
            {
                "beat": chunk["beat"][border_size:-border_size],
                "downbeat": chunk["downbeat"][border_size:-border_size],
            }
            for chunk in pred_chunks
        ]
        if border_size > 0
        else pred_chunks
    )
    beat_full = torch.full((full_size,), -1000.0, device=device)
    downbeat_full = torch.full((full_size,), -1000.0, device=device)

    pairs: list[tuple[int, dict[str, torch.Tensor]]] = [
        (int(s), c) for s, c in zip(starts, trimmed, strict=True)
    ]
    if overlap_mode == "keep_first":
        pairs.reverse()

    for start, chunk in pairs:
        s_idx = start + border_size
        e_idx = start + chunk_size - border_size
        beat_full[s_idx:e_idx] = chunk["beat"]
        downbeat_full[s_idx:e_idx] = chunk["downbeat"]
    return beat_full, downbeat_full


def split_predict_aggregate(
    spect: torch.Tensor,
    model: nn.Module,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    border_size: int = DEFAULT_BORDER_SIZE,
    overlap_mode: str = "keep_first",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` on an arbitrarily long ``(time, bins)`` spectrogram.

    Args:
        spect: Input log-mel spectrogram of shape ``(time, bins)``.
        model: :class:`BeatThis` model instance.
        chunk_size: Chunk length in frames (default ``1500``).
        border_size: Border frames discarded on each side (default ``6``).
        overlap_mode: Chunk overlap policy (``"keep_first"`` or ``"keep_last"``).

    Returns:
        Tuple ``(beat_logits, downbeat_logits)`` of float32 tensors with shape
        ``(time,)``.
    """
    chunks, starts = split_piece(
        spect, chunk_size=chunk_size, border_size=border_size, avoid_short_end=True
    )
    pred_chunks: list[dict[str, torch.Tensor]] = []
    for chunk in chunks:
        out = model(chunk.unsqueeze(0))
        pred_chunks.append({"beat": out["beat"][0], "downbeat": out["downbeat"][0]})
    beat_logits, downbeat_logits = aggregate_prediction(
        pred_chunks,
        starts,
        int(spect.shape[0]),
        chunk_size=chunk_size,
        border_size=border_size,
        overlap_mode=overlap_mode,
        device=spect.device,
    )
    return beat_logits.float(), downbeat_logits.float()


def deduplicate_peaks(peaks: np.ndarray, width: int = 1) -> np.ndarray:
    """Replace adjacent peak frame clusters within ``width`` frames by their mean.

    Args:
        peaks: Sorted 1D integer array of peak frame indices.
        width: Maximum frame gap merged into a single cluster.

    Returns:
        1D float64 array of deduplicated frame indices.
    """
    result: list[float] = []
    it = iter(int(p) for p in peaks)
    try:
        current = float(next(it))
    except StopIteration:
        return np.zeros(0, dtype=np.float64)
    count = 1
    for nxt in it:
        if nxt - current <= width:
            count += 1
            current += (nxt - current) / count
        else:
            result.append(current)
            current = float(nxt)
            count = 1
    result.append(current)
    return np.asarray(result, dtype=np.float64)


def postprocess_minimal(
    beat_logits: torch.Tensor,
    downbeat_logits: torch.Tensor,
    *,
    fps: int = BEAT_THIS_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Beat This! minimal peak-picking postprocessor (no DBN required).

    Applies 7-frame (±70 ms at 50 FPS) local-maximum picking, retains peaks
    with ``logit > 0`` (probability ``> 0.5``), deduplicates adjacent peaks,
    converts frames to seconds, and snaps each downbeat to its nearest beat.

    Args:
        beat_logits: 1D framewise beat logit tensor of shape ``(time,)``.
        downbeat_logits: 1D framewise downbeat logit tensor of shape ``(time,)``.
        fps: Frame rate in frames per second (default ``50``).

    Returns:
        Tuple ``(beat_times_sec, downbeat_times_sec)`` as 1D float64 NumPy
        arrays.
    """
    packed = torch.stack([beat_logits.float(), downbeat_logits.float()], dim=0)
    pooled = nn_func.max_pool1d(packed, kernel_size=7, stride=1, padding=3)
    peaks = (packed == pooled) & (packed > 0.0)

    beat_frames = torch.nonzero(peaks[0], as_tuple=False).cpu().numpy().reshape(-1)
    downbeat_frames = torch.nonzero(peaks[1], as_tuple=False).cpu().numpy().reshape(-1)

    beat_times = deduplicate_peaks(beat_frames, width=1) / float(fps)
    downbeat_times = deduplicate_peaks(downbeat_frames, width=1) / float(fps)

    if beat_times.size > 0 and downbeat_times.size > 0:
        snapped = np.empty_like(downbeat_times)
        for idx, db_time in enumerate(downbeat_times):
            snapped[idx] = beat_times[int(np.argmin(np.abs(beat_times - db_time)))]
        downbeat_times = np.unique(snapped)
    elif downbeat_times.size > 0:
        downbeat_times = np.unique(downbeat_times)

    return beat_times, downbeat_times


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip Lightning ``model.`` and ``torch.compile`` ``_orig_mod.`` prefixes."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "").replace("_orig_mod.", "")
        cleaned[new_key] = value
    return cleaned


def load_checkpoint(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a Beat This! checkpoint from a local path, URL, or CPJKU shortname.

    Args:
        checkpoint_path: Local file path, HTTP(S) URL, or pretrained shortname
            such as ``"final0"``.
        device: Map location for loaded tensors.

    Returns:
        Loaded checkpoint dictionary containing ``"state_dict"`` and
        ``"hyper_parameters"``.

    Raises:
        ValueError: If the checkpoint cannot be loaded from disk or downloaded.
    """
    local_path = pathlib.Path(checkpoint_path)
    if local_path.is_file():
        return torch.load(local_path, map_location=device, weights_only=True)

    if checkpoint_path.startswith(("https://", "http://")):
        url = checkpoint_path
        file_name = None
    else:
        url = f"{CHECKPOINT_BASE_URL}/{checkpoint_path}.ckpt"
        file_name = f"beat_this-{checkpoint_path}.ckpt"

    try:
        return torch.hub.load_state_dict_from_url(
            url,
            file_name=file_name,
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load Beat This! checkpoint {checkpoint_path!r} from {url}"
        ) from exc


@functools.lru_cache(maxsize=4)
def _load_cached_bundle(
    checkpoint_path: str,
    device_str: str,
) -> tuple[BeatThis, LogMelSpectrogram]:
    """Load and cache a ``(BeatThis, LogMelSpectrogram)`` pair on ``device_str``."""
    target = torch.device(device_str)
    checkpoint = load_checkpoint(checkpoint_path, device=target)
    raw_hparams = checkpoint.get("hyper_parameters", {})
    valid_params = set(inspect.signature(BeatThis).parameters)
    hparams = {k: v for k, v in raw_hparams.items() if k in valid_params}
    model = BeatThis(**hparams)
    state_dict = _clean_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(target).eval()
    spect = LogMelSpectrogram().to(target).eval()
    return model, spect


def load_beat_this_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> BeatThis:
    """Load a pretrained :class:`BeatThis` model in eval mode.

    Args:
        checkpoint_path: Local path, URL, or CPJKU checkpoint name
            (default ``"final0"``).
        device: Target compute device.

    Returns:
        Pretrained :class:`BeatThis` model on ``device`` in eval mode.
    """
    model, _ = _load_cached_bundle(str(checkpoint_path), str(torch.device(device)))
    return model


def predict_beats_from_waveform(
    waveform: np.ndarray,
    sr: int,
    *,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
    float16: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict beat and downbeat timestamps from an in-memory audio waveform.

    Args:
        waveform: 1D mono or 2D ``(samples, channels)`` audio array.
        sr: Sample rate of ``waveform`` in Hz.
        checkpoint_path: Beat This! checkpoint name, URL, or local file path.
        device: Target compute device (``"cpu"``, ``"cuda"``, etc.).
        float16: Enable fp16 autocast during transformer inference.

    Returns:
        Tuple ``(beat_times_sec, downbeat_times_sec)`` as 1D float64 arrays.

    Raises:
        ValueError: If ``waveform`` is not 1D or 2D.
    """
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {arr.shape}")
    if int(sr) != BEAT_THIS_SR:
        arr = librosa.resample(arr, orig_sr=int(sr), target_sr=BEAT_THIS_SR)

    target = torch.device(device)
    model, spect_fn = _load_cached_bundle(str(checkpoint_path), str(target))
    wav_tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
        target
    )

    with torch.inference_mode():
        spect = spect_fn(wav_tensor)
        use_autocast = bool(float16) and (
            not hasattr(torch.amp, "is_autocast_available")
            or torch.amp.is_autocast_available(target.type)
        )
        ctx: contextlib.AbstractContextManager[Any] = (
            torch.autocast(device_type=target.type, enabled=True)
            if use_autocast
            else contextlib.nullcontext()
        )
        with ctx:
            beat_logits, downbeat_logits = split_predict_aggregate(
                spect,
                model,
                chunk_size=DEFAULT_CHUNK_SIZE,
                border_size=DEFAULT_BORDER_SIZE,
                overlap_mode="keep_first",
            )
    return postprocess_minimal(beat_logits, downbeat_logits, fps=BEAT_THIS_FPS)


def predict_beats_from_audio(
    audio_path: str,
    *,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
    float16: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an audio file and predict beat and downbeat timestamps in seconds.

    Args:
        audio_path: Path to the input audio file (OGG, MP3, WAV, FLAC, etc.).
        checkpoint_path: Beat This! checkpoint name, URL, or local file path.
        device: Target compute device (``"cpu"``, ``"cuda"``, etc.).
        float16: Enable fp16 autocast during transformer inference.

    Returns:
        Tuple ``(beat_times_sec, downbeat_times_sec)`` as 1D float64 arrays.
    """
    waveform, sr = librosa.load(audio_path, sr=BEAT_THIS_SR, mono=True)
    return predict_beats_from_waveform(
        waveform,
        int(sr),
        checkpoint_path=checkpoint_path,
        device=device,
        float16=float16,
    )
