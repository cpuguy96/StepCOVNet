"""Hierarchical Transformer building blocks for ITGPT placement (`omalley2026itgpt`).

Uses :func:`torch.nn.functional.scaled_dot_product_attention` (FlashAttention /
cuDNN SDPA on CUDA) and Xavier uniform weight initialization.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func


def _init_linear_or_conv(module: nn.Module) -> None:
    """Apply Glorot/Xavier uniform weights and zero bias.

    Args:
        module: Layer to initialize if it is ``Linear``, ``Conv1d``, or ``Conv2d``.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ItgptRmsNorm(nn.Module):
    """RMSNorm matching ITGPT ``nn.RMSNorm``.

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

    Uses separate Q/K/V/O linear projections with FlashAttention / cuDNN SDPA.

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
