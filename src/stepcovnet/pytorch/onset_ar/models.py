"""PyTorch encoder-decoder Transformer for AR onset with true O(L) KV caching."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.pytorch.onset_ar import config, targets


def _init_xavier_uniform(module: nn.Module) -> None:
    """Apply Glorot/Xavier uniform weights and zero biases.

    Args:
        module: Module to initialize in-place when linear.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal positional encoding table for AR onset sequences.

    Args:
        max_len: Maximum supported sequence length.
        d_model: Hidden embedding width.

    Attributes:
        pe: Precomputed ``(1, max_len, d_model)`` float32 encoding buffer.
    """

    pe: torch.Tensor

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]
        angle_rates = np.power(
            10000.0,
            (2 * (dims // 2)) / np.float32(self.d_model),
        )
        angle = positions / angle_rates
        table = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        table[:, 0::2] = np.sin(angle[:, 0::2])
        table[:, 1::2] = np.cos(angle[:, 1::2])
        self.register_buffer(
            "pe", torch.from_numpy(table).unsqueeze(0), persistent=False
        )

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        """Add sinusoidal encodings starting at ``offset``.

        Args:
            x: Input sequence ``(B, T, D)``.
            offset: Starting position index (used for single-step KV decoding).

        Returns:
            Position-encoded tensor ``(B, T, D)``.
        """
        seq_len = int(x.shape[1])
        return x + self.pe[:, offset : offset + seq_len, :].to(x.dtype)


class MultiHeadAttentionSDPA(nn.Module):
    """Multi-head attention using PyTorch SDPA with optional KV caching.

    Args:
        d_model: Embedding width.
        num_heads: Number of attention heads.
        dropout_rate: Attention dropout probability during training.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = max(1, self.d_model // self.num_heads)
        inner_dim = self.num_heads * self.head_dim
        self.dropout_rate = float(dropout_rate)

        self.q_proj = nn.Linear(self.d_model, inner_dim)
        self.k_proj = nn.Linear(self.d_model, inner_dim)
        self.v_proj = nn.Linear(self.d_model, inner_dim)
        self.out_proj = nn.Linear(inner_dim, self.d_model)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _init_xavier_uniform(lin)

    def project_kv(self, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project key and value tensors into multi-head layout ``(B, H, S, Dh)``.

        Args:
            kv: Source sequence ``(B, S, D)``.

        Returns:
            Tuple ``(k, v)`` each of shape ``(B, H, S, Dh)``.
        """
        batch_size, seq_len = int(kv.shape[0]), int(kv.shape[1])
        k = (
            self.k_proj(kv)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(kv)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        return k, v

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor | None = None,
        *,
        attn_bias: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute multi-head scaled dot-product attention.

        Args:
            query: Query tensor ``(B, Tq, D)``.
            key_value: Key/value source ``(B, Tk, D)`` when ``cached_kv`` is None.
            attn_bias: Optional additive mask broadcastable to ``(B, H, Tq, Tk)``.
            cached_kv: Optional pre-projected ``(k, v)`` of shape ``(B, H, Tk, Dh)``.

        Returns:
            Attention output ``(B, Tq, D)``.
        """
        batch_size, q_len = int(query.shape[0]), int(query.shape[1])
        q = (
            self.q_proj(query)
            .reshape(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if cached_kv is not None:
            k, v = cached_kv
        else:
            src = query if key_value is None else key_value
            k, v = self.project_kv(src)

        drop_p = self.dropout_rate if self.training else 0.0
        bias = None if attn_bias is None else attn_bias.to(q.dtype)
        attn = nn_func.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=drop_p,
        )
        merged = (
            attn.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, q_len, self.num_heads * self.head_dim)
        )
        return self.out_proj(merged)


class ArEncoderBlock(nn.Module):
    """Post-LN bidirectional Transformer encoder block for AR onset.

    Args:
        d_model: Hidden width.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttentionSDPA(d_model, num_heads, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-3)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn_drop = nn.Dropout(dropout_rate)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-3)
        _init_xavier_uniform(self.ffn1)
        _init_xavier_uniform(self.ffn2)

    def forward(self, x: torch.Tensor, *, attn_bias: torch.Tensor) -> torch.Tensor:
        """Run one bidirectional encoder layer.

        Args:
            x: Patch representations ``(B, P, D)``.
            attn_bias: Additive self-attention mask ``(B, 1, P, P)``.

        Returns:
            Updated representations ``(B, P, D)``.
        """
        x = self.ln1(x + self.self_attn(x, attn_bias=attn_bias))
        ffn_out = self.ffn2(self.ffn_drop(nn_func.gelu(self.ffn1(x))))
        return self.ln2(x + ffn_out)


class ArDecoderBlock(nn.Module):
    """Post-LN causal Transformer decoder block with cross-attention & KV cache.

    Args:
        d_model: Hidden width.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttentionSDPA(d_model, num_heads, dropout_rate)
        self.self_ln = nn.LayerNorm(d_model, eps=1e-3)
        self.cross_attn = MultiHeadAttentionSDPA(d_model, num_heads, dropout_rate)
        self.cross_ln = nn.LayerNorm(d_model, eps=1e-3)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn_drop = nn.Dropout(dropout_rate)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        self.ffn_ln = nn.LayerNorm(d_model, eps=1e-3)
        _init_xavier_uniform(self.ffn1)
        _init_xavier_uniform(self.ffn2)

    def forward(
        self,
        x: torch.Tensor,
        cross_memory: torch.Tensor,
        *,
        self_attn_bias: torch.Tensor,
        cross_attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Run parallel causal decoder forward pass over full sequence.

        Args:
            x: Decoder hidden states ``(B, T, D)``.
            cross_memory: Encoder cross-attention memory ``(B, P, D)``.
            self_attn_bias: Causal + padding bias ``(B, 1, T, T)``.
            cross_attn_bias: Cross-attention padding bias ``(B, 1, T, P)``.

        Returns:
            Updated decoder states ``(B, T, D)``.
        """
        x = self.self_ln(x + self.self_attn(x, attn_bias=self_attn_bias))
        x = self.cross_ln(
            x + self.cross_attn(x, cross_memory, attn_bias=cross_attn_bias)
        )
        ffn_out = self.ffn2(self.ffn_drop(nn_func.gelu(self.ffn1(x))))
        return self.ffn_ln(x + ffn_out)

    def forward_step_cached(
        self,
        x_step: torch.Tensor,
        *,
        self_k_cache: torch.Tensor,
        self_v_cache: torch.Tensor,
        position: int,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        cross_attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single incremental decode step using pre-allocated KV buffers.

        Args:
            x_step: Single-position decoder hidden state ``(B, 1, D)``.
            self_k_cache: Pre-allocated key buffer ``(B, H, max_len, Dh)``.
            self_v_cache: Pre-allocated value buffer ``(B, H, max_len, Dh)``.
            position: Current 0-based decoder step index.
            cross_kv: Pre-projected cross-attention ``(k, v)`` of shape ``(B, H, P, Dh)``.
            cross_attn_bias: Cross-attention bias ``(B, 1, 1, P)``.

        Returns:
            Updated single-position decoder state ``(B, 1, D)``.
        """
        k_new, v_new = self.self_attn.project_kv(x_step)
        self_k_cache[:, :, position : position + 1, :] = k_new
        self_v_cache[:, :, position : position + 1, :] = v_new
        k_prefix = self_k_cache[:, :, : position + 1, :]
        v_prefix = self_v_cache[:, :, : position + 1, :]
        self_out = self.self_attn(x_step, cached_kv=(k_prefix, v_prefix))
        x = self.self_ln(x_step + self_out)
        cross_out = self.cross_attn(
            x,
            cached_kv=cross_kv,
            attn_bias=cross_attn_bias,
        )
        x = self.cross_ln(x + cross_out)
        ffn_out = self.ffn2(self.ffn_drop(nn_func.gelu(self.ffn1(x))))
        return self.ffn_ln(x + ffn_out)


@dataclasses.dataclass
class ArKvCache:
    """Per-song KV cache for O(L) incremental AR onset decoding.

    Attributes:
        memory: Encoder positional memory ``(B, P, D)``.
        content_memory: Encoder PE-free content memory ``(B, P, D)``.
        patch_mask: Patch validity mask ``(B, P)``.
        cross_attn_bias: Cross-attention bias ``(B, 1, 1, P)``.
        self_k: Per-layer pre-allocated self-attention key tensors.
        self_v: Per-layer pre-allocated self-attention value tensors.
        cross_kv: Per-layer pre-projected cross-attention ``(k, v)`` tuples.
        pointer_cross_kv: Optional pre-projected pointer cross-attention ``(k, v)``.
        pointer_keys: Optional pre-projected float32 content pointer keys ``(B, P, D)``.
        density_embed: Optional global density embedding ``(B, 1, D)``.
    """

    memory: torch.Tensor
    content_memory: torch.Tensor
    patch_mask: torch.Tensor
    cross_attn_bias: torch.Tensor
    self_k: list[torch.Tensor]
    self_v: list[torch.Tensor]
    cross_kv: list[tuple[torch.Tensor, torch.Tensor]]
    pointer_cross_kv: tuple[torch.Tensor, torch.Tensor] | None
    pointer_keys: torch.Tensor | None
    density_embed: torch.Tensor | None


class ArOnsetModel(nn.Module):
    """PyTorch encoder-decoder AR onset Transformer with content pointer & gap heads.

    Args:
        experiment_config: Full :class:`config.ArExperimentConfig`.

    Attributes:
        gap_delta_lookup: Optional int64 buffer mapping gap ids to patch deltas.
    """

    gap_delta_lookup: torch.Tensor | None

    def __init__(self, experiment_config: config.ArExperimentConfig) -> None:
        super().__init__()
        self.experiment_config = experiment_config
        mc = experiment_config.model
        self.max_patches = experiment_config.max_encoder_patches()
        self.max_decoder_len = experiment_config.max_decoder_len()
        self.patch_dim = experiment_config.patch_input_dim()
        self.vocab_size = experiment_config.build_vocab().vocab_size
        self.d_model = int(mc.d_model)
        self.num_heads = int(mc.num_heads)
        self.n_enc_layers = int(mc.n_enc_layers)
        self.n_dec_layers = int(mc.n_dec_layers)
        self.dropout_rate = float(mc.dropout_rate)
        self.patch_duration = float(mc.patch_frames) * float(
            experiment_config.dataset.hop_sec
        )

        self.build_absolute_pointer = config.absolute_pointer_head_active(mc)
        self.content_pointer = (
            self.build_absolute_pointer and config.content_pointer_active(mc)
        )
        self.gap_alignment = config.gap_alignment_active(mc)
        self.content_gap = config.content_gap_active(mc)
        self.need_content_qk = bool(self.content_pointer or self.content_gap)
        self.pe_free_keys = bool(mc.pointer_keys_pe_free)
        self.query_from_cross_attn = bool(mc.pointer_query_from_cross_attn)
        self.decoder_cross_content_only = bool(mc.decoder_cross_content_only)
        self.qk_layernorm = bool(mc.pointer_qk_layernorm)
        self.use_density = config.density_conditioning_active(mc)

        gap_vocab = experiment_config.build_gap_vocab() if self.gap_alignment else None
        self.gap_vocab_size = 0 if gap_vocab is None else gap_vocab.vocab_size
        if gap_vocab is not None:
            lookup = targets.gap_delta_lookup_table(gap_vocab)
            self.register_buffer(
                "gap_delta_lookup",
                torch.from_numpy(lookup).to(torch.long),
                persistent=False,
            )
        else:
            self.gap_delta_lookup = None

        # Patch encoder (PE applied AFTER encoder layers for PE-free content keys).
        self.patch_embed = nn.Linear(self.patch_dim, self.d_model)
        _init_xavier_uniform(self.patch_embed)
        self.enc_layers = nn.ModuleList(
            [
                ArEncoderBlock(self.d_model, self.num_heads, self.dropout_rate)
                for _ in range(self.n_enc_layers)
            ]
        )
        self.enc_pos = SinusoidalPositionEncoding(self.max_patches, self.d_model)

        # Decoder input embeddings.
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.dec_pos = SinusoidalPositionEncoding(self.max_decoder_len, self.d_model)
        self.density_proj: nn.Linear | None = None
        if self.use_density:
            self.density_proj = nn.Linear(1, self.d_model)
            _init_xavier_uniform(self.density_proj)

        # Optional cross-memory projection when pe_free_keys & not content-only.
        self.cross_memory_proj: nn.Linear | None = None
        if (
            self.need_content_qk
            and self.pe_free_keys
            and not self.decoder_cross_content_only
        ):
            self.cross_memory_proj = nn.Linear(self.d_model, self.d_model)
            _init_xavier_uniform(self.cross_memory_proj)

        self.dec_layers = nn.ModuleList(
            [
                ArDecoderBlock(self.d_model, self.num_heads, self.dropout_rate)
                for _ in range(self.n_dec_layers)
            ]
        )

        # Token head.
        self.token_logits = nn.Linear(self.d_model, self.vocab_size)
        _init_xavier_uniform(self.token_logits)

        # Content pointer / gap QK pathway.
        self.pointer_cross_attn: MultiHeadAttentionSDPA | None = None
        self.pointer_query_ln: nn.LayerNorm | None = None
        self.pointer_key_ln: nn.LayerNorm | None = None
        self.pointer_query: nn.Linear | None = None
        self.pointer_key: nn.Linear | None = None
        if self.need_content_qk:
            if self.query_from_cross_attn:
                self.pointer_cross_attn = MultiHeadAttentionSDPA(
                    self.d_model, self.num_heads, self.dropout_rate
                )
            if self.qk_layernorm:
                self.pointer_query_ln = nn.LayerNorm(self.d_model, eps=1e-3)
                self.pointer_key_ln = nn.LayerNorm(self.d_model, eps=1e-3)
            self.pointer_query = nn.Linear(self.d_model, self.d_model)
            self.pointer_key = nn.Linear(self.d_model, self.d_model)
            _init_xavier_uniform(self.pointer_query)
            _init_xavier_uniform(self.pointer_key)

        # Legacy index pointer head when pointer_head == "index".
        self.index_pointer_head: nn.Linear | None = None
        if self.build_absolute_pointer and not self.content_pointer:
            self.index_pointer_head = nn.Linear(self.d_model, self.max_patches)
            _init_xavier_uniform(self.index_pointer_head)

        # Dense gap head when gap_head == "dense".
        self.dense_gap_head: nn.Linear | None = None
        if self.gap_vocab_size > 0 and not self.content_gap:
            self.dense_gap_head = nn.Linear(self.d_model, self.gap_vocab_size)
            _init_xavier_uniform(self.dense_gap_head)

        # Residual head: sigmoid * patch_duration.
        self.residual_head = nn.Linear(self.d_model, 1)
        _init_xavier_uniform(self.residual_head)

    def model_config(self) -> dict[str, Any]:
        """Return JSON-serializable experiment config for checkpoint persistence.

        Returns:
            Dictionary representation of :attr:`experiment_config`.
        """
        return self.experiment_config.as_dict()

    @staticmethod
    def _encoder_attn_bias(patch_mask: torch.Tensor) -> torch.Tensor:
        valid = patch_mask > 0.5
        pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        return (~pair).to(torch.float32).unsqueeze(1) * (-1e9)

    @staticmethod
    def _decoder_self_attn_bias(decoder_mask: torch.Tensor) -> torch.Tensor:
        seq_len = int(decoder_mask.shape[1])
        valid = decoder_mask > 0.5
        pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, device=decoder_mask.device, dtype=torch.bool)
        ).unsqueeze(0)
        keep = pair & causal
        return (~keep).to(torch.float32).unsqueeze(1) * (-1e9)

    @staticmethod
    def _cross_attn_bias(
        decoder_mask: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_valid = decoder_mask > 0.5
        k_valid = patch_mask > 0.5
        pair = q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
        return (~pair).to(torch.float32).unsqueeze(1) * (-1e9)

    def _build_cross_memory(
        self,
        memory: torch.Tensor,
        content_memory: torch.Tensor,
    ) -> torch.Tensor:
        if self.need_content_qk and self.pe_free_keys:
            if self.decoder_cross_content_only:
                return content_memory
            assert self.cross_memory_proj is not None
            return content_memory + self.cross_memory_proj(memory)
        return memory

    def _project_pointer_keys(
        self,
        memory: torch.Tensor,
        content_memory: torch.Tensor,
    ) -> torch.Tensor:
        assert self.pointer_key is not None
        key_src = content_memory if self.pe_free_keys else memory
        if self.pointer_key_ln is not None:
            key_src = self.pointer_key_ln(key_src)
        return self.pointer_key(key_src).float()

    def _gather_content_gap_logits(
        self,
        absolute_scores: torch.Tensor,
        prev_patch_indices: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self.gap_delta_lookup is not None
        n_patches = int(absolute_scores.shape[-1])
        prev = prev_patch_indices.to(torch.long)
        idx = prev.unsqueeze(-1) + self.gap_delta_lookup.view(1, 1, -1)
        in_range = (idx >= 0) & (idx < n_patches)
        idx_clamped = idx.clamp(0, n_patches - 1)
        gathered = torch.gather(absolute_scores, dim=-1, index=idx_clamped)
        mask_valid = (
            (patch_mask > 0.5).unsqueeze(1).expand(-1, int(idx_clamped.shape[1]), -1)
        )
        patch_valid = torch.gather(mask_valid, dim=-1, index=idx_clamped)
        valid = in_range & patch_valid
        return gathered + (~valid).to(gathered.dtype) * (-1e9)

    def _mask_pointer_logits(
        self,
        pointer_logits: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_patches = int(patch_mask.shape[1])
        logits = pointer_logits[..., :n_patches]
        valid = (patch_mask > 0.5).to(logits.dtype).unsqueeze(1)
        return logits + (1.0 - valid) * (-1e9)

    def _compute_heads(
        self,
        decoder: torch.Tensor,
        *,
        memory: torch.Tensor,
        content_memory: torch.Tensor,
        patch_mask: torch.Tensor,
        cross_attn_bias: torch.Tensor,
        prev_patch_indices: torch.Tensor | None = None,
        cached_pointer_cross_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        cached_pointer_keys: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {
            "token_logits": self.token_logits(decoder).float(),
        }
        absolute_content_scores: torch.Tensor | None = None
        if self.need_content_qk:
            assert self.pointer_query is not None
            key_src = content_memory if self.pe_free_keys else memory
            if self.pointer_cross_attn is not None:
                if cached_pointer_cross_kv is not None:
                    query_src = self.pointer_cross_attn(
                        decoder,
                        cached_kv=cached_pointer_cross_kv,
                        attn_bias=cross_attn_bias,
                    )
                else:
                    query_src = self.pointer_cross_attn(
                        decoder,
                        key_src,
                        attn_bias=cross_attn_bias,
                    )
            else:
                query_src = decoder
            if self.pointer_query_ln is not None:
                query_src = self.pointer_query_ln(query_src)
            q = self.pointer_query(query_src).float()
            k = (
                cached_pointer_keys
                if cached_pointer_keys is not None
                else self._project_pointer_keys(memory, content_memory)
            )
            scale = 1.0 / math.sqrt(float(self.d_model))
            absolute_content_scores = torch.einsum("btd,bpd->btp", q, k) * scale

        if self.build_absolute_pointer:
            if self.content_pointer:
                assert absolute_content_scores is not None
                raw_ptr = absolute_content_scores
            else:
                assert self.index_pointer_head is not None
                raw_ptr = self.index_pointer_head(decoder).float()
            outputs["pointer_logits"] = self._mask_pointer_logits(raw_ptr, patch_mask)

        if self.gap_vocab_size > 0:
            if self.content_gap:
                if absolute_content_scores is None or prev_patch_indices is None:
                    raise ValueError(
                        "content gap requires absolute content scores and prev_patch_indices"
                    )
                outputs["gap_logits"] = self._gather_content_gap_logits(
                    absolute_content_scores,
                    prev_patch_indices,
                    patch_mask,
                )
            else:
                assert self.dense_gap_head is not None
                outputs["gap_logits"] = self.dense_gap_head(decoder).float()

        res_ratio = torch.sigmoid(self.residual_head(decoder).float()).squeeze(-1)
        outputs["residual_sec"] = res_ratio * self.patch_duration
        return outputs

    def encode_patches(
        self,
        mert_patches: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the patch encoder stack before applying sinusoidal PE.

        Args:
            mert_patches: Patched MERT features ``(B, P, patch_dim)``.
            patch_mask: Patch validity mask ``(B, P)``.

        Returns:
            Tuple ``(memory, content_memory)`` each ``(B, P, d_model)``.
        """
        enc_bias = self._encoder_attn_bias(patch_mask)
        content_memory = self.patch_embed(mert_patches)
        for layer in self.enc_layers:
            content_memory = layer(content_memory, attn_bias=enc_bias)
        memory = self.enc_pos(content_memory)
        return memory, content_memory

    def decode_from_memory(
        self,
        memory: torch.Tensor,
        content_memory: torch.Tensor,
        patch_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        *,
        prev_patch_indices: torch.Tensor | None = None,
        density_scalar: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run parallel causal decoder over teacher-forced or prefix tokens.

        Args:
            memory: Encoder positional memory ``(B, P, D)``.
            content_memory: Encoder PE-free content memory ``(B, P, D)``.
            patch_mask: Patch validity mask ``(B, P)``.
            decoder_input_ids: Decoder token ids ``(B, T)``.
            decoder_mask: Decoder validity mask ``(B, T)``.
            prev_patch_indices: Previous onset patch indices ``(B, T)`` for content gap.
            density_scalar: Optional density scalar ``(B, 1)``.

        Returns:
            Dictionary of float32 head outputs.
        """
        decoder = self.dec_pos(self.token_embed(decoder_input_ids.to(torch.long)))
        if self.density_proj is not None and density_scalar is not None:
            density_in = density_scalar.to(decoder.dtype).reshape(-1, 1)
            decoder = decoder + self.density_proj(density_in).unsqueeze(1)

        self_bias = self._decoder_self_attn_bias(decoder_mask)
        cross_bias = self._cross_attn_bias(decoder_mask, patch_mask)
        cross_memory = self._build_cross_memory(memory, content_memory)
        for layer in self.dec_layers:
            decoder = layer(
                decoder,
                cross_memory,
                self_attn_bias=self_bias,
                cross_attn_bias=cross_bias,
            )
        return self._compute_heads(
            decoder,
            memory=memory,
            content_memory=content_memory,
            patch_mask=patch_mask,
            cross_attn_bias=cross_bias,
            prev_patch_indices=prev_patch_indices,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run full encoder-decoder forward pass.

        Args:
            inputs: Dict with ``mert_patches``, ``patch_mask``,
                ``decoder_input_ids``, ``decoder_mask``, and optional
                ``prev_patch_indices`` / ``density_scalar``.

        Returns:
            Dictionary of float32 head outputs (``token_logits``, ``residual_sec``,
            and ``pointer_logits`` / ``gap_logits`` as configured).
        """
        memory, content_memory = self.encode_patches(
            inputs["mert_patches"],
            inputs["patch_mask"],
        )
        return self.decode_from_memory(
            memory,
            content_memory,
            inputs["patch_mask"],
            inputs["decoder_input_ids"],
            inputs["decoder_mask"],
            prev_patch_indices=inputs.get("prev_patch_indices"),
            density_scalar=inputs.get("density_scalar"),
        )

    def init_kv_cache(
        self,
        mert_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        *,
        max_decoder_len: int | None = None,
        density_scalar: torch.Tensor | None = None,
    ) -> ArKvCache:
        """Encode audio once and pre-allocate per-layer KV buffers for O(L) decode.

        Args:
            mert_patches: Patched MERT features ``(B, P, patch_dim)``.
            patch_mask: Patch validity mask ``(B, P)``.
            max_decoder_len: Optional override for KV cache length.
            density_scalar: Optional density conditioning ``(B, 1)``.

        Returns:
            Initialized :class:`ArKvCache`.
        """
        memory, content_memory = self.encode_patches(mert_patches, patch_mask)
        batch_size = int(memory.shape[0])
        cache_len = int(max_decoder_len or self.max_decoder_len)
        head_dim = max(1, self.d_model // self.num_heads)
        dtype = memory.dtype
        device = memory.device

        cross_memory = self._build_cross_memory(memory, content_memory)
        self_k = [
            torch.zeros(
                batch_size,
                self.num_heads,
                cache_len,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.n_dec_layers)
        ]
        self_v = [
            torch.zeros(
                batch_size,
                self.num_heads,
                cache_len,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.n_dec_layers)
        ]
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.dec_layers:
            assert isinstance(layer, ArDecoderBlock)
            cross_kv.append(layer.cross_attn.project_kv(cross_memory))

        pointer_cross_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        pointer_keys: torch.Tensor | None = None
        if self.need_content_qk:
            key_src = content_memory if self.pe_free_keys else memory
            if self.pointer_cross_attn is not None:
                pointer_cross_kv = self.pointer_cross_attn.project_kv(key_src)
            pointer_keys = self._project_pointer_keys(memory, content_memory)

        density_embed: torch.Tensor | None = None
        if self.density_proj is not None and density_scalar is not None:
            density_in = density_scalar.to(dtype).reshape(-1, 1)
            density_embed = self.density_proj(density_in).unsqueeze(1)

        ones_q = torch.ones((batch_size, 1), device=device, dtype=patch_mask.dtype)
        cross_attn_bias = self._cross_attn_bias(ones_q, patch_mask)
        return ArKvCache(
            memory=memory,
            content_memory=content_memory,
            patch_mask=patch_mask,
            cross_attn_bias=cross_attn_bias,
            self_k=self_k,
            self_v=self_v,
            cross_kv=cross_kv,
            pointer_cross_kv=pointer_cross_kv,
            pointer_keys=pointer_keys,
            density_embed=density_embed,
        )

    def decode_step_cached(
        self,
        token_id: torch.Tensor,
        position: int,
        cache: ArKvCache,
        *,
        prev_patch_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a single O(L) decoder step using ``cache`` (mutated in-place).

        Args:
            token_id: Single-step token id tensor ``(B, 1)`` or ``(B,)``.
            position: Zero-based decoder step index.
            cache: Mutable :class:`ArKvCache` from :meth:`init_kv_cache`.
            prev_patch_indices: Optional previous patch index ``(B, 1)`` for content gap.

        Returns:
            Dictionary of single-step float32 outputs (length 1 along time axis).
        """
        tok = token_id.to(torch.long)
        if tok.ndim == 1:
            tok = tok.unsqueeze(1)
        x = self.dec_pos(self.token_embed(tok), offset=position)
        if cache.density_embed is not None:
            x = x + cache.density_embed

        for idx, layer in enumerate(self.dec_layers):
            assert isinstance(layer, ArDecoderBlock)
            x = layer.forward_step_cached(
                x,
                self_k_cache=cache.self_k[idx],
                self_v_cache=cache.self_v[idx],
                position=position,
                cross_kv=cache.cross_kv[idx],
                cross_attn_bias=cache.cross_attn_bias,
            )

        prev_arg = prev_patch_indices
        if prev_arg is not None and prev_arg.ndim == 1:
            prev_arg = prev_arg.unsqueeze(1)
        return self._compute_heads(
            x,
            memory=cache.memory,
            content_memory=cache.content_memory,
            patch_mask=cache.patch_mask,
            cross_attn_bias=cache.cross_attn_bias,
            prev_patch_indices=prev_arg,
            cached_pointer_cross_kv=cache.pointer_cross_kv,
            cached_pointer_keys=cache.pointer_keys,
        )


def build_ar_onset_model(experiment_config: config.ArExperimentConfig) -> ArOnsetModel:
    """Build a PyTorch :class:`ArOnsetModel` from ``experiment_config``.

    Args:
        experiment_config: Full AR onset experiment config.

    Returns:
        Initialized :class:`ArOnsetModel`.
    """
    return ArOnsetModel(experiment_config)
