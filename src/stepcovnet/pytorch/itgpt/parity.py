"""Weight transfer helpers for Keras -> PyTorch ITGPT parity verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from stepcovnet.pytorch.itgpt import models


def _as_np(value: Any) -> np.ndarray:
    """Convert a Keras variable or array-like weight to a float32 NumPy array.

    Args:
        value: Keras weight variable or NumPy array.

    Returns:
        Float32 NumPy array.
    """
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _copy_linear(keras_dense: Any, torch_linear: nn.Linear) -> None:
    """Copy a Keras ``Dense`` layer into a PyTorch ``nn.Linear`` layer.

    Args:
        keras_dense: Built Keras ``Dense`` layer.
        torch_linear: Target PyTorch ``nn.Linear`` layer.
    """
    kernel = _as_np(keras_dense.kernel)
    with torch.no_grad():
        torch_linear.weight.copy_(torch.from_numpy(kernel.T))
        if torch_linear.bias is not None and keras_dense.bias is not None:
            torch_linear.bias.copy_(torch.from_numpy(_as_np(keras_dense.bias)))


def _copy_conv1d(keras_conv: Any, torch_conv: nn.Conv1d) -> None:
    """Copy a Keras ``Conv1D`` layer into a PyTorch ``nn.Conv1d`` layer.

    Args:
        keras_conv: Built Keras ``Conv1D`` layer.
        torch_conv: Target PyTorch ``nn.Conv1d`` layer.
    """
    kernel = _as_np(keras_conv.kernel)
    # Keras: (kernel_size, in_channels // groups, out_channels)
    # PyTorch: (out_channels, in_channels // groups, kernel_size)
    with torch.no_grad():
        torch_conv.weight.copy_(torch.from_numpy(np.transpose(kernel, (2, 1, 0))))
        if torch_conv.bias is not None and keras_conv.bias is not None:
            torch_conv.bias.copy_(torch.from_numpy(_as_np(keras_conv.bias)))


def _copy_conv2d(keras_conv: Any, torch_conv: nn.Conv2d) -> None:
    """Copy a Keras ``Conv2D`` layer into a PyTorch ``nn.Conv2d`` layer.

    Args:
        keras_conv: Built Keras ``Conv2D`` layer.
        torch_conv: Target PyTorch ``nn.Conv2d`` layer.
    """
    kernel = _as_np(keras_conv.kernel)
    # Keras: (kh, kw, in_c, out_c) -> PyTorch: (out_c, in_c, kh, kw)
    with torch.no_grad():
        torch_conv.weight.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))))
        if torch_conv.bias is not None and keras_conv.bias is not None:
            torch_conv.bias.copy_(torch.from_numpy(_as_np(keras_conv.bias)))


def _copy_embedding(keras_emb: Any, torch_emb: nn.Embedding) -> None:
    """Copy a Keras ``Embedding`` layer into a PyTorch ``nn.Embedding`` layer.

    Args:
        keras_emb: Built Keras ``Embedding`` layer.
        torch_emb: Target PyTorch ``nn.Embedding`` layer.
    """
    weights = _as_np(keras_emb.embeddings)
    with torch.no_grad():
        torch_emb.weight.copy_(torch.from_numpy(weights))


def _copy_rmsnorm(keras_norm: Any, torch_norm: models.ItgptRmsNorm) -> None:
    """Copy a Keras ``ItgptRmsNorm`` scale vector into PyTorch.

    Args:
        keras_norm: Built Keras ``ItgptRmsNorm``.
        torch_norm: Target PyTorch ``ItgptRmsNorm``.
    """
    with torch.no_grad():
        torch_norm.scale.copy_(torch.from_numpy(_as_np(keras_norm.scale)))


def _copy_mha(
    keras_mha: Any,
    torch_mha: models.ItgptMultiHeadSelfAttention,
) -> None:
    """Copy a Keras ``MultiHeadAttention`` layer into PyTorch SDPA attention.

    Args:
        keras_mha: Built Keras ``MultiHeadAttention`` layer.
        torch_mha: Target :class:`ItgptMultiHeadSelfAttention`.
    """
    d_model = torch_mha.d_model
    q_w = _as_np(keras_mha._query_dense.kernel).reshape(d_model, d_model).T
    k_w = _as_np(keras_mha._key_dense.kernel).reshape(d_model, d_model).T
    v_w = _as_np(keras_mha._value_dense.kernel).reshape(d_model, d_model).T
    o_w = _as_np(keras_mha._output_dense.kernel).reshape(d_model, d_model).T
    q_b = _as_np(keras_mha._query_dense.bias).reshape(d_model)
    k_b = _as_np(keras_mha._key_dense.bias).reshape(d_model)
    v_b = _as_np(keras_mha._value_dense.bias).reshape(d_model)
    o_b = _as_np(keras_mha._output_dense.bias).reshape(d_model)
    with torch.no_grad():
        torch_mha.q_proj.weight.copy_(torch.from_numpy(q_w))
        torch_mha.k_proj.weight.copy_(torch.from_numpy(k_w))
        torch_mha.v_proj.weight.copy_(torch.from_numpy(v_w))
        torch_mha.out_proj.weight.copy_(torch.from_numpy(o_w))
        if torch_mha.q_proj.bias is not None:
            torch_mha.q_proj.bias.copy_(torch.from_numpy(q_b))
        if torch_mha.k_proj.bias is not None:
            torch_mha.k_proj.bias.copy_(torch.from_numpy(k_b))
        if torch_mha.v_proj.bias is not None:
            torch_mha.v_proj.bias.copy_(torch.from_numpy(v_b))
        if torch_mha.out_proj.bias is not None:
            torch_mha.out_proj.bias.copy_(torch.from_numpy(o_b))


def _copy_ffn(keras_ffn: Any, torch_ffn: models.ItgptFFN) -> None:
    """Copy a Keras ``ItgptFFN`` into PyTorch.

    Args:
        keras_ffn: Built Keras ``ItgptFFN``.
        torch_ffn: Target PyTorch ``ItgptFFN``.
    """
    _copy_linear(keras_ffn.dense1, torch_ffn.dense1)
    _copy_linear(keras_ffn.dense2, torch_ffn.dense2)


def _copy_hier_block(
    keras_block: Any,
    torch_block: models.ItgptHierarchicalAttnBlock,
) -> None:
    """Copy a Keras ``ItgptHierarchicalAttnBlock`` into PyTorch.

    Args:
        keras_block: Built Keras hierarchical block.
        torch_block: Target PyTorch hierarchical block.
    """
    with torch.no_grad():
        torch_block.pos_emb.copy_(torch.from_numpy(_as_np(keras_block.pos_emb)))
    _copy_embedding(keras_block.chunk_pos_emb, torch_block.chunk_pos_emb)
    _copy_mha(keras_block.attn_block.attn, torch_block.attn_block.attn)
    _copy_ffn(keras_block.attn_block.ffn, torch_block.attn_block.ffn)
    _copy_rmsnorm(keras_block.attn_block.norm, torch_block.attn_block.norm)
    _copy_rmsnorm(keras_block.norm1, torch_block.norm1)
    _copy_rmsnorm(keras_block.norm2, torch_block.norm2)


def _copy_gated_compression(
    keras_comp: Any,
    torch_comp: models.ItgptGatedCompression,
) -> None:
    """Copy a Keras ``ItgptGatedCompression`` layer into PyTorch.

    Args:
        keras_comp: Built Keras gated compression layer.
        torch_comp: Target PyTorch gated compression layer.
    """
    _copy_linear(keras_comp.proj, torch_comp.proj)
    _copy_rmsnorm(keras_comp.norm, torch_comp.norm)


def _copy_cond_proj(
    keras_proj: Any,
    torch_proj: models.ItgptConditionProj,
) -> None:
    """Copy a Keras ``ItgptConditionProj`` layer into PyTorch.

    Args:
        keras_proj: Built Keras condition projection.
        torch_proj: Target PyTorch condition projection.
    """
    _copy_linear(keras_proj.dense0, torch_proj.dense0)
    _copy_linear(keras_proj.dense1, torch_proj.dense1)


def _copy_film(keras_film: Any, torch_film: models.ItgptConditionFilm) -> None:
    """Copy a Keras ``ItgptConditionFilm`` layer into PyTorch.

    Args:
        keras_film: Built Keras FiLM layer.
        torch_film: Target PyTorch FiLM layer.
    """
    _copy_linear(keras_film.hidden, torch_film.hidden)
    _copy_linear(keras_film.gamma_beta, torch_film.gamma_beta)


def copy_keras_weights_to_torch(
    keras_model: Any,
    torch_model: models.ItgptPlacementModel,
) -> models.ItgptPlacementModel:
    """Copy every parameter from a built Keras ``ItgptPlacementModel`` into PyTorch.

    Useful both for migrating existing ``.keras`` champion checkpoints to
    ``.pt`` and for verifying ``< 1e-5`` forward-pass numerical parity.

    Args:
        keras_model: Built :class:`stepcovnet.itgpt.models.ItgptPlacementModel`.
        torch_model: Matching PyTorch :class:`ItgptPlacementModel`.

    Returns:
        The updated ``torch_model``.

    Raises:
        TypeError: If ``torch_model.level6`` contains an unexpected module type.
    """
    if torch_model.beat_cnn is not None and keras_model.beat_cnn is not None:
        _copy_conv2d(keras_model.beat_cnn.conv0, torch_model.beat_cnn.conv0)
        _copy_conv2d(keras_model.beat_cnn.conv1, torch_model.beat_cnn.conv1)
        _copy_conv2d(keras_model.beat_cnn.conv2, torch_model.beat_cnn.conv2)
        _copy_conv2d(keras_model.beat_cnn.conv3, torch_model.beat_cnn.conv3)
        _copy_linear(keras_model.beat_cnn.proj0, torch_model.beat_cnn.proj0)
        _copy_linear(keras_model.beat_cnn.proj1, torch_model.beat_cnn.proj1)

    if torch_model.beat_mert is not None and keras_model.beat_mert is not None:
        if (
            torch_model.beat_mert.layer_logits is not None
            and keras_model.beat_mert.layer_logits is not None
        ):
            with torch.no_grad():
                torch_model.beat_mert.layer_logits.copy_(
                    torch.from_numpy(_as_np(keras_model.beat_mert.layer_logits))
                )
        _copy_rmsnorm(keras_model.beat_mert.norm, torch_model.beat_mert.norm)
        _copy_linear(keras_model.beat_mert.proj0, torch_model.beat_mert.proj0)
        _copy_conv1d(keras_model.beat_mert.conv, torch_model.beat_mert.conv)
        _copy_linear(keras_model.beat_mert.proj1, torch_model.beat_mert.proj1)

    _copy_cond_proj(keras_model.diff_proj, torch_model.diff_proj)
    _copy_cond_proj(keras_model.bpm_proj, torch_model.bpm_proj)

    if (
        torch_model.early_film_subbeat is not None
        and keras_model.early_film_subbeat is not None
    ):
        _copy_film(keras_model.early_film_subbeat, torch_model.early_film_subbeat)
    if (
        torch_model.early_film_beat is not None
        and keras_model.early_film_beat is not None
    ):
        _copy_film(keras_model.early_film_beat, torch_model.early_film_beat)

    _copy_hier_block(keras_model.level1, torch_model.level1)
    _copy_hier_block(keras_model.level2, torch_model.level2)
    _copy_hier_block(keras_model.level3, torch_model.level3)
    _copy_hier_block(keras_model.level4, torch_model.level4)
    _copy_gated_compression(keras_model.frame_compress, torch_model.frame_compress)
    _copy_hier_block(keras_model.level5, torch_model.level5)
    _copy_gated_compression(keras_model.beat_compress, torch_model.beat_compress)
    _copy_embedding(keras_model.level6_pos, torch_model.level6_pos)

    if torch_model.bar_pos_emb is not None and keras_model.bar_pos_emb is not None:
        _copy_embedding(keras_model.bar_pos_emb, torch_model.bar_pos_emb)
    if (
        torch_model.phrase_pos_emb is not None
        and keras_model.phrase_pos_emb is not None
    ):
        _copy_embedding(keras_model.phrase_pos_emb, torch_model.phrase_pos_emb)

    for k_blk, t_blk in zip(keras_model.level6, torch_model.level6, strict=True):
        if not isinstance(t_blk, models.ItgptEncoderBlock):
            raise TypeError(f"Expected ItgptEncoderBlock, got {type(t_blk).__name__}")
        _copy_rmsnorm(k_blk.ln1, t_blk.ln1)
        _copy_mha(k_blk.attn, t_blk.attn)
        _copy_rmsnorm(k_blk.ln2, t_blk.ln2)
        _copy_ffn(k_blk.ffn, t_blk.ffn)

    _copy_rmsnorm(keras_model.conv_head.norm, torch_model.conv_head.norm)
    _copy_conv1d(keras_model.conv_head.dw, torch_model.conv_head.dw)
    _copy_conv1d(keras_model.conv_head.pw, torch_model.conv_head.pw)
    _copy_linear(keras_model.onset_head, torch_model.onset_head)
    return torch_model
