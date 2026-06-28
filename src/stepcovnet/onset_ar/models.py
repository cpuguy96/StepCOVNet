"""Encoder-decoder Transformer for AR onset (patched MERT memory + causal decoder)."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class SinusoidalPositionEncoding(keras.layers.Layer):
    """Add fixed sinusoidal position encodings up to ``max_len``."""

    def __init__(self, max_len: int, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape) -> None:
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]
        angle_rates = np.power(
            10000.0,
            (2 * (dims // 2)) / np.float32(self.d_model),
        )
        angle = positions / angle_rates
        pe = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self._position_encoding = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self._position_encoding[:, :seq_len, :]

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"max_len": self.max_len, "d_model": self.d_model})
        return config_dict


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class PairwiseValidMask(keras.layers.Layer):
    """Self-attention mask from per-position validity."""

    def call(self, valid: tf.Tensor) -> tf.Tensor:
        valid_bool = keras.ops.cast(valid > 0.5, "bool")
        valid_q = keras.ops.expand_dims(valid_bool, axis=-1)
        valid_k = keras.ops.expand_dims(valid_bool, axis=-2)
        return keras.ops.logical_and(valid_q, valid_k)


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class CrossAttentionMask(keras.layers.Layer):
    """Cross-attention mask from query and memory validity."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        query_valid, memory_valid = inputs
        query_bool = keras.ops.cast(query_valid > 0.5, "bool")
        memory_bool = keras.ops.cast(memory_valid > 0.5, "bool")
        query_q = keras.ops.expand_dims(query_bool, axis=-1)
        memory_k = keras.ops.expand_dims(memory_bool, axis=-2)
        return keras.ops.logical_and(query_q, memory_k)


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class DecoderSelfAttentionMask(keras.layers.Layer):
    """Causal decoder self-attention mask combined with padding."""

    def __init__(self, max_decoder_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        causal = np.tril(np.ones((max_decoder_len, max_decoder_len), dtype=bool))
        self._causal_mask = tf.constant(causal[np.newaxis, ...])

    def call(self, decoder_mask: tf.Tensor) -> tf.Tensor:
        dec_valid = keras.ops.cast(decoder_mask > 0.5, "bool")
        valid_q = keras.ops.expand_dims(dec_valid, axis=-1)
        valid_k = keras.ops.expand_dims(dec_valid, axis=-2)
        valid_mask = keras.ops.logical_and(valid_q, valid_k)
        return keras.ops.logical_and(self._causal_mask, valid_mask)

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update(
            {"max_decoder_len": int(self._causal_mask.shape[1])},
        )
        return config_dict


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class MaskPointerLogits(keras.layers.Layer):
    """Mask padded patch logits before softmax."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        pointer_logits, patch_mask = inputs
        valid = keras.ops.cast(patch_mask > 0.5, pointer_logits.dtype)
        valid = keras.ops.expand_dims(valid, axis=1)
        bias = (1.0 - valid) * -1e9
        return pointer_logits + bias


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ScaleByPatchDuration(keras.layers.Layer):
    """Scale sigmoid residual ratios by patch duration in seconds."""

    def __init__(self, patch_duration: float, max_decoder_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.patch_duration = float(patch_duration)
        self.max_decoder_len = max_decoder_len

    def call(self, residual_ratio: tf.Tensor) -> tf.Tensor:
        return residual_ratio * self.patch_duration

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update(
            {
                "patch_duration": self.patch_duration,
                "max_decoder_len": self.max_decoder_len,
            },
        )
        return config_dict


def _transformer_encoder_block(
    x: tf.Tensor,
    *,
    attention_mask: tf.Tensor,
    d_model: int,
    num_heads: int,
    dropout_rate: float,
    name: str,
) -> tf.Tensor:
    """One bidirectional encoder block."""
    key_dim = max(1, d_model // num_heads)
    attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=f"{name}_self_attn",
    )
    attn_out = attn(
        query=x,
        value=x,
        key=x,
        attention_mask=attention_mask,
    )
    x = keras.layers.LayerNormalization(name=f"{name}_ln1")(x + attn_out)
    ffn = keras.Sequential(
        [
            keras.layers.Dense(d_model * 4, activation="gelu", name=f"{name}_ffn1"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model, name=f"{name}_ffn2"),
        ],
        name=f"{name}_ffn",
    )
    x = keras.layers.LayerNormalization(name=f"{name}_ln2")(x + ffn(x))
    return x


def _transformer_decoder_block(
    x: tf.Tensor,
    memory: tf.Tensor,
    *,
    self_attention_mask: tf.Tensor,
    cross_attention_mask: tf.Tensor,
    d_model: int,
    num_heads: int,
    dropout_rate: float,
    name: str,
) -> tf.Tensor:
    """One causal decoder block with cross-attention to encoder memory."""
    key_dim = max(1, d_model // num_heads)
    self_attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=f"{name}_self_attn",
    )
    self_out = self_attn(
        query=x,
        value=x,
        key=x,
        attention_mask=self_attention_mask,
    )
    x = keras.layers.LayerNormalization(name=f"{name}_self_ln")(x + self_out)
    cross_attn = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=f"{name}_cross_attn",
    )
    cross_out = cross_attn(
        query=x,
        value=memory,
        key=memory,
        attention_mask=cross_attention_mask,
    )
    x = keras.layers.LayerNormalization(name=f"{name}_cross_ln")(x + cross_out)
    ffn = keras.Sequential(
        [
            keras.layers.Dense(d_model * 4, activation="gelu", name=f"{name}_ffn1"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model, name=f"{name}_ffn2"),
        ],
        name=f"{name}_ffn",
    )
    x = keras.layers.LayerNormalization(name=f"{name}_ffn_ln")(x + ffn(x))
    return x


def build_ar_onset_model(
    experiment_config: config.ArExperimentConfig,
) -> keras.Model:
    """Build the locked v1 AR onset encoder-decoder for ``gate-tide-overfit``."""
    model_config = experiment_config.model
    max_patches = experiment_config.max_encoder_patches()
    max_decoder_len = experiment_config.max_decoder_len()
    patch_dim = experiment_config.patch_input_dim()
    vocab_size = experiment_config.build_vocab().vocab_size
    d_model = model_config.d_model
    num_heads = model_config.num_heads
    dropout_rate = model_config.dropout_rate

    mert_patches = keras.Input(
        shape=(max_patches, patch_dim),
        name="mert_patches",
        dtype=tf.float32,
    )
    patch_mask = keras.Input(
        shape=(max_patches,),
        name="patch_mask",
        dtype=tf.float32,
    )
    decoder_input_ids = keras.Input(
        shape=(max_decoder_len,),
        name="decoder_input_ids",
        dtype=tf.int32,
    )
    decoder_mask = keras.Input(
        shape=(max_decoder_len,),
        name="decoder_mask",
        dtype=tf.float32,
    )

    memory = keras.layers.Dense(d_model, name="patch_embed")(mert_patches)
    memory = SinusoidalPositionEncoding(max_patches, d_model, name="enc_pos")(
        memory,
    )
    enc_mask = PairwiseValidMask(name="enc_mask")(patch_mask)
    for layer_idx in range(model_config.n_enc_layers):
        memory = _transformer_encoder_block(
            memory,
            attention_mask=enc_mask,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"enc_{layer_idx}",
        )

    token_embed = keras.layers.Embedding(
        vocab_size,
        d_model,
        name="token_embed",
    )
    decoder = token_embed(decoder_input_ids)
    decoder = SinusoidalPositionEncoding(
        max_decoder_len,
        d_model,
        name="dec_pos",
    )(decoder)

    dec_self_mask = DecoderSelfAttentionMask(
        max_decoder_len,
        name="dec_self_mask",
    )(decoder_mask)
    cross_mask = CrossAttentionMask(name="cross_mask")([decoder_mask, patch_mask])
    for layer_idx in range(model_config.n_dec_layers):
        decoder = _transformer_decoder_block(
            decoder,
            memory,
            self_attention_mask=dec_self_mask,
            cross_attention_mask=cross_mask,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"dec_{layer_idx}",
        )

    token_logits = keras.layers.Dense(vocab_size, name="token_logits")(decoder)
    pointer_logits = keras.layers.Dense(max_patches, name="pointer_logits")(decoder)
    pointer_logits = MaskPointerLogits(name="mask_pointer_logits")(
        [pointer_logits, patch_mask],
    )
    residual_ratio = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="residual_ratio",
    )(decoder)
    residual_ratio = keras.layers.Reshape(
        (max_decoder_len,), name="residual_ratio_flat"
    )(
        residual_ratio,
    )
    patch_duration = float(model_config.patch_frames) * float(
        experiment_config.dataset.hop_sec,
    )
    residual_sec = ScaleByPatchDuration(
        patch_duration,
        max_decoder_len,
        name="residual_sec",
    )(residual_ratio)

    return keras.Model(
        inputs={
            "mert_patches": mert_patches,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_mask": decoder_mask,
        },
        outputs={
            "token_logits": token_logits,
            "pointer_logits": pointer_logits,
            "residual_sec": residual_sec,
        },
        name="ar_onset_model",
    )
