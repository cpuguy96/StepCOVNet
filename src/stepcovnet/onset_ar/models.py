"""Encoder-decoder Transformer for AR onset (patched MERT memory + causal decoder)."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, targets


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
        pe = tf.cast(self._position_encoding[:, :seq_len, :], x.dtype)
        return x + pe

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"max_len": self.max_len, "d_model": self.d_model})
        return config_dict


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class PairwiseValidMask(keras.layers.Layer):
    """Self-attention mask from per-position validity.

    Keras ``MultiHeadAttention`` uses ``True`` to keep an attention pair.
    """

    def __init__(self, keep_valid: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.keep_valid = bool(keep_valid)

    def call(self, valid: tf.Tensor) -> tf.Tensor:
        valid_bool = keras.ops.cast(valid > 0.5, "bool")
        valid_q = keras.ops.expand_dims(valid_bool, axis=-1)
        valid_k = keras.ops.expand_dims(valid_bool, axis=-2)
        can_attend = keras.ops.logical_and(valid_q, valid_k)
        return can_attend if self.keep_valid else keras.ops.logical_not(can_attend)

    def get_config(self) -> dict:
        return {**super().get_config(), "keep_valid": self.keep_valid}


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class CrossAttentionMask(keras.layers.Layer):
    """Cross-attention mask from query and memory validity."""

    def __init__(self, keep_valid: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.keep_valid = bool(keep_valid)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        query_valid, memory_valid = inputs
        query_bool = keras.ops.cast(query_valid > 0.5, "bool")
        memory_bool = keras.ops.cast(memory_valid > 0.5, "bool")
        query_q = keras.ops.expand_dims(query_bool, axis=-1)
        memory_k = keras.ops.expand_dims(memory_bool, axis=-2)
        can_attend = keras.ops.logical_and(query_q, memory_k)
        return can_attend if self.keep_valid else keras.ops.logical_not(can_attend)

    def get_config(self) -> dict:
        return {**super().get_config(), "keep_valid": self.keep_valid}


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class DecoderSelfAttentionMask(keras.layers.Layer):
    """Causal decoder self-attention mask combined with padding."""

    def __init__(
        self,
        max_decoder_len: int,
        keep_valid: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.keep_valid = bool(keep_valid)
        positions = np.arange(max_decoder_len)
        future = positions[:, np.newaxis] < positions[np.newaxis, :]
        self._future_mask = tf.constant(future[np.newaxis, ...], dtype=tf.bool)

    def call(self, decoder_mask: tf.Tensor) -> tf.Tensor:
        seq_len = keras.ops.shape(decoder_mask)[1]
        dec_valid = keras.ops.cast(decoder_mask > 0.5, "bool")
        valid_q = keras.ops.expand_dims(dec_valid, axis=-1)
        valid_k = keras.ops.expand_dims(dec_valid, axis=-2)
        can_attend = keras.ops.logical_and(valid_q, valid_k)
        future = self._future_mask[:, :seq_len, :seq_len]
        if self.keep_valid:
            return keras.ops.logical_and(can_attend, keras.ops.logical_not(future))
        return keras.ops.logical_or(future, keras.ops.logical_not(can_attend))

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update(
            {
                "max_decoder_len": int(self._future_mask.shape[1]),
                "keep_valid": self.keep_valid,
            },
        )
        return config_dict


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ContentOnlyCrossMemory(keras.layers.Layer):
    """Decoder cross stream = content + 0 * PE memory (keeps enc_pos in-graph).

    A plain ``Lambda`` cannot reload under Keras safe_mode; this named layer is
    the serializable equivalent.
    """

    def call(self, inputs: list[tf.Tensor] | tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        content, pe_memory = inputs
        return content + (0.0 * pe_memory)


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ContentPointerLogits(keras.layers.Layer):
    """Scaled dot-product pointer scores of decoder queries against patch keys.

    Logit ``k`` is a similarity against ``memory[k]``, so patch choice depends on
    what the patch *contains* rather than on its absolute index. Replaces a
    ``Dense(max_patches)`` head that could score well with no audio at all.
    """

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        query, keys = inputs
        d_model = keras.ops.cast(keras.ops.shape(query)[-1], query.dtype)
        scores = keras.ops.einsum("btd,bpd->btp", query, keys)
        return scores / keras.ops.sqrt(d_model)


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class MaskPointerLogits(keras.layers.Layer):
    """Mask padded patch logits before softmax."""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        pointer_logits, patch_mask = inputs
        n_patches = keras.ops.shape(patch_mask)[1]
        pointer_logits = pointer_logits[..., :n_patches]
        valid = keras.ops.cast(patch_mask > 0.5, pointer_logits.dtype)
        valid = keras.ops.expand_dims(valid, axis=1)
        bias = (1.0 - valid) * -1e9
        return pointer_logits + bias


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ContentGapLogits(keras.layers.Layer):
    """Map absolute content-pointer scores to relative Δ gap-vocab logits.

    For each gap id with decoded Δ, gathers ``absolute_scores[..., prev+Δ]``.
    Invalid landings (OOB or padded) get ``-1e9``. Log overflow buckets use the
    bin-center Δ so every class stays audio-tied.
    """

    def __init__(
        self, delta_lookup: list[int] | tuple[int, ...] | np.ndarray, **kwargs
    ):
        super().__init__(**kwargs)
        lookup = np.asarray(delta_lookup, dtype=np.int32).reshape(-1)
        self.delta_lookup = [int(v) for v in lookup.tolist()]

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tf.Tensor:
        absolute_scores, prev_patch_indices, patch_mask = inputs
        n_patches = keras.ops.shape(absolute_scores)[-1]
        deltas = tf.constant(self.delta_lookup, dtype=tf.int32)
        prev = keras.ops.cast(prev_patch_indices, "int32")
        idx = keras.ops.expand_dims(prev, axis=-1) + deltas
        in_range = keras.ops.logical_and(idx >= 0, idx < n_patches)
        idx_clamped = keras.ops.clip(idx, 0, n_patches - 1)
        gathered = tf.gather(absolute_scores, idx_clamped, batch_dims=2)
        mask_f = keras.ops.cast(patch_mask > 0.5, gathered.dtype)
        patch_valid = tf.gather(mask_f, idx_clamped, batch_dims=1)
        valid = keras.ops.logical_and(
            in_range,
            keras.ops.cast(patch_valid > 0.5, "bool"),
        )
        bias = (1.0 - keras.ops.cast(valid, gathered.dtype)) * -1e9
        return gathered + bias

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict["delta_lookup"] = list(self.delta_lookup)
        return config_dict


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class MonotonicPointerMask(keras.layers.Layer):
    """Mask patch logits below the previous onset's patch index.

    Enforces ``patch_idx >= patch_idx_prev`` (design §7.5). ``prev_patch_indices``
    is per decoder step (teacher targets when training; predictions when decoding).
    """

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        pointer_logits, prev_patch_indices = inputs
        n_patches = keras.ops.shape(pointer_logits)[-1]
        patch_ids = keras.ops.arange(n_patches, dtype="int32")
        patch_ids = keras.ops.reshape(patch_ids, (1, 1, n_patches))
        prev = keras.ops.expand_dims(
            keras.ops.cast(prev_patch_indices, "int32"),
            axis=-1,
        )
        invalid = keras.ops.cast(patch_ids < prev, pointer_logits.dtype)
        return pointer_logits + invalid * (-1e9)


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


def _encode_patches(
    mert_patches: tf.Tensor,
    patch_mask: tf.Tensor,
    *,
    max_patches: int,
    d_model: int,
    num_heads: int,
    n_enc_layers: int,
    dropout_rate: float,
    keep_valid_attention_mask: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Run the patch encoder stack.

    Absolute PE is applied **after** the encoder so pointer keys can use
    contextualized, PE-free features. Encoding with PE first made post-encoder
    ``memory`` nearly shuffle-invariant (PE-dominated), and the previous
    pe-free workaround keyed the pointer on raw ``Dense(MERT)`` — skipping the
    encoder entirely (NOTE-20260806-02).

    Returns:
        Tuple ``(memory, content_memory)`` where ``content_memory`` is the
        PE-free encoder output (pointer keys / pe-free cross stream) and
        ``memory`` is ``content_memory`` plus absolute position encodings.
    """
    patch_embed = keras.layers.Dense(d_model, name="patch_embed")(mert_patches)
    enc_mask = PairwiseValidMask(
        keep_valid=keep_valid_attention_mask,
        name="enc_mask",
    )(patch_mask)
    content_memory = patch_embed
    for layer_idx in range(n_enc_layers):
        content_memory = _transformer_encoder_block(
            content_memory,
            attention_mask=enc_mask,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"enc_{layer_idx}",
        )
    memory = SinusoidalPositionEncoding(max_patches, d_model, name="enc_pos")(
        content_memory,
    )
    return memory, content_memory


def _apply_density_conditioning(
    decoder: tf.Tensor,
    density_scalar: tf.Tensor | None,
    *,
    density_proj: keras.layers.Layer | None,
) -> tf.Tensor:
    """Add a global density embedding to every decoder position."""
    if density_scalar is None or density_proj is None:
        return decoder
    density_embed = density_proj(density_scalar)
    return decoder + keras.ops.expand_dims(density_embed, axis=1)


def _content_pointer_logits(
    decoder: tf.Tensor,
    memory: tf.Tensor,
    patch_embed: tf.Tensor,
    *,
    cross_mask: tf.Tensor,
    d_model: int,
    num_heads: int,
    dropout_rate: float,
    pe_free_keys: bool,
    query_from_cross_attn: bool,
    qk_layernorm: bool = True,
) -> tf.Tensor:
    """Build content-pointer logits with optional PE-free keys and forced cross-attn queries.

    When ``pe_free_keys`` is set, both pointer keys and the dedicated pointer
    cross-attn read ``content_memory`` (encoder output before absolute PE).
    Absolute-PE ``memory`` remains available for positional decoder pathways.
    """
    key_source = patch_embed if pe_free_keys else memory
    if query_from_cross_attn:
        key_dim = max(1, d_model // num_heads)
        pointer_cross = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name="pointer_cross_attn",
        )(
            query=decoder,
            value=key_source,
            key=key_source,
            attention_mask=cross_mask,
        )
        query_source = pointer_cross
    else:
        query_source = decoder
    if qk_layernorm:
        query_source = keras.layers.LayerNormalization(
            name="pointer_query_ln",
        )(query_source)
        key_source = keras.layers.LayerNormalization(
            name="pointer_key_ln",
        )(key_source)
    pointer_query = keras.layers.Dense(
        d_model,
        name="pointer_query",
        dtype="float32",
    )(query_source)
    pointer_keys = keras.layers.Dense(
        d_model,
        name="pointer_key",
        dtype="float32",
    )(key_source)
    return ContentPointerLogits(name="pointer_logits_content")(
        [pointer_query, pointer_keys],
    )


def _decode_from_memory(
    memory: tf.Tensor,
    patch_embed: tf.Tensor,
    patch_mask: tf.Tensor,
    decoder_input_ids: tf.Tensor,
    decoder_mask: tf.Tensor,
    *,
    max_decoder_len: int,
    max_patches: int,
    vocab_size: int,
    d_model: int,
    num_heads: int,
    n_dec_layers: int,
    dropout_rate: float,
    patch_duration: float,
    keep_valid_attention_mask: bool,
    content_pointer: bool,
    pe_free_keys: bool,
    query_from_cross_attn: bool,
    decoder_cross_content_only: bool = False,
    qk_layernorm: bool = True,
    gap_vocab_size: int = 0,
    content_gap: bool = False,
    gap_delta_lookup: np.ndarray | None = None,
    prev_patch_indices: tf.Tensor | None = None,
    build_absolute_pointer: bool = True,
    density_scalar: tf.Tensor | None = None,
    density_proj: keras.layers.Layer | None = None,
) -> dict[str, tf.Tensor]:
    """Run the causal decoder and return logits and residual outputs."""
    need_content_qk = bool(content_pointer or content_gap)
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
    decoder = _apply_density_conditioning(
        decoder,
        density_scalar,
        density_proj=density_proj,
    )

    dec_self_mask = DecoderSelfAttentionMask(
        max_decoder_len,
        keep_valid=keep_valid_attention_mask,
        name="dec_self_mask",
    )(decoder_mask)
    cross_mask = CrossAttentionMask(
        keep_valid=keep_valid_attention_mask,
        name="cross_mask",
    )([decoder_mask, patch_mask])
    # Pe-free + content-only: decoder cross-attn reads contextualized content.
    # The prior content + Dense(PE memory) mix re-injected absolute PE and kept
    # queries shuffle-invariant (NOTE-20260806-02 follow-up).
    # Keep ``memory`` (enc_pos) in the graph via a zero add so Keras does not
    # prune it — inference rebuild still needs ``encoder_memory``.
    if need_content_qk and pe_free_keys and decoder_cross_content_only:
        cross_memory = ContentOnlyCrossMemory(name="cross_memory")(
            [patch_embed, memory],
        )
    elif need_content_qk and pe_free_keys:
        memory_proj = keras.layers.Dense(
            d_model,
            name="cross_memory_proj",
        )(memory)
        cross_memory = keras.layers.Add(name="cross_memory")(
            [patch_embed, memory_proj],
        )
    else:
        cross_memory = memory
    for layer_idx in range(n_dec_layers):
        decoder = _transformer_decoder_block(
            decoder,
            cross_memory,
            self_attention_mask=dec_self_mask,
            cross_attention_mask=cross_mask,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            name=f"dec_{layer_idx}",
        )

    token_logits = keras.layers.Dense(
        vocab_size,
        name="token_logits",
        dtype="float32",
    )(decoder)
    outputs: dict[str, tf.Tensor] = {"token_logits": token_logits}
    absolute_content_scores = None
    if need_content_qk:
        absolute_content_scores = _content_pointer_logits(
            decoder,
            memory,
            patch_embed,
            cross_mask=cross_mask,
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            pe_free_keys=pe_free_keys,
            query_from_cross_attn=query_from_cross_attn,
            qk_layernorm=qk_layernorm,
        )
    if build_absolute_pointer:
        if content_pointer:
            assert absolute_content_scores is not None
            pointer_logits = absolute_content_scores
        else:
            pointer_logits = keras.layers.Dense(
                max_patches,
                name="pointer_logits",
                dtype="float32",
            )(decoder)
        pointer_logits = MaskPointerLogits(name="mask_pointer_logits")(
            [pointer_logits, patch_mask],
        )
        outputs["pointer_logits"] = pointer_logits
    if gap_vocab_size > 0:
        if content_gap:
            if absolute_content_scores is None or prev_patch_indices is None:
                msg = "content gap requires absolute content scores and prev_patch_indices"
                raise ValueError(msg)
            if gap_delta_lookup is None:
                msg = "content gap requires gap_delta_lookup"
                raise ValueError(msg)
            outputs["gap_logits"] = ContentGapLogits(
                gap_delta_lookup,
                name="gap_logits",
            )(
                [absolute_content_scores, prev_patch_indices, patch_mask],
            )
        else:
            outputs["gap_logits"] = keras.layers.Dense(
                gap_vocab_size,
                name="gap_logits",
                dtype="float32",
            )(decoder)
    residual_ratio = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="residual_ratio",
        dtype="float32",
    )(decoder)
    residual_ratio = keras.layers.Reshape((-1,), name="residual_ratio_flat")(
        residual_ratio,
    )
    residual_sec = ScaleByPatchDuration(
        patch_duration,
        max_decoder_len,
        name="residual_sec",
    )(residual_ratio)
    outputs["residual_sec"] = residual_sec
    return outputs


def _has_layer(model: keras.Model, name: str) -> bool:
    """Whether ``model`` contains a layer called ``name``."""
    return any(layer.name == name for layer in model.layers)


def unpack_encoder_outputs(
    encoder_out: tf.Tensor | dict[str, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize encoder outputs to ``(memory, pointer_key_input)``."""
    if isinstance(encoder_out, dict):
        memory = encoder_out["memory"]
        key_input = encoder_out.get("pointer_key_input", memory)
        return memory, key_input
    return encoder_out, encoder_out


def build_ar_onset_inference_models(
    full_model: keras.Model,
    experiment_config: config.ArExperimentConfig,
) -> tuple[keras.Model, keras.Model]:
    """Encoder + decoder submodels sharing weights with ``full_model``.

    Detects the pointer head from ``full_model`` rather than from
    ``experiment_config`` so checkpoints trained with the legacy
    ``Dense(max_patches)`` head still rebuild.
    """
    model_config = experiment_config.model
    n_enc_layers = model_config.n_enc_layers
    n_dec_layers = model_config.n_dec_layers
    d_model = model_config.d_model
    use_density = config.density_conditioning_active(model_config)
    density_proj = full_model.get_layer("density_proj") if use_density else None
    content_qk = _has_layer(full_model, "pointer_logits_content")
    has_absolute_pointer = _has_layer(full_model, "mask_pointer_logits")
    has_gap_head = _has_layer(full_model, "gap_logits")
    content_gap = bool(
        has_gap_head
        and content_qk
        and isinstance(full_model.input, dict)
        and "prev_patch_indices" in full_model.input,
    )
    # Prefer graph topology over config so a drifted JSON cannot feed PE keys.
    pe_free_keys = content_qk and (
        _has_layer(full_model, "cross_memory")
        or bool(model_config.pointer_keys_pe_free)
    )
    query_from_cross = (
        content_qk
        and bool(
            model_config.pointer_query_from_cross_attn,
        )
        and _has_layer(full_model, "pointer_cross_attn")
    )
    content_pointer = content_qk  # pe-free key stream / QK rebuild

    # Encoder runs before absolute PE; ``enc_*_ln2`` is contextualized content.
    content_tensor = full_model.get_layer(f"enc_{n_enc_layers - 1}_ln2").output
    memory_tensor = (
        full_model.get_layer("enc_pos").output
        if _has_layer(full_model, "enc_pos")
        else content_tensor
    )
    encoder = keras.Model(
        inputs={
            "mert_patches": full_model.input["mert_patches"],
            "patch_mask": full_model.input["patch_mask"],
        },
        outputs={
            "memory": memory_tensor,
            "pointer_key_input": (content_tensor if pe_free_keys else memory_tensor),
        },
        name="ar_onset_encoder_infer",
    )

    memory = keras.Input(
        shape=(None, d_model),
        name="encoder_memory",
        dtype=tf.float32,
    )
    patch_mask = keras.Input(
        shape=(None,),
        name="patch_mask",
        dtype=tf.float32,
    )
    decoder_input_ids = keras.Input(
        shape=(None,),
        name="decoder_input_ids",
        dtype=tf.int32,
    )
    decoder_mask = keras.Input(
        shape=(None,),
        name="decoder_mask",
        dtype=tf.float32,
    )
    decoder_inputs: dict[str, keras.KerasTensor] = {
        "encoder_memory": memory,
        "patch_mask": patch_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_mask": decoder_mask,
    }
    pointer_key_input = memory
    if content_pointer:
        pointer_key_input = keras.Input(
            shape=(None, d_model),
            name="pointer_key_input",
            dtype=tf.float32,
        )
        decoder_inputs["pointer_key_input"] = pointer_key_input
    prev_patch_indices = None
    if content_gap:
        prev_patch_indices = keras.Input(
            shape=(None,),
            name="prev_patch_indices",
            dtype=tf.int32,
        )
        decoder_inputs["prev_patch_indices"] = prev_patch_indices
    density_scalar = None
    if use_density:
        density_scalar = keras.Input(
            shape=(1,),
            name="density_scalar",
            dtype=tf.float32,
        )
        decoder_inputs["density_scalar"] = density_scalar

    decoder = full_model.get_layer("token_embed")(decoder_input_ids)
    decoder = full_model.get_layer("dec_pos")(decoder)
    decoder = _apply_density_conditioning(
        decoder,
        density_scalar,
        density_proj=density_proj,
    )
    dec_self_mask = full_model.get_layer("dec_self_mask")(decoder_mask)
    cross_mask = full_model.get_layer("cross_mask")([decoder_mask, patch_mask])
    if pe_free_keys and _has_layer(full_model, "cross_memory"):
        if _has_layer(full_model, "cross_memory_proj"):
            memory_proj = full_model.get_layer("cross_memory_proj")(memory)
            cross_memory = full_model.get_layer("cross_memory")(
                [pointer_key_input, memory_proj],
            )
        else:
            # Content-only: Lambda(content + 0 * PE memory).
            cross_memory = full_model.get_layer("cross_memory")(
                [pointer_key_input, memory],
            )
    else:
        cross_memory = memory
    # Pointer cross-attn / keys stay on the pe-free stream when enabled.
    pointer_attn_memory = pointer_key_input if pe_free_keys else memory
    for layer_idx in range(n_dec_layers):
        prefix = f"dec_{layer_idx}"
        self_attn = full_model.get_layer(f"{prefix}_self_attn")
        self_out = self_attn(
            query=decoder,
            value=decoder,
            key=decoder,
            attention_mask=dec_self_mask,
        )
        decoder = full_model.get_layer(f"{prefix}_self_ln")(decoder + self_out)
        cross_attn = full_model.get_layer(f"{prefix}_cross_attn")
        cross_out = cross_attn(
            query=decoder,
            value=cross_memory,
            key=cross_memory,
            attention_mask=cross_mask,
        )
        decoder = full_model.get_layer(f"{prefix}_cross_ln")(decoder + cross_out)
        ffn = full_model.get_layer(f"{prefix}_ffn")
        decoder = full_model.get_layer(f"{prefix}_ffn_ln")(decoder + ffn(decoder))

    token_logits = full_model.get_layer("token_logits")(decoder)
    decoder_outputs: dict[str, tf.Tensor] = {"token_logits": token_logits}
    absolute_content_scores = None
    if content_pointer:
        if query_from_cross:
            query_source = full_model.get_layer("pointer_cross_attn")(
                query=decoder,
                value=pointer_attn_memory,
                key=pointer_attn_memory,
                attention_mask=cross_mask,
            )
        else:
            query_source = decoder
        key_source = pointer_key_input
        if _has_layer(full_model, "pointer_query_ln"):
            query_source = full_model.get_layer("pointer_query_ln")(query_source)
        if _has_layer(full_model, "pointer_key_ln"):
            key_source = full_model.get_layer("pointer_key_ln")(key_source)
        absolute_content_scores = full_model.get_layer("pointer_logits_content")(
            [
                full_model.get_layer("pointer_query")(query_source),
                full_model.get_layer("pointer_key")(key_source),
            ],
        )
    if has_absolute_pointer:
        if absolute_content_scores is not None:
            pointer_logits = absolute_content_scores
        else:
            pointer_logits = full_model.get_layer("pointer_logits")(decoder)
        pointer_logits = full_model.get_layer("mask_pointer_logits")(
            [pointer_logits, patch_mask],
        )
        decoder_outputs["pointer_logits"] = pointer_logits
    if has_gap_head:
        gap_layer = full_model.get_layer("gap_logits")
        if content_gap:
            assert absolute_content_scores is not None
            assert prev_patch_indices is not None
            decoder_outputs["gap_logits"] = gap_layer(
                [absolute_content_scores, prev_patch_indices, patch_mask],
            )
        else:
            decoder_outputs["gap_logits"] = gap_layer(decoder)
    residual_ratio = full_model.get_layer("residual_ratio")(decoder)
    residual_ratio = full_model.get_layer("residual_ratio_flat")(residual_ratio)
    residual_sec = full_model.get_layer("residual_sec")(residual_ratio)
    decoder_outputs["residual_sec"] = residual_sec
    decoder_model = keras.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs,
        name="ar_onset_decoder_infer",
    )
    return encoder, decoder_model


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
    keep_valid_attention_mask = not model_config.legacy_inverted_attention_masks
    build_absolute_pointer = config.absolute_pointer_head_active(model_config)
    content_pointer = build_absolute_pointer and config.content_pointer_active(
        model_config,
    )
    content_gap = config.content_gap_active(model_config)
    gap_vocab = (
        experiment_config.build_gap_vocab()
        if config.gap_alignment_active(model_config)
        else None
    )
    gap_vocab_size = 0 if gap_vocab is None else gap_vocab.vocab_size
    gap_delta_lookup = (
        None if gap_vocab is None else targets.gap_delta_lookup_table(gap_vocab)
    )

    mert_patches = keras.Input(
        shape=(None, patch_dim),
        name="mert_patches",
        dtype=tf.float32,
    )
    patch_mask = keras.Input(
        shape=(None,),
        name="patch_mask",
        dtype=tf.float32,
    )
    decoder_input_ids = keras.Input(
        shape=(None,),
        name="decoder_input_ids",
        dtype=tf.int32,
    )
    decoder_mask = keras.Input(
        shape=(None,),
        name="decoder_mask",
        dtype=tf.float32,
    )

    memory, patch_embed = _encode_patches(
        mert_patches,
        patch_mask,
        max_patches=max_patches,
        d_model=d_model,
        num_heads=num_heads,
        n_enc_layers=model_config.n_enc_layers,
        dropout_rate=dropout_rate,
        keep_valid_attention_mask=keep_valid_attention_mask,
    )
    patch_duration = float(model_config.patch_frames) * float(
        experiment_config.dataset.hop_sec,
    )
    use_density = config.density_conditioning_active(model_config)
    density_scalar = None
    density_proj = None
    model_inputs: dict[str, keras.KerasTensor] = {
        "mert_patches": mert_patches,
        "patch_mask": patch_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_mask": decoder_mask,
    }
    prev_patch_indices = None
    if content_gap:
        prev_patch_indices = keras.Input(
            shape=(None,),
            name="prev_patch_indices",
            dtype=tf.int32,
        )
        model_inputs["prev_patch_indices"] = prev_patch_indices
    if use_density:
        density_scalar = keras.Input(
            shape=(1,),
            name="density_scalar",
            dtype=tf.float32,
        )
        density_proj = keras.layers.Dense(d_model, name="density_proj")
        model_inputs["density_scalar"] = density_scalar
    outputs = _decode_from_memory(
        memory,
        patch_embed,
        patch_mask,
        decoder_input_ids,
        decoder_mask,
        max_decoder_len=max_decoder_len,
        max_patches=max_patches,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        n_dec_layers=model_config.n_dec_layers,
        dropout_rate=dropout_rate,
        patch_duration=patch_duration,
        keep_valid_attention_mask=keep_valid_attention_mask,
        content_pointer=content_pointer,
        pe_free_keys=bool(model_config.pointer_keys_pe_free),
        query_from_cross_attn=bool(model_config.pointer_query_from_cross_attn),
        decoder_cross_content_only=bool(model_config.decoder_cross_content_only),
        qk_layernorm=bool(model_config.pointer_qk_layernorm),
        gap_vocab_size=gap_vocab_size,
        content_gap=content_gap,
        gap_delta_lookup=gap_delta_lookup,
        prev_patch_indices=prev_patch_indices,
        build_absolute_pointer=build_absolute_pointer,
        density_scalar=density_scalar,
        density_proj=density_proj,
    )

    return keras.Model(
        inputs=model_inputs,
        outputs=outputs,
        name="ar_onset_model",
    )
