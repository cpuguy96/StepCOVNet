"""Incremental AR decode with self/cross attention caches."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config
from stepcovnet.onset_ar import models as ar_models

if TYPE_CHECKING:
    SelfKvCache = tuple[tf.Tensor | None, tf.Tensor | None]
    SelfKvCaches = tuple[SelfKvCache, ...]


@dataclasses.dataclass
class _DecoderLayerBundle:
    """Cached layer handles for one decoder block."""

    self_attn: keras.layers.MultiHeadAttention
    self_ln: keras.layers.LayerNormalization
    cross_attn: keras.layers.MultiHeadAttention
    cross_ln: keras.layers.LayerNormalization
    ffn: keras.Sequential
    ffn_ln: keras.layers.LayerNormalization


def _mha_attention_output(
    mha: keras.layers.MultiHeadAttention,
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    *,
    attention_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    """Run MHA projections + attention + output dense."""
    attn_out = mha._compute_attention(
        query,
        key,
        value,
        attention_mask=attention_mask,
        training=False,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return mha.output_dense(attn_out)


@dataclasses.dataclass
class ArOnsetKvDecoder:
    """Single-step decoder reusing weights from a trained AR onset model."""

    full_model: keras.Model
    experiment_config: config.ArExperimentConfig
    n_dec_layers: int
    d_model: int
    token_embed: keras.layers.Layer
    dec_pos: keras.layers.Layer
    mask_pointer_logits: keras.layers.Layer
    token_logits_layer: keras.layers.Layer
    pointer_logits_layer: keras.layers.Layer
    residual_ratio_layer: keras.layers.Layer
    residual_sec_layer: keras.layers.Layer
    cross_mask_layer: keras.layers.Layer
    layer_bundles: tuple[_DecoderLayerBundle, ...]
    _position_encoding: tf.Tensor
    _cross_keys: tuple[tf.Tensor, ...] | None = None
    _cross_values: tuple[tf.Tensor, ...] | None = None

    @classmethod
    def from_model(
        cls,
        full_model: keras.Model,
        experiment_config: config.ArExperimentConfig,
    ) -> ArOnsetKvDecoder:
        n_dec_layers = experiment_config.model.n_dec_layers
        layer_bundles: list[_DecoderLayerBundle] = []
        for layer_idx in range(n_dec_layers):
            prefix = f"dec_{layer_idx}"
            layer_bundles.append(
                _DecoderLayerBundle(
                    self_attn=full_model.get_layer(f"{prefix}_self_attn"),
                    self_ln=full_model.get_layer(f"{prefix}_self_ln"),
                    cross_attn=full_model.get_layer(f"{prefix}_cross_attn"),
                    cross_ln=full_model.get_layer(f"{prefix}_cross_ln"),
                    ffn=full_model.get_layer(f"{prefix}_ffn"),
                    ffn_ln=full_model.get_layer(f"{prefix}_ffn_ln"),
                ),
            )
        dec_pos = full_model.get_layer("dec_pos")
        return cls(
            full_model=full_model,
            experiment_config=experiment_config,
            n_dec_layers=n_dec_layers,
            d_model=experiment_config.model.d_model,
            token_embed=full_model.get_layer("token_embed"),
            dec_pos=dec_pos,
            mask_pointer_logits=full_model.get_layer("mask_pointer_logits"),
            token_logits_layer=full_model.get_layer("token_logits"),
            pointer_logits_layer=full_model.get_layer("pointer_logits"),
            residual_ratio_layer=full_model.get_layer("residual_ratio"),
            residual_sec_layer=full_model.get_layer("residual_sec"),
            cross_mask_layer=full_model.get_layer("cross_mask"),
            layer_bundles=tuple(layer_bundles),
            _position_encoding=dec_pos._position_encoding,
        )

    def build_cross_attention_mask(self, patch_mask: tf.Tensor) -> tf.Tensor:
        decoder_mask = tf.ones((tf.shape(patch_mask)[0], 1), dtype=tf.float32)
        return self.cross_mask_layer([decoder_mask, patch_mask])

    def precompute_cross_attention_kv(self, memory: tf.Tensor) -> None:
        """Project encoder memory to cross-attn K/V once per decode sequence."""
        cross_keys: list[tf.Tensor] = []
        cross_values: list[tf.Tensor] = []
        for bundle in self.layer_bundles:
            cross_keys.append(bundle.cross_attn.key_dense(memory))
            cross_values.append(bundle.cross_attn.value_dense(memory))
        self._cross_keys = tuple(cross_keys)
        self._cross_values = tuple(cross_values)

    def initial_self_kv_cache(self) -> SelfKvCaches:
        return tuple((None, None) for _ in range(self.n_dec_layers))

    def _position_encoding_at(self, position: int) -> tf.Tensor:
        return self._position_encoding[:, position : position + 1, :]

    def _embed_token(self, token_id: tf.Tensor, position: int) -> tf.Tensor:
        token_id = tf.cast(token_id, tf.int32)
        if token_id.shape.rank == 1:
            token_id = token_id[:, tf.newaxis]
        x = self.token_embed(token_id)
        return x + self._position_encoding_at(position)

    def _decode_step_impl(
        self,
        token_id: tf.Tensor,
        position: int,
        self_k_caches: tuple[tf.Tensor | None, ...],
        self_v_caches: tuple[tf.Tensor | None, ...],
        cross_keys: tuple[tf.Tensor, ...],
        cross_values: tuple[tf.Tensor, ...],
        cross_attention_mask: tf.Tensor,
        patch_mask: tf.Tensor,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tuple[tf.Tensor, ...],
        tuple[tf.Tensor, ...],
    ]:
        """Single decoder step; returns logits tensors and updated self KV caches."""
        x = self._embed_token(token_id, position)
        next_k_caches: list[tf.Tensor] = []
        next_v_caches: list[tf.Tensor] = []

        for layer_idx, bundle in enumerate(self.layer_bundles):
            k_new = bundle.self_attn.key_dense(x)
            v_new = bundle.self_attn.value_dense(x)
            k_cache = self_k_caches[layer_idx]
            v_cache = self_v_caches[layer_idx]
            if k_cache is None:
                k_full = k_new
                v_full = v_new
            else:
                k_full = tf.concat([k_cache, k_new], axis=1)
                v_full = tf.concat([v_cache, v_new], axis=1)

            q = bundle.self_attn.query_dense(x)
            self_out = _mha_attention_output(
                bundle.self_attn,
                q,
                k_full,
                v_full,
            )
            x = bundle.self_ln(x + self_out)
            next_k_caches.append(k_full)
            next_v_caches.append(v_full)

            cross_q = bundle.cross_attn.query_dense(x)
            cross_out = _mha_attention_output(
                bundle.cross_attn,
                cross_q,
                cross_keys[layer_idx],
                cross_values[layer_idx],
                attention_mask=cross_attention_mask,
            )
            x = bundle.cross_ln(x + cross_out)
            x = bundle.ffn_ln(x + bundle.ffn(x))

        token_logits = self.token_logits_layer(x)
        pointer_logits = self.pointer_logits_layer(x)
        pointer_logits = self.mask_pointer_logits([pointer_logits, patch_mask])
        residual_ratio = self.residual_ratio_layer(x)
        residual_sec = self.residual_sec_layer(residual_ratio)
        return (
            token_logits,
            pointer_logits,
            residual_sec,
            tuple(next_k_caches),
            tuple(next_v_caches),
        )

    def decode_step(
        self,
        token_id: tf.Tensor,
        position: int,
        *,
        patch_mask: tf.Tensor,
        self_kv_cache: SelfKvCaches,
        cross_attention_mask: tf.Tensor,
    ) -> tuple[dict[str, tf.Tensor], SelfKvCaches]:
        """Run one decoder position; return logits and updated self KV cache."""
        if self._cross_keys is None or self._cross_values is None:
            msg = "Call precompute_cross_attention_kv() before decode_step()."
            raise RuntimeError(msg)

        self_k_caches = tuple(cache[0] for cache in self_kv_cache)
        self_v_caches = tuple(cache[1] for cache in self_kv_cache)
        token_logits, pointer_logits, residual_sec, next_k, next_v = (
            self._decode_step_impl(
                token_id,
                position,
                self_k_caches,
                self_v_caches,
                self._cross_keys,
                self._cross_values,
                cross_attention_mask,
                patch_mask,
            )
        )
        return (
            {
                "token_logits": token_logits,
                "pointer_logits": pointer_logits,
                "residual_sec": residual_sec,
            },
            tuple(zip(next_k, next_v, strict=True)),
        )


def decode_autoregressive_with_kv_cache_numpy(
    model: keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    experiment_config: config.ArExperimentConfig,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    bos_id: int,
    eos_id: int,
) -> tuple[np.ndarray, int, int, bool]:
    """KV-cache free-run decode; returns times, forward steps, onset count, eos flag."""
    mert_patches = np.asarray(mert_patches, dtype=np.float32)
    patch_mask = np.asarray(patch_mask, dtype=np.float32)
    if mert_patches.ndim == 2:
        mert_patches = mert_patches[np.newaxis, ...]
        patch_mask = patch_mask[np.newaxis, ...]

    cache_key = "_ar_onset_kv_decoder"
    kv_decoder = getattr(model, cache_key, None)
    if kv_decoder is None:
        kv_decoder = ArOnsetKvDecoder.from_model(model, experiment_config)
        setattr(model, cache_key, kv_decoder)

    infer_key = "_ar_onset_infer_models"
    infer_models = getattr(model, infer_key, None)
    if infer_models is None:
        infer_models = ar_models.build_ar_onset_inference_models(
            model, experiment_config
        )
        setattr(model, infer_key, infer_models)
    encoder, _decoder = infer_models

    memory = encoder(
        {"mert_patches": mert_patches, "patch_mask": patch_mask},
        training=False,
    )
    patch_mask_tf = tf.constant(patch_mask, dtype=tf.float32)
    kv_decoder.precompute_cross_attention_kv(memory)
    cross_attention_mask = kv_decoder.build_cross_attention_mask(patch_mask_tf)
    self_kv_cache = kv_decoder.initial_self_kv_cache()

    patch_duration = float(patch_frames) * float(hop_sec)
    times: list[float] = []
    n_forward_steps = 0
    stopped_on_eos = False
    cur_len = 1
    token_id = tf.constant([[bos_id]], dtype=tf.int32)

    while cur_len < max_decoder_len:
        n_forward_steps += 1
        outputs, self_kv_cache = kv_decoder.decode_step(
            token_id,
            cur_len - 1,
            patch_mask=patch_mask_tf,
            self_kv_cache=self_kv_cache,
            cross_attention_mask=cross_attention_mask,
        )
        token_logits = np.asarray(outputs["token_logits"][0, 0], dtype=np.float32)
        pointer_logits = np.asarray(outputs["pointer_logits"][0, 0], dtype=np.float32)
        residual_sec = float(np.asarray(outputs["residual_sec"]).reshape(-1)[0])
        next_token = int(np.argmax(token_logits))
        if next_token == eos_id:
            stopped_on_eos = True
            break
        patch_idx = int(np.argmax(pointer_logits))
        times.append(float(patch_idx) * patch_duration + residual_sec)
        if cur_len >= max_decoder_len - 1:
            break
        token_id = tf.constant([[next_token]], dtype=tf.int32)
        cur_len += 1

    return (
        np.asarray(times, dtype=np.float32),
        n_forward_steps,
        len(times),
        stopped_on_eos,
    )
