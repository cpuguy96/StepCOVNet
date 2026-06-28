"""Incremental AR decode with cached encoder memory and prefix-matched decoder steps."""

from __future__ import annotations

import dataclasses

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config
from stepcovnet.onset_ar import models as ar_models


@dataclasses.dataclass
class ArOnsetKvDecoder:
    """Single-step decoder reusing weights from a trained AR onset model."""

    decoder: keras.Model
    experiment_config: config.ArExperimentConfig
    max_decoder_len: int
    _decoder_input_ids: tf.Tensor | None = None
    _decoder_mask: tf.Tensor | None = None
    _memory: tf.Tensor | None = None

    @classmethod
    def from_model(
        cls,
        full_model: keras.Model,
        experiment_config: config.ArExperimentConfig,
        decoder: keras.Model,
    ) -> ArOnsetKvDecoder:
        return cls(
            decoder=decoder,
            experiment_config=experiment_config,
            max_decoder_len=experiment_config.max_decoder_len(),
        )

    def reset_decode_state(self, batch_size: int = 1) -> None:
        """Clear decoder token/mask state for a new sequence."""
        self._decoder_input_ids = tf.zeros(
            (batch_size, self.max_decoder_len),
            dtype=tf.int32,
        )
        self._decoder_mask = tf.zeros(
            (batch_size, self.max_decoder_len),
            dtype=tf.float32,
        )

    def set_memory(self, memory: tf.Tensor) -> None:
        """Store encoder memory for the current sequence."""
        self._memory = memory

    def decode_step(
        self,
        token_id: tf.Tensor,
        position: int,
        *,
        patch_mask: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Run one decoder position using the prefix inference decoder."""
        if self._decoder_input_ids is None or self._decoder_mask is None:
            msg = "Call reset_decode_state() before decode_step()."
            raise RuntimeError(msg)
        if self._memory is None:
            msg = "Call set_memory() before decode_step()."
            raise RuntimeError(msg)

        token_id = tf.cast(token_id, tf.int32)
        if token_id.shape.rank == 1:
            token_id = token_id[:, tf.newaxis]

        self._decoder_input_ids = tf.tensor_scatter_nd_update(
            self._decoder_input_ids,
            [[0, position]],
            tf.reshape(token_id[0, 0], [1]),
        )
        self._decoder_mask = tf.tensor_scatter_nd_update(
            self._decoder_mask,
            [[0, position]],
            tf.constant([1.0], dtype=tf.float32),
        )

        outputs = self.decoder(
            {
                "encoder_memory": self._memory,
                "patch_mask": patch_mask,
                "decoder_input_ids": self._decoder_input_ids,
                "decoder_mask": self._decoder_mask,
            },
            training=False,
        )
        return {
            "token_logits": outputs["token_logits"][:, position : position + 1, :],
            "pointer_logits": outputs["pointer_logits"][:, position : position + 1, :],
            "residual_sec": outputs["residual_sec"][:, position : position + 1],
        }


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
    """Free-run decode with cached encoder memory; returns times and decode stats."""
    mert_patches = np.asarray(mert_patches, dtype=np.float32)
    patch_mask = np.asarray(patch_mask, dtype=np.float32)
    if mert_patches.ndim == 2:
        mert_patches = mert_patches[np.newaxis, ...]
        patch_mask = patch_mask[np.newaxis, ...]

    cache_key = "_ar_onset_kv_decoder"
    infer_key = "_ar_onset_infer_models"
    infer_models = getattr(model, infer_key, None)
    if infer_models is None:
        infer_models = ar_models.build_ar_onset_inference_models(
            model, experiment_config
        )
        setattr(model, infer_key, infer_models)
    encoder, decoder = infer_models

    kv_decoder = getattr(model, cache_key, None)
    if kv_decoder is None:
        kv_decoder = ArOnsetKvDecoder.from_model(model, experiment_config, decoder)
        setattr(model, cache_key, kv_decoder)

    memory = encoder(
        {"mert_patches": mert_patches, "patch_mask": patch_mask},
        training=False,
    )
    patch_mask_tf = tf.constant(patch_mask, dtype=tf.float32)
    kv_decoder.reset_decode_state(batch_size=int(patch_mask.shape[0]))
    kv_decoder.set_memory(memory)

    patch_duration = float(patch_frames) * float(hop_sec)
    times: list[float] = []
    n_forward_steps = 0
    stopped_on_eos = False
    cur_len = 1
    token_id = tf.constant([[bos_id]], dtype=tf.int32)

    while cur_len < max_decoder_len:
        n_forward_steps += 1
        outputs = kv_decoder.decode_step(
            token_id,
            cur_len - 1,
            patch_mask=patch_mask_tf,
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
