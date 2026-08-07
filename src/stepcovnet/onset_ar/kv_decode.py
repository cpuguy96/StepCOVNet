"""Incremental AR decode with cached encoder memory and prefix-matched decoder steps."""

from __future__ import annotations

import dataclasses

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config
from stepcovnet.onset_ar import inference as ar_inference
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
    _pointer_key_input: tf.Tensor | None = None
    _density_scalar: tf.Tensor | None = None

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
        """Store encoder memory for the current sequence.

        Does not set pointer keys — call :meth:`set_pointer_key_input` for
        pe-free content pointers so PE-laden memory is never used as keys.
        """
        self._memory = memory

    def set_pointer_key_input(self, pointer_key_input: tf.Tensor) -> None:
        """Store PE-free (or memory) key source for the content pointer."""
        self._pointer_key_input = pointer_key_input

    def set_density_scalar(self, density_scalar: tf.Tensor | None) -> None:
        """Store per-sequence density conditioning for decoder steps."""
        self._density_scalar = density_scalar

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
            self._decoder_step_inputs(patch_mask),
            training=False,
        )
        return {
            "token_logits": outputs["token_logits"][:, position : position + 1, :],
            "pointer_logits": outputs["pointer_logits"][:, position : position + 1, :],
            "residual_sec": outputs["residual_sec"][:, position : position + 1],
        }

    def _decoder_step_inputs(self, patch_mask: tf.Tensor) -> dict[str, tf.Tensor]:
        """Build decoder inputs for one incremental decode step."""
        if self._decoder_input_ids is None or self._decoder_mask is None:
            msg = "Call reset_decode_state() before decode_step()."
            raise RuntimeError(msg)
        if self._memory is None:
            msg = "Call set_memory() before decode_step()."
            raise RuntimeError(msg)
        inputs: dict[str, tf.Tensor] = {
            "encoder_memory": self._memory,
            "patch_mask": patch_mask,
            "decoder_input_ids": self._decoder_input_ids,
            "decoder_mask": self._decoder_mask,
        }
        if config.content_pointer_active(self.experiment_config.model):
            if self._pointer_key_input is None:
                if self.experiment_config.model.pointer_keys_pe_free:
                    msg = (
                        "Call set_pointer_key_input() before decode when "
                        "pointer_keys_pe_free is enabled (do not reuse PE memory)."
                    )
                    raise RuntimeError(msg)
                key_input = self._memory
            else:
                key_input = self._pointer_key_input
            inputs["pointer_key_input"] = key_input
        if self._density_scalar is not None:
            inputs["density_scalar"] = self._density_scalar
        return inputs


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
    time_source: ar_inference.ArTimeSource = "pointer_residual",
    length_control: ar_inference.ArLengthControl | None = None,
    density_scalar: np.ndarray | float | None = None,
) -> ar_inference.ArDecodeStats:
    """Free-run decode with cached encoder memory; returns times and decode stats.

    Args:
        model: Trained AR onset model.
        mert_patches: Patched MERT features for one song.
        patch_mask: Validity mask over encoder patches.
        experiment_config: Config used to build the inference submodels.
        max_decoder_len: Hard cap on decoder positions.
        patch_frames: Hop frames per encoder patch.
        hop_sec: Seconds per hop frame.
        bos_id: Vocabulary id of the start token.
        eos_id: Vocabulary id of the end token.
        time_source: Whether times come from pointer+residual or token deltas.
        length_control: Optional decode-time ``<EOS>`` constraints.
    """
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
    _, decoder = infer_models

    kv_decoder = getattr(model, cache_key, None)
    if kv_decoder is None:
        kv_decoder = ArOnsetKvDecoder.from_model(model, experiment_config, decoder)
        setattr(model, cache_key, kv_decoder)

    memory_np, key_np, patch_mask = ar_inference.get_encoder_memory_numpy(
        model,
        mert_patches,
        patch_mask,
        experiment_config,
    )
    memory = tf.constant(memory_np, dtype=tf.float32)
    key_input = tf.constant(key_np, dtype=tf.float32)
    patch_mask_tf = tf.constant(patch_mask, dtype=tf.float32)
    kv_decoder.reset_decode_state(batch_size=int(patch_mask.shape[0]))
    kv_decoder.set_memory(memory)
    kv_decoder.set_pointer_key_input(key_input)
    if config.density_conditioning_active(experiment_config.model):
        if density_scalar is None:
            density_scalar = np.asarray([0.0], dtype=np.float32)
        density_arr = np.asarray(density_scalar, dtype=np.float32).reshape(-1)
        if density_arr.size == 1:
            density_arr = np.repeat(density_arr, int(patch_mask.shape[0]))
        kv_decoder.set_density_scalar(
            tf.constant(density_arr.reshape(-1, 1), dtype=tf.float32),
        )

    patch_duration = float(patch_frames) * float(hop_sec)
    pointer_times: list[float] = []
    onset_token_ids: list[int] = []
    eos_probs: list[float] = []
    prev_patch = 0
    monotonic = bool(experiment_config.model.monotonic_pointer)
    max_ahead = config.pointer_decode_max_ahead(experiment_config.run)
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
        eos_probs.append(ar_inference.eos_probability(token_logits, eos_id=eos_id))
        next_token = ar_inference.select_next_token(
            token_logits,
            eos_id=eos_id,
            n_emitted=len(onset_token_ids),
            length_control=length_control,
        )
        if next_token == eos_id:
            stopped_on_eos = True
            break
        onset_token_ids.append(next_token)
        patch_idx = ar_inference._argmax_pointer_patch(  # noqa: SLF001
            pointer_logits,
            prev_patch=prev_patch,
            monotonic=monotonic,
            max_ahead=max_ahead,
        )
        if monotonic:
            prev_patch = patch_idx
        pointer_times.append(float(patch_idx) * patch_duration + residual_sec)
        if cur_len >= max_decoder_len - 1:
            break
        token_id = tf.constant([[next_token]], dtype=tf.int32)
        cur_len += 1

    token_arr = np.asarray(onset_token_ids, dtype=np.int32)
    if time_source == "tokens":
        times = ar_inference.decode_onset_tokens_to_times(
            token_arr,
            experiment_config=experiment_config,
            patch_mask=patch_mask,
        )
    else:
        times = np.asarray(pointer_times, dtype=np.float32)

    return ar_inference.ArDecodeStats(
        times=times,
        n_forward_steps=n_forward_steps,
        n_onset_tokens=len(onset_token_ids),
        stopped_on_eos=stopped_on_eos,
        onset_token_ids=token_arr,
        eos_prob_trace=np.asarray(eos_probs, dtype=np.float32),
    )
