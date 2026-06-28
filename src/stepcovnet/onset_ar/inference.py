"""Decode AR onset model outputs into event times for evaluation."""

from __future__ import annotations

import dataclasses

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, kv_decode, losses, targets
from stepcovnet.onset_ar import models as ar_models


@dataclasses.dataclass(frozen=True)
class ArDecodeStats:
    """Free-running autoregressive decode summary."""

    times: np.ndarray
    n_forward_steps: int
    n_onset_tokens: int
    stopped_on_eos: bool


def build_scheduled_decoder_inputs(
    decoder_input_ids: tf.Tensor,
    token_logits: tf.Tensor,
    decoder_mask: tf.Tensor,
    sample_prob: tf.Tensor | float,
) -> tf.Tensor:
    """Mix teacher tokens with model argmax predictions for scheduled sampling."""
    sample_prob_f = tf.cast(sample_prob, tf.float32)
    teacher_input = tf.cast(decoder_input_ids, tf.int32)
    predictions = tf.argmax(token_logits, axis=-1, output_type=tf.int32)
    prev_predictions = tf.concat([predictions[:, :1], predictions[:, :-1]], axis=1)
    positions = tf.range(tf.shape(teacher_input)[1])[tf.newaxis, :]
    rand = tf.random.uniform(tf.shape(teacher_input), dtype=tf.float32)
    use_prediction = tf.logical_and(
        decoder_mask > 0.5,
        tf.logical_and(positions > 0, rand < sample_prob_f),
    )
    return tf.where(use_prediction, prev_predictions, teacher_input)


def _decode_autoregressive_prefix_numpy(
    decoder,
    decoder_inputs: dict[str, np.ndarray],
    *,
    max_decoder_len: int,
    patch_duration: float,
    eos_id: int,
) -> ArDecodeStats:
    """Legacy full-prefix decode loop (one forward per token)."""
    decoder_input = decoder_inputs["decoder_input_ids"]
    decoder_mask_arr = decoder_inputs["decoder_mask"]
    times: list[float] = []
    cur_len = 1
    n_forward_steps = 0
    stopped_on_eos = False
    while cur_len < max_decoder_len:
        n_forward_steps += 1
        outputs = decoder(decoder_inputs, training=False)
        pos = cur_len - 1
        token_logits = np.asarray(outputs["token_logits"][0, pos], dtype=np.float32)
        pointer_logits = np.asarray(outputs["pointer_logits"][0, pos], dtype=np.float32)
        residual_sec = float(outputs["residual_sec"][0, pos].numpy())
        next_token = int(np.argmax(token_logits))
        if next_token == eos_id:
            stopped_on_eos = True
            break
        patch_idx = int(np.argmax(pointer_logits))
        times.append(float(patch_idx) * patch_duration + residual_sec)
        if cur_len >= max_decoder_len - 1:
            break
        decoder_input[0, cur_len] = next_token
        decoder_mask_arr[0, cur_len] = 1.0
        decoder_inputs["decoder_input_ids"] = decoder_input
        decoder_inputs["decoder_mask"] = decoder_mask_arr
        cur_len += 1
    return ArDecodeStats(
        times=np.asarray(times, dtype=np.float32),
        n_forward_steps=n_forward_steps,
        n_onset_tokens=len(times),
        stopped_on_eos=stopped_on_eos,
    )


def decode_autoregressive_with_stats_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    experiment_config: config.ArExperimentConfig | None = None,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
    use_kv_cache: bool = True,
) -> ArDecodeStats:
    """Free-running decode until ``<EOS>`` or ``max_decoder_len``."""
    if experiment_config is not None and use_kv_cache:
        times, n_forward_steps, n_onset_tokens, stopped_on_eos = (
            kv_decode.decode_autoregressive_with_kv_cache_numpy(
                model,
                mert_patches,
                patch_mask,
                experiment_config=experiment_config,
                max_decoder_len=max_decoder_len,
                patch_frames=patch_frames,
                hop_sec=hop_sec,
                bos_id=bos_id,
                eos_id=eos_id,
            )
        )
        return ArDecodeStats(
            times=times,
            n_forward_steps=n_forward_steps,
            n_onset_tokens=n_onset_tokens,
            stopped_on_eos=stopped_on_eos,
        )

    mert_patches = np.asarray(mert_patches, dtype=np.float32)
    patch_mask = np.asarray(patch_mask, dtype=np.float32)
    if mert_patches.ndim == 2:
        mert_patches = mert_patches[np.newaxis, ...]
        patch_mask = patch_mask[np.newaxis, ...]

    decoder_input = np.zeros((1, max_decoder_len), dtype=np.int32)
    decoder_mask_arr = np.zeros((1, max_decoder_len), dtype=np.float32)
    decoder_input[0, 0] = bos_id
    decoder_mask_arr[0, 0] = 1.0

    if experiment_config is not None:
        cache_key = "_ar_onset_infer_models"
        infer_models = getattr(model, cache_key, None)
        if infer_models is None:
            infer_models = ar_models.build_ar_onset_inference_models(
                model,
                experiment_config,
            )
            setattr(model, cache_key, infer_models)
        encoder, decoder = infer_models
        memory = encoder(
            {"mert_patches": mert_patches, "patch_mask": patch_mask},
            training=False,
        )
        memory_np = np.asarray(memory.numpy(), dtype=np.float32)
        decoder_inputs = {
            "encoder_memory": memory_np,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input,
            "decoder_mask": decoder_mask_arr,
        }
    else:
        decoder = model
        decoder_inputs = {
            "mert_patches": mert_patches,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input,
            "decoder_mask": decoder_mask_arr,
        }

    patch_duration = float(patch_frames) * float(hop_sec)
    return _decode_autoregressive_prefix_numpy(
        decoder,
        decoder_inputs,
        max_decoder_len=max_decoder_len,
        patch_duration=patch_duration,
        eos_id=eos_id,
    )


def decode_autoregressive_times_numpy(
    model: tf.keras.Model,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    experiment_config: config.ArExperimentConfig | None = None,
    bos_id: int = targets.BOS_ID,
    eos_id: int = targets.EOS_ID,
) -> np.ndarray:
    """Free-running token decode until ``<EOS>``; return onset times in seconds."""
    return decode_autoregressive_with_stats_numpy(
        model,
        mert_patches,
        patch_mask,
        max_decoder_len=max_decoder_len,
        patch_frames=patch_frames,
        hop_sec=hop_sec,
        experiment_config=experiment_config,
        bos_id=bos_id,
        eos_id=eos_id,
    ).times


def decode_teacher_fed_times_numpy(
    pointer_logits: np.ndarray,
    residual_sec: np.ndarray,
    onset_step_mask: np.ndarray,
    *,
    patch_frames: int,
    hop_sec: float,
) -> np.ndarray:
    """Extract sorted onset times from teacher-fed decoder outputs."""
    pointer_logits = np.asarray(pointer_logits, dtype=np.float32)
    residual_sec = np.asarray(residual_sec, dtype=np.float32)
    onset_step_mask = np.asarray(onset_step_mask, dtype=np.float32)
    if pointer_logits.ndim == 3:
        pointer_logits = pointer_logits[0]
        residual_sec = residual_sec[0]
        onset_step_mask = onset_step_mask[0]

    patch_duration = float(patch_frames) * float(hop_sec)
    times: list[float] = []
    for step_idx, active in enumerate(onset_step_mask):
        if active <= 0.5:
            continue
        patch_idx = int(np.argmax(pointer_logits[step_idx]))
        times.append(float(patch_idx) * patch_duration + float(residual_sec[step_idx]))
    return np.asarray(times, dtype=np.float32)


def decode_teacher_fed_times_tf(
    outputs: dict[str, tf.Tensor],
    batch: dict[str, tf.Tensor],
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tensor wrapper returning padded predicted times and a validity mask."""
    pred_times = losses.predicted_times_from_outputs(
        outputs["pointer_logits"],
        outputs["residual_sec"],
        patch_frames=patch_frames,
        hop_sec=hop_sec,
        use_soft_expected=use_soft_expected,
    )
    pred_mask = tf.cast(batch["onset_step_mask"], tf.float32)
    return pred_times, pred_mask
