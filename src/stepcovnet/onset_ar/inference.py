"""Decode AR onset model outputs into event times for evaluation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import losses


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
