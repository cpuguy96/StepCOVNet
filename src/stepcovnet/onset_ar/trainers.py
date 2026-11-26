"""Training loop and checkpoint I/O for AR onset models (PyTorch)."""

from __future__ import annotations

from stepcovnet.pytorch.onset_ar.trainers import (
    BEST_CHECKPOINT_FILENAME,
    EVAL_REPORT_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    ArTrainResult,
    load_ar_onset_model,
    save_checkpoint,
    train_ar_onset,
)

__all__ = [
    "BEST_CHECKPOINT_FILENAME",
    "EVAL_REPORT_FILENAME",
    "LAST_CHECKPOINT_FILENAME",
    "ArTrainResult",
    "load_ar_onset_model",
    "save_checkpoint",
    "train_ar_onset",
]
