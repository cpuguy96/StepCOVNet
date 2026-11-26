"""Autoregressive KV-cached decoding and evaluation (PyTorch)."""

from __future__ import annotations

from stepcovnet.pytorch.onset_ar.inference import (
    ArDecodeStats,
    ArEvalReport,
    ArLengthControl,
    ArTimeSource,
    decode_autoregressive_kv,
    eos_probability,
    evaluate_ar_samples,
    select_next_token,
)

__all__ = [
    "ArDecodeStats",
    "ArEvalReport",
    "ArLengthControl",
    "ArTimeSource",
    "decode_autoregressive_kv",
    "eos_probability",
    "evaluate_ar_samples",
    "select_next_token",
]
