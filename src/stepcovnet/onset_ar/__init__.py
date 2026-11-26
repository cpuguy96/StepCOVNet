"""Autoregressive onset Transformer package (PyTorch)."""

from __future__ import annotations

from stepcovnet.onset_ar import (
    audio_ablation,
    config,
    datasets,
    density_presets,
    inference,
    kv_decode,
    losses,
    models,
    pointer_mask,
    targets,
    trainers,
)

__all__ = [
    "audio_ablation",
    "config",
    "datasets",
    "density_presets",
    "inference",
    "kv_decode",
    "losses",
    "models",
    "pointer_mask",
    "targets",
    "trainers",
]
