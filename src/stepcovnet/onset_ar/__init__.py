"""Autoregressive onset detection (times-only seq2seq on patched MERT)."""

from stepcovnet.onset_ar import (
    config,
    datasets,
    inference,
    losses,
    models,
    targets,
    trainers,
)

__all__ = [
    "config",
    "datasets",
    "inference",
    "losses",
    "models",
    "targets",
    "trainers",
]
