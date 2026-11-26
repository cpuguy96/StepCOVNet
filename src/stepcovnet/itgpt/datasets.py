"""Dataset helpers for ITGPT placement training and evaluation (PyTorch)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.pytorch.itgpt.datasets import (
    ItgptRandomChartDataset,
    build_dataloader,
    load_split_charts,
    pack_chart,
    pack_chart_tensors,
    pad_length,
    sample_weight,
)


def batch_generator(
    charts: list[ddcl_datasets.DdclChart],
    *,
    max_beats: int,
    seed: int,
    max_bar_shift: int = 0,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """Infinite NumPy generator of one padded chart per step.

    Args:
        charts: Loaded split.
        max_beats: Pad cap.
        seed: RNG seed.
        max_bar_shift: Max 4-beat bars of masked leading shift augmentation.

    Yields:
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
            Packed ``(inputs, slots, beat_mask)`` arrays.
    """
    rng = np.random.default_rng(seed)
    bar_shift_cap = max(0, int(max_bar_shift))
    while True:
        chart = charts[int(rng.integers(0, len(charts)))]
        lead_pad = (
            int(rng.integers(0, bar_shift_cap + 1)) * 4 if bar_shift_cap > 0 else 0
        )
        yield pack_chart(chart, max_beats=max_beats, lead_pad_beats=lead_pad)


__all__ = [
    "ItgptRandomChartDataset",
    "batch_generator",
    "build_dataloader",
    "load_split_charts",
    "pack_chart",
    "pack_chart_tensors",
    "pad_length",
    "sample_weight",
]
