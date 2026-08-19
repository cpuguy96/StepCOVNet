"""Full-chart tensors for ITGPT placement (pad beats to a multiple of 64)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from stepcovnet.ddcl import config as ddcl_config
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants


def pad_length(n_beats: int, max_beats: int) -> int:
    """Return the padded beat count (multiple of 64, capped at ``max_beats``).

    Args:
        n_beats: True integer-beat count.
        max_beats: Inclusive cap (already a multiple of 64 in configs).

    Returns:
        Padded length.
    """
    usable = min(int(n_beats), int(max_beats))
    aligned = (
        (usable + constants.CHUNK_ALIGN - 1)
        // constants.CHUNK_ALIGN
        * constants.CHUNK_ALIGN
    )
    return max(constants.CHUNK_ALIGN, aligned)


def pack_chart(
    chart: ddcl_datasets.DdclChart,
    *,
    max_beats: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Pad one chart to a 64-beat multiple.

    Args:
        chart: Loaded DDCL beat-grid chart.
        max_beats: Length cap.

    Returns:
        ``(inputs, slots, beat_mask)`` with a leading batch axis of 1.
    """
    n_beats = chart.n_beats
    padded = pad_length(n_beats, max_beats)
    usable = min(n_beats, padded)
    audio = np.zeros(
        (
            1,
            padded,
            constants.N_FRAMES_PER_BEAT,
            constants.N_MELS,
            constants.N_CHANNELS,
        ),
        dtype=np.float32,
    )
    slots = np.zeros((1, padded, constants.N_SLOTS), dtype=np.float32)
    mask = np.zeros((1, padded), dtype=np.float32)
    audio[0, :usable] = chart.beat_audio[:usable]
    slots[0, :usable] = chart.slots[:usable]
    mask[0, :usable] = 1.0
    bpm = float(np.mean(chart.stream[:usable, 1]))
    difficulty = float(chart.meter)
    inputs = {
        "audio": audio,
        "bpm": np.array([[bpm]], dtype=np.float32),
        "difficulty": np.array([[difficulty]], dtype=np.float32),
    }
    return inputs, slots, mask


def load_split_charts(
    dataset_config: config.ItgptDatasetConfig,
    split: str,
) -> list[ddcl_datasets.DdclChart]:
    """Load Dataset B charts via the DDCL beat-grid loader.

    Args:
        dataset_config: ITGPT dataset config.
        split: ``train`` or ``val``.

    Returns:
        Loaded charts.
    """
    ddcl_dataset = ddcl_config.DdclDatasetConfig(
        training_index_path=dataset_config.training_index_path,
        data_root=dataset_config.data_root,
        batch_size=1,
        memlen=0,
        max_train_songs=dataset_config.max_train_songs,
        max_val_songs=dataset_config.max_val_songs,
        cache_features=dataset_config.cache_features,
    )
    return ddcl_datasets.load_split_charts(ddcl_dataset, split)


def sample_weight(mask: np.ndarray) -> np.ndarray:
    """Broadcast the beat mask across the 48-slot grid weights.

    Args:
        mask: ``(batch, beats)`` 1 on real beats.

    Returns:
        ``(batch, beats, 48)`` sample weights.
    """
    grid = np.ones((constants.N_SLOTS,), dtype=np.float32) * constants.GRID_WEIGHT_MICRO
    for index in constants.INDICES_16TH:
        grid[index] = constants.GRID_WEIGHT_16TH
    for index in constants.INDICES_24TH:
        grid[index] = constants.GRID_WEIGHT_24TH
    for index in constants.INDICES_32ND:
        grid[index] = 1.0
    return mask[..., None] * grid[None, None, :]


def batch_generator(
    charts: list[ddcl_datasets.DdclChart],
    *,
    max_beats: int,
    seed: int,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """Infinite generator of one padded chart per step.

    Args:
        charts: Loaded split.
        max_beats: Pad cap.
        seed: RNG seed.

    Yields:
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]: Packed
        Packed ``(inputs, slots, beat_mask)`` for ``model.fit``.
    """
    rng = np.random.default_rng(seed)
    while True:
        chart = charts[int(rng.integers(0, len(charts)))]
        inputs, slots, mask = pack_chart(chart, max_beats=max_beats)
        yield inputs, slots, mask
