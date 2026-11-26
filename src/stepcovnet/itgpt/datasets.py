"""Full-chart tensors for ITGPT placement (pad beats to a multiple of 64)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import tensorflow as tf

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
    lead_pad_beats: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Pad one chart to a 64-beat multiple, with optional masked leading shift.

    Args:
        chart: Loaded DDCL beat-grid chart.
        max_beats: Length cap.
        lead_pad_beats: Masked zero-padded beats prepended before beat 0
            (default ``0``). Clamped so real beats are preserved whenever
            ``chart.n_beats <= max_beats``.

    Returns:
        ``(inputs, slots, beat_mask)`` with a leading batch axis of 1.
    """
    n_beats = chart.n_beats
    lead = max(0, int(lead_pad_beats))
    if lead > 0:
        max_lead = max(0, int(max_beats) - n_beats)
        lead = min(lead, (max_lead // 4) * 4)
    padded = pad_length(lead + n_beats, max_beats)
    usable = min(n_beats, max(0, padded - lead))
    audio = np.zeros(
        (1, padded) + chart.beat_audio.shape[1:],
        dtype=np.float32,
    )
    slots = np.zeros((1, padded, constants.N_SLOTS), dtype=np.float32)
    mask = np.zeros((1, padded), dtype=np.float32)
    audio[0, lead : lead + usable] = chart.beat_audio[:usable]
    slots[0, lead : lead + usable] = chart.slots[:usable]
    mask[0, lead : lead + usable] = 1.0
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
    """Load charts via the DDCL beat-grid loader (log-mel or MERT).

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
        exclude_normalized_ids=list(dataset_config.exclude_normalized_ids),
        audio_feature_type=dataset_config.audio_feature_type,
        n_frames_per_beat=dataset_config.n_frames_per_beat,
        mert_cache_suffix=dataset_config.mert_cache_suffix,
        mert_spectral_flux=dataset_config.mert_spectral_flux,
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
    max_bar_shift: int = 0,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """Infinite generator of one padded chart per step.

    Args:
        charts: Loaded split.
        max_beats: Pad cap.
        seed: RNG seed.
        max_bar_shift: Max 4-beat bars of masked leading padding prepended
            randomly per step (``0`` disables shift augmentation).

    Yields:
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]: Packed
        Packed ``(inputs, slots, beat_mask)`` for ``model.fit``.
    """
    rng = np.random.default_rng(seed)
    bar_shift_cap = max(0, int(max_bar_shift))
    while True:
        chart = charts[int(rng.integers(0, len(charts)))]
        lead_pad = (
            int(rng.integers(0, bar_shift_cap + 1)) * 4 if bar_shift_cap > 0 else 0
        )
        inputs, slots, mask = pack_chart(
            chart, max_beats=max_beats, lead_pad_beats=lead_pad
        )
        yield inputs, slots, mask


def tf_dataset(
    charts: list[ddcl_datasets.DdclChart],
    *,
    max_beats: int,
    seed: int,
    max_bar_shift: int = 0,
) -> tf.data.Dataset:
    """Return an infinite ``tf.data.Dataset`` with dynamic beat length.

    Uses an explicit ``output_signature`` with ``None`` on the beat axis so
    Keras 3 never locks the sequence length when two initial charts share the
    same 64-beat padded bucket.

    Args:
        charts: Loaded split (non-empty).
        max_beats: Pad cap.
        seed: RNG seed.
        max_bar_shift: Max 4-beat bars of masked leading shift augmentation
            (``0`` disables).

    Returns:
        ``tf.data.Dataset`` yielding ``(inputs, slots, beat_mask)`` batches.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    audio_tail = tuple(int(dim) for dim in charts[0].beat_audio.shape[1:])
    signature = (
        {
            "audio": tf.TensorSpec(shape=(1, None, *audio_tail), dtype=tf.float32),
            "bpm": tf.TensorSpec(shape=(1, 1), dtype=tf.float32),
            "difficulty": tf.TensorSpec(shape=(1, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(1, None, constants.N_SLOTS), dtype=tf.float32),
        tf.TensorSpec(shape=(1, None), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(
        lambda: batch_generator(
            charts,
            max_beats=max_beats,
            seed=seed,
            max_bar_shift=max_bar_shift,
        ),
        output_signature=signature,
    )
