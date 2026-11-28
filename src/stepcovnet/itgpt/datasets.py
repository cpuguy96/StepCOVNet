"""Pure NumPy/PyTorch dataset helpers for ITGPT placement."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

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
    difficulty_source: str = constants.DIFFICULTY_SOURCE_METER,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Pad one chart to a 64-beat multiple.

    Args:
        chart: Loaded DDCL beat-grid chart.
        max_beats: Length cap.
        difficulty_source: Scalar conditioning signal (``"meter"`` or
            ``"density"`` for mean steps per beat).

    Returns:
        ``(inputs, slots, beat_mask)`` with a leading batch axis of 1.

    Raises:
        ValueError: If ``difficulty_source`` is unsupported.
    """
    if difficulty_source not in constants.DIFFICULTY_SOURCES:
        raise ValueError(
            "difficulty_source must be one of "
            f"{constants.DIFFICULTY_SOURCES}, got {difficulty_source!r}"
        )
    n_beats = chart.n_beats
    padded = pad_length(n_beats, max_beats)
    usable = min(n_beats, padded)
    audio = np.zeros(
        (1, padded) + chart.beat_audio.shape[1:],
        dtype=np.float32,
    )
    slots = np.zeros((1, padded, constants.N_SLOTS), dtype=np.float32)
    mask = np.zeros((1, padded), dtype=np.float32)
    audio[0, :usable] = chart.beat_audio[:usable]
    slots[0, :usable] = chart.slots[:usable]
    mask[0, :usable] = 1.0
    bpm = float(np.mean(chart.stream[:usable, 1]))
    if difficulty_source == constants.DIFFICULTY_SOURCE_DENSITY:
        difficulty = float(np.sum(chart.slots[:usable])) / float(max(1, usable))
    else:
        difficulty = float(chart.meter)
    song_progress = np.zeros((1, padded, 2), dtype=np.float32)
    if usable > 0:
        frac = np.arange(usable, dtype=np.float32) / float(max(usable - 1, 1))
        song_progress[0, :usable, 0] = frac
        song_progress[0, :usable, 1] = 1.0 - frac
    inputs = {
        "audio": audio,
        "bpm": np.array([[bpm]], dtype=np.float32),
        "difficulty": np.array([[difficulty]], dtype=np.float32),
        "song_progress": song_progress,
    }
    return inputs, slots, mask


def load_split_charts(
    dataset_config: config.ItgptDatasetConfig,
    split: str,
) -> list[ddcl_datasets.DdclChart]:
    """Load charts via the pure-NumPy DDCL beat-grid loader (log-mel or MERT).

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
        lazy_mert_cache_size=dataset_config.lazy_mert_cache_size,
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


def pack_chart_tensors(
    chart: ddcl_datasets.DdclChart,
    *,
    max_beats: int,
    difficulty_source: str = constants.DIFFICULTY_SOURCE_METER,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Pack one chart into float32 PyTorch tensors with leading batch axis 1.

    Args:
        chart: Loaded DDCL beat-grid chart.
        max_beats: Length cap.
        difficulty_source: Scalar conditioning signal (``"meter"`` or
            ``"density"``).

    Returns:
        ``(inputs, slots, beat_mask)`` as float32 :class:`torch.Tensor` objects.
    """
    inputs_np, slots_np, mask_np = pack_chart(
        chart,
        max_beats=max_beats,
        difficulty_source=difficulty_source,
    )
    inputs_t = {key: torch.from_numpy(val) for key, val in inputs_np.items()}
    return inputs_t, torch.from_numpy(slots_np), torch.from_numpy(mask_np)


class ItgptRandomChartDataset(
    IterableDataset[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]
):
    """Infinite PyTorch iterable dataset sampling one padded chart per step.

    Matches :func:`stepcovnet.itgpt.datasets.batch_generator` so each step
    draws uniformly at random with replacement from ``charts``.

    Args:
        charts: Non-empty list of loaded charts.
        max_beats: Padded beat length cap.
        seed: Base NumPy RNG seed.
        difficulty_source: Scalar conditioning signal (``"meter"`` or
            ``"density"``).

    Raises:
        ValueError: If ``charts`` is empty or ``difficulty_source`` is invalid.
    """

    def __init__(
        self,
        charts: list[ddcl_datasets.DdclChart],
        *,
        max_beats: int,
        seed: int,
        difficulty_source: str = constants.DIFFICULTY_SOURCE_METER,
    ) -> None:
        super().__init__()
        if not charts:
            raise ValueError("charts must be non-empty")
        if difficulty_source not in constants.DIFFICULTY_SOURCES:
            raise ValueError(
                "difficulty_source must be one of "
                f"{constants.DIFFICULTY_SOURCES}, got {difficulty_source!r}"
            )
        self.charts = charts
        self.max_beats = int(max_beats)
        self.seed = int(seed)
        self.difficulty_source = str(difficulty_source)

    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Yield ``(inputs, slots, beat_mask)`` batches indefinitely.

        Yields:
            tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
                Packed ``(inputs_dict, slots_tensor, beat_mask_tensor)``.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_offset = 0 if worker_info is None else int(worker_info.id) + 1
        rng = np.random.default_rng(self.seed + worker_offset * 100003)
        n_charts = len(self.charts)
        while True:
            chart = self.charts[int(rng.integers(0, n_charts))]
            yield pack_chart_tensors(
                chart,
                max_beats=self.max_beats,
                difficulty_source=self.difficulty_source,
            )


def build_dataloader(
    charts: list[ddcl_datasets.DdclChart],
    *,
    max_beats: int,
    seed: int,
    pin_memory: bool = False,
    num_workers: int = 0,
    difficulty_source: str = constants.DIFFICULTY_SOURCE_METER,
) -> DataLoader:
    """Build a PyTorch :class:`DataLoader` yielding pre-batched chart tuples.

    Args:
        charts: Loaded split (non-empty).
        max_beats: Padded beat length cap.
        seed: Base RNG seed.
        pin_memory: Whether to pin CPU host memory for faster CUDA transfers.
        num_workers: Subprocess worker count (``0`` loads on the main process).
        difficulty_source: Scalar conditioning signal (``"meter"`` or
            ``"density"``).

    Returns:
        Configured :class:`DataLoader` (``batch_size=None`` because charts are
        already packed with leading batch axis 1 and have dynamic beat lengths).

    Raises:
        ValueError: If ``num_workers`` is negative.
    """
    workers = int(num_workers)
    if workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    dataset = ItgptRandomChartDataset(
        charts,
        max_beats=max_beats,
        seed=seed,
        difficulty_source=difficulty_source,
    )
    if workers > 0:
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=pin_memory,
    )


def batch_generator(
    charts: list[ddcl_datasets.DdclChart],
    *,
    max_beats: int,
    seed: int,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """Infinite NumPy generator of one padded chart per step.

    Args:
        charts: Loaded split.
        max_beats: Pad cap.
        seed: RNG seed.

    Yields:
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
            Packed ``(inputs, slots, beat_mask)`` arrays.
    """
    rng = np.random.default_rng(seed)
    while True:
        chart = charts[int(rng.integers(0, len(charts)))]
        yield pack_chart(chart, max_beats=max_beats)
