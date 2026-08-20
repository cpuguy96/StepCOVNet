"""Chart windows for DDC C-LSTM placement training and eval."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.dataset_prep import training_index, training_loader
from stepcovnet.ddc import config, constants, features


@dataclasses.dataclass
class PlacementChart:
    """One standard-difficulty chart on the 10 ms DDC grid.

    Attributes:
        song_key: ``bundle/id`` identifier.
        difficulty: Lowercase DDR difficulty label.
        spec: Z-scored log-mel, shape ``(time, 80, 3)``.
        target: Binary onset frames, shape ``(time,)``.
        gt_times: Ground-truth onset times in seconds (float64).
        first_onset: Inclusive first labeled frame.
        last_onset: Inclusive last labeled frame.
        offset_sec: Simfile ``#OFFSET`` (for ``M-slot48`` peak snap).
        bpm_segments: BPM ladder (for ``M-slot48`` peak snap).
    """

    song_key: str
    difficulty: str
    spec: np.ndarray
    target: np.ndarray
    gt_times: np.ndarray
    first_onset: int
    last_onset: int
    offset_sec: float = 0.0
    bpm_segments: tuple[prep_models.BpmSegment, ...] = ()

    @property
    def n_frames(self) -> int:
        """Return the number of time steps."""
        return int(self.spec.shape[0])

    @property
    def difficulty_vec(self) -> np.ndarray:
        """Return the length-5 difficulty one-hot."""
        return features.difficulty_one_hot(self.difficulty)


def _limit_songs(
    entries: list[training_index.TrainingIndexEntry],
    max_songs: int,
) -> list[training_index.TrainingIndexEntry]:
    """Keep the first ``max_songs`` unique songs in sort order.

    Args:
        entries: Manifest rows already filtered to one split.
        max_songs: Maximum unique songs (-1 keeps all).

    Returns:
        Filtered entries.
    """
    if max_songs < 0:
        return list(entries)
    seen: list[str] = []
    kept: list[training_index.TrainingIndexEntry] = []
    for entry in entries:
        key = training_index.song_key(entry.normalized_bundle, entry.normalized_id)
        if key not in seen:
            if len(seen) >= max_songs:
                continue
            seen.append(key)
        if key in seen:
            kept.append(entry)
    return kept


def list_split_entries(
    dataset_config: config.PlacementDatasetConfig,
    split: str,
) -> list[training_index.TrainingIndexEntry]:
    """Return standard-difficulty rows for one split, optionally song-capped.

    Args:
        dataset_config: Placement dataset config pointing at a manifest.
        split: ``train`` or ``val``.

    Returns:
        Sorted manifest rows.

    Raises:
        ValueError: If the manifest path is missing or the split is empty.
    """
    index_path = dataset_config.training_index_path
    index = training_index.load_training_index(index_path)
    entries = [entry for entry in index.entries if entry.split == split]
    entries.sort(
        key=lambda entry: (
            entry.split,
            entry.normalized_bundle,
            entry.normalized_id,
            entry.chart_index,
        )
    )
    cap = (
        dataset_config.max_train_songs
        if split == training_index.SPLIT_TRAIN
        else dataset_config.max_val_songs
    )
    entries = _limit_songs(entries, cap)
    if not entries:
        raise ValueError(f"no {split} rows in {index_path}")
    return entries


def resolve_data_root(
    dataset_config: config.PlacementDatasetConfig,
) -> str:
    """Resolve the prepared output root for audio and chart paths.

    Args:
        dataset_config: Placement dataset config.

    Returns:
        Absolute data root as a string.
    """
    configured = str(dataset_config.data_root).strip()
    if configured:
        return configured
    index_path = dataset_config.training_index_path
    index = training_index.load_training_index(index_path)
    return str(training_index.resolve_output_dir(index, index_path))


def load_placement_chart(
    entry: training_index.TrainingIndexEntry,
    data_root: str,
    *,
    cache_features: bool = True,
    spec: np.ndarray | None = None,
) -> PlacementChart:
    """Load DDC features and a binary onset target for one chart.

    Args:
        entry: Manifest row.
        data_root: Prepared output root.
        cache_features: Write/read ``*.ddc_mel.npy`` beside audio.
        spec: Optional precomputed log-mel shared across difficulties.

    Returns:
        In-memory chart example.

    Raises:
        ValueError: If the chart has no onsets inside the spectrogram.
    """
    root = pathlib.Path(data_root)
    audio_path = root / entry.audio_relpath
    chart_path = root / entry.chart_relpath
    if spec is None:
        spec = features.load_or_compute_ddc_logmel(
            audio_path,
            cache=cache_features,
        )
    gt_times = training_loader.load_chart_times_sec(
        str(chart_path),
        entry.chart_index,
    )
    pack = prep_models.load_parsed_song(
        data_root,
        entry.normalized_bundle,
        entry.normalized_id,
    )
    target = features.times_to_frame_target(gt_times, spec.shape[0])
    onset_frames = np.flatnonzero(target >= 0.5)
    if onset_frames.size == 0:
        raise ValueError(
            f"no onsets on the DDC grid for {entry.output_relpath} "
            f"chart_index={entry.chart_index}"
        )
    return PlacementChart(
        song_key=training_index.song_key(
            entry.normalized_bundle,
            entry.normalized_id,
        ),
        difficulty=entry.difficulty.lower(),
        spec=spec,
        target=target,
        gt_times=np.asarray(gt_times, dtype=np.float64),
        first_onset=int(onset_frames[0]),
        last_onset=int(onset_frames[-1]),
        offset_sec=float(pack.metadata.offset_sec),
        bpm_segments=tuple(pack.metadata.bpm_segments),
    )


def valid_window_starts(chart: PlacementChart, nunroll: int) -> np.ndarray:
    """Return legal truncated-BPTT start frames inside the labeled span.

    Args:
        chart: Loaded placement chart.
        nunroll: Window length in frames.

    Returns:
        Integer start indices, possibly empty when the span is shorter than
        ``nunroll``.
    """
    last_start = chart.last_onset - nunroll + 1
    if last_start < chart.first_onset:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(chart.first_onset, last_start + 1, dtype=np.int64)


def extract_unroll_window(
    chart: PlacementChart,
    start: int,
    nunroll: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice one truncated-BPTT example from a chart.

    Args:
        chart: Loaded placement chart.
        start: Inclusive start frame.
        nunroll: Window length in frames.

    Returns:
        ``(audio, difficulty, target)`` with shapes ``(nunroll, 15, 80, 3)``,
        ``(nunroll, 5)``, and ``(nunroll, 1)``.
    """
    audio = features.context_windows_span(chart.spec, start, nunroll)
    target = chart.target[start : start + nunroll, np.newaxis]
    difficulty = np.broadcast_to(
        chart.difficulty_vec[np.newaxis, :],
        (nunroll, constants.N_DIFFICULTIES),
    ).copy()
    return (
        np.asarray(audio, dtype=np.float32),
        np.asarray(difficulty, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
    )


def sample_train_batch(
    charts: list[PlacementChart],
    *,
    batch_size: int,
    nunroll: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sample random truncated-BPTT windows from loaded charts.

    Args:
        charts: Train-split charts with at least one valid window.
        batch_size: Number of sequences.
        nunroll: Window length in frames.
        rng: Seeded generator.

    Returns:
        Model inputs dict and target array of shape ``(batch, nunroll, 1)``.

    Raises:
        ValueError: If no chart has a valid window of length ``nunroll``.
    """
    eligible = [
        chart for chart in charts if valid_window_starts(chart, nunroll).size > 0
    ]
    if not eligible:
        raise ValueError("no charts have a valid DDC unroll window")
    audio_batch = np.zeros(
        (
            batch_size,
            nunroll,
            constants.CONTEXT_FRAMES,
            constants.N_MELS,
            constants.N_CHANNELS,
        ),
        dtype=np.float32,
    )
    difficulty_batch = np.zeros(
        (batch_size, nunroll, constants.N_DIFFICULTIES),
        dtype=np.float32,
    )
    target_batch = np.zeros((batch_size, nunroll, 1), dtype=np.float32)
    for row in range(batch_size):
        chart = eligible[int(rng.integers(0, len(eligible)))]
        starts = valid_window_starts(chart, nunroll)
        start = int(starts[int(rng.integers(0, starts.size))])
        audio, difficulty, target = extract_unroll_window(chart, start, nunroll)
        audio_batch[row] = audio
        difficulty_batch[row] = difficulty
        target_batch[row] = target
    return {"audio": audio_batch, "difficulty": difficulty_batch}, target_batch


def load_split_charts(
    dataset_config: config.PlacementDatasetConfig,
    split: str,
) -> list[PlacementChart]:
    """Load every chart for one split (song-capped by config).

    Args:
        dataset_config: Placement dataset config.
        split: ``train`` or ``val``.

    Returns:
        Loaded charts.
    """
    data_root = resolve_data_root(dataset_config)
    entries = list_split_entries(dataset_config, split)
    spec_cache: dict[str, np.ndarray] = {}
    charts: list[PlacementChart] = []
    for entry in entries:
        spec = spec_cache.get(entry.audio_relpath)
        chart = load_placement_chart(
            entry,
            data_root,
            cache_features=dataset_config.cache_features,
            spec=spec,
        )
        spec_cache[entry.audio_relpath] = chart.spec
        charts.append(chart)
    return charts
