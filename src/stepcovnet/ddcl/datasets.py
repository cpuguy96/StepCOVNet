"""Chart windows for DDCL ConvLSTM placement training and eval."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, features, slots


@dataclasses.dataclass
class DdclChart:
    """One standard-difficulty chart on the 48-slot beat grid.

    Attributes:
        song_key: ``bundle/id`` identifier.
        difficulty: Lowercase DDR difficulty label.
        meter: Chart ``#METER``.
        beat_audio: Z-scored per-beat log-mel ``(n_beats, 32, 80, 3)``.
        stream: Per-beat ``[meter, bpm]`` ``(n_beats, 2)``.
        slots: Binary ``M-slot48`` targets ``(n_beats, 48)``.
        audio_fwd: Causal windows ``(n_beats, memlen+1, 32, 80, 3)``.
        audio_bwd: Reverse windows of the same shape.
        stream_fwd: Causal stream windows ``(n_beats, memlen+1, 2)``.
        stream_bwd: Reverse stream windows, time-flipped as in DDCL.
    """

    song_key: str
    difficulty: str
    meter: int
    beat_audio: np.ndarray
    stream: np.ndarray
    slots: np.ndarray
    audio_fwd: np.ndarray
    audio_bwd: np.ndarray
    stream_fwd: np.ndarray
    stream_bwd: np.ndarray

    @property
    def n_beats(self) -> int:
        """Return the number of integer beats."""
        return int(self.slots.shape[0])


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
    dataset_config: config.DdclDatasetConfig,
    split: str,
) -> list[training_index.TrainingIndexEntry]:
    """Return standard-difficulty rows for one split, optionally song-capped.

    Args:
        dataset_config: Dataset config.
        split: ``train`` or ``val``.

    Returns:
        Manifest rows.

    Raises:
        ValueError: If the split is empty.
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


def resolve_data_root(dataset_config: config.DdclDatasetConfig) -> str:
    """Resolve the prepared output root for audio and chart paths.

    Args:
        dataset_config: Dataset config.

    Returns:
        Absolute data root as a string.
    """
    configured = str(dataset_config.data_root).strip()
    if configured:
        return configured
    index_path = dataset_config.training_index_path
    index = training_index.load_training_index(index_path)
    return str(training_index.resolve_output_dir(index, index_path))


def load_ddcl_chart(
    entry: training_index.TrainingIndexEntry,
    data_root: str,
    *,
    memlen: int,
    cache_features: bool = True,
    spec: np.ndarray | None = None,
) -> DdclChart:
    """Load beat-grid features and ``M-slot48`` targets for one chart.

    Args:
        entry: Manifest row.
        data_root: Prepared output root.
        memlen: Context length for windows.
        cache_features: Write/read ``*.ddc_mel.npy`` beside audio.
        spec: Optional precomputed log-mel shared across difficulties.

    Returns:
        In-memory chart example.

    Raises:
        ValueError: If the chart has no onsets.
    """
    root = pathlib.Path(data_root)
    audio_path = root / entry.audio_relpath
    pack = prep_models.load_parsed_song(
        data_root,
        entry.normalized_bundle,
        entry.normalized_id,
    )
    if entry.chart_index < 0 or entry.chart_index >= len(pack.charts):
        raise ValueError(
            f"chart_index {entry.chart_index} out of range for {entry.output_relpath}"
        )
    chart_block = pack.charts[entry.chart_index]
    times = np.asarray(chart_block.times_sec, dtype=np.float64)
    slot_matrix = slots.times_to_slot_matrix(
        times,
        pack.metadata.offset_sec,
        pack.metadata.bpm_segments,
    )
    n_beats = slot_matrix.shape[0]
    beat_times = slots.beat_times_sec(
        n_beats,
        pack.metadata.offset_sec,
        pack.metadata.bpm_segments,
    )
    if spec is None:
        spec = features.load_or_compute_logmel(audio_path, cache=cache_features)
    beat_audio = features.zscore_beats(features.beats_to_audio_tensor(spec, beat_times))
    stream = slots.stream_features(
        n_beats,
        int(chart_block.summary.meter),
        pack.metadata.bpm_segments,
    )
    audio_fwd = features.causal_windows(beat_audio, memlen, reverse=False)
    audio_bwd = features.causal_windows(beat_audio, memlen, reverse=True)
    stream_fwd = features.causal_windows(stream, memlen, reverse=False)
    stream_bwd = np.flip(
        features.causal_windows(stream, memlen, reverse=True),
        axis=1,
    )
    return DdclChart(
        song_key=training_index.song_key(
            entry.normalized_bundle,
            entry.normalized_id,
        ),
        difficulty=entry.difficulty.lower(),
        meter=int(chart_block.summary.meter),
        beat_audio=beat_audio,
        stream=stream,
        slots=slot_matrix,
        audio_fwd=audio_fwd,
        audio_bwd=audio_bwd,
        stream_fwd=stream_fwd,
        stream_bwd=stream_bwd,
    )


def load_split_charts(
    dataset_config: config.DdclDatasetConfig,
    split: str,
) -> list[DdclChart]:
    """Load all charts for one split, sharing per-song log-mel.

    Args:
        dataset_config: Dataset config.
        split: ``train`` or ``val``.

    Returns:
        Loaded charts.
    """
    data_root = resolve_data_root(dataset_config)
    entries = list_split_entries(dataset_config, split)
    charts: list[DdclChart] = []
    spec_cache: dict[str, np.ndarray] = {}
    for entry in entries:
        audio_key = str(pathlib.Path(data_root) / entry.audio_relpath)
        spec = spec_cache.get(audio_key)
        if spec is None:
            spec = features.load_or_compute_logmel(
                audio_key,
                cache=dataset_config.cache_features,
            )
            spec_cache[audio_key] = spec
        charts.append(
            load_ddcl_chart(
                entry,
                data_root,
                memlen=dataset_config.memlen,
                cache_features=dataset_config.cache_features,
                spec=spec,
            )
        )
    return charts


def chart_model_inputs(chart: DdclChart) -> dict[str, np.ndarray]:
    """Return a full-chart batch (one row per beat) for ``model.predict``.

    Args:
        chart: Loaded chart.

    Returns:
        Keras input dict with a leading batch axis equal to ``n_beats``.
    """
    return {
        "audio_fwd": np.asarray(chart.audio_fwd, dtype=np.float32),
        "audio_bwd": np.asarray(chart.audio_bwd, dtype=np.float32),
        "stream_fwd": np.asarray(chart.stream_fwd, dtype=np.float32),
        "stream_bwd": np.asarray(chart.stream_bwd, dtype=np.float32),
    }


def sample_train_batch(
    charts: list[DdclChart],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sample random beats across charts.

    Args:
        charts: Loaded train charts.
        batch_size: Beats per batch.
        rng: NumPy generator.

    Returns:
        ``(inputs, slots)`` for ``model.fit``.

    Raises:
        ValueError: If ``charts`` is empty or ``batch_size`` is not positive.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    audio_fwd = []
    audio_bwd = []
    stream_fwd = []
    stream_bwd = []
    labels = []
    for _ in range(batch_size):
        chart = charts[int(rng.integers(0, len(charts)))]
        beat_idx = int(rng.integers(0, chart.n_beats))
        audio_fwd.append(chart.audio_fwd[beat_idx])
        audio_bwd.append(chart.audio_bwd[beat_idx])
        stream_fwd.append(chart.stream_fwd[beat_idx])
        stream_bwd.append(chart.stream_bwd[beat_idx])
        labels.append(chart.slots[beat_idx])
    inputs = {
        "audio_fwd": np.stack(audio_fwd, axis=0),
        "audio_bwd": np.stack(audio_bwd, axis=0),
        "stream_fwd": np.stack(stream_fwd, axis=0),
        "stream_bwd": np.stack(stream_bwd, axis=0),
    }
    return inputs, np.stack(labels, axis=0)
