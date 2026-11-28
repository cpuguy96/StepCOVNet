"""Chart windows for DDCL ConvLSTM placement training and eval."""

from __future__ import annotations

import collections
import dataclasses
import pathlib
import threading
from collections.abc import Callable, Iterator

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, constants, features, slots


class _LazyMertBeatCache:
    """Thread-safe bounded LRU cache of per-song beat-aligned MERT tensors."""

    def __init__(self, max_songs: int) -> None:
        """Initialize the cache with a positive song capacity.

        Args:
            max_songs: Maximum number of songs kept resident in RAM.

        Raises:
            ValueError: If ``max_songs`` is less than 1.
        """
        if max_songs < 1:
            raise ValueError(f"max_songs must be >= 1, got {max_songs}")
        self._max_songs = int(max_songs)
        self._lock = threading.Lock()
        self._cache: collections.OrderedDict[str, np.ndarray] = (
            collections.OrderedDict()
        )

    @property
    def cached_song_count(self) -> int:
        """Return the number of songs currently resident in the LRU cache."""
        with self._lock:
            return len(self._cache)

    def get_or_load(
        self,
        song_key: str,
        loader: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Return cached beat-aligned MERT tensor or load and cache it.

        Args:
            song_key: Unique ``bundle/id`` song identifier.
            loader: Zero-argument callable producing ``(max_beats, n_frames, ...)``.

        Returns:
            Read-only beat-aligned MERT array for ``song_key``.
        """
        with self._lock:
            cached = self._cache.get(song_key)
            if cached is not None:
                self._cache.move_to_end(song_key)
                return cached
        loaded = loader()
        loaded.setflags(write=False)
        with self._lock:
            existing = self._cache.get(song_key)
            if existing is not None:
                self._cache.move_to_end(song_key)
                return existing
            self._cache[song_key] = loaded
            while len(self._cache) > self._max_songs:
                self._cache.popitem(last=False)
            return loaded


@dataclasses.dataclass
class DdclChart:
    """One standard-difficulty chart on the 48-slot beat grid.

    Stores per-beat tensors only. Causal ``memlen+1`` windows are built on
    demand so a full Dataset A split fits in WSL RAM.

    Attributes:
        song_key: ``bundle/id`` identifier.
        difficulty: Lowercase DDR difficulty label.
        meter: Chart ``#METER``.
        beat_audio: Z-scored per-beat log-mel ``(n_beats, 32, 80, 3)`` or
            beat-aligned MERT embeddings ``(n_beats, n_frames, feature_dim)``.
        stream: Per-beat ``[meter, bpm]`` ``(n_beats, 2)``.
        slots: Binary ``M-slot48`` targets ``(n_beats, 48)``.
        memlen: Context beats besides the current one (paper: 15).
        offset_sec: Simfile ``#OFFSET`` for decoding slot centers to seconds.
        bpm_segments: BPM ladder aligned with ``slots``.
        gt_times_sec: Chart onset times in seconds, sorted.
    """

    song_key: str
    difficulty: str
    meter: int
    beat_audio: np.ndarray
    stream: np.ndarray
    slots: np.ndarray
    memlen: int
    offset_sec: float = 0.0
    bpm_segments: tuple[prep_models.BpmSegment, ...] = ()
    gt_times_sec: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    _beat_audio_loader: Callable[[], np.ndarray] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )

    def __getattribute__(self, name: str):
        """Materialize ``beat_audio`` on demand for lazy-loaded MERT charts."""
        if name == "beat_audio":
            value = object.__getattribute__(self, "beat_audio")
            if value.size == 0:
                loader = object.__getattribute__(self, "_beat_audio_loader")
                if loader is not None:
                    return loader()
            return value
        return object.__getattribute__(self, name)

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
    excluded = {
        song_id.strip()
        for song_id in dataset_config.exclude_normalized_ids
        if str(song_id).strip()
    }
    if excluded:
        before = len(entries)
        entries = [entry for entry in entries if entry.normalized_id not in excluded]
        print(
            f"excluded {before - len(entries)} {split} rows ({sorted(excluded)})",
            flush=True,
        )
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


def _make_lazy_song_loader(
    *,
    cache: _LazyMertBeatCache,
    song_key: str,
    audio_key: str,
    beat_times: np.ndarray,
    n_frames: int,
    mert_cache_suffix: str,
) -> Callable[[int], np.ndarray]:
    """Build a per-song lazy MERT loader backed by ``cache``.

    Args:
        cache: Bounded per-split LRU MERT beat cache.
        song_key: Unique song key within the manifest.
        audio_key: Absolute path to the song audio file.
        beat_times: Beat boundary times in seconds.
        n_frames: Sub-beat frames resampled per beat.
        mert_cache_suffix: Sibling filename suffix for cached MERT arrays.

    Returns:
        Callable mapping ``n_beats`` to a sliced beat-aligned MERT tensor.
    """

    def _load_full_song() -> np.ndarray:
        spec = features.load_mert_features(
            audio_key,
            cache_suffix=mert_cache_suffix,
            mmap=True,
        )
        return features.beats_to_mert_tensor(
            spec,
            beat_times,
            n_frames=n_frames,
        )

    def _load_prefix(n_beats: int) -> np.ndarray:
        return cache.get_or_load(song_key, _load_full_song)[:n_beats]

    return _load_prefix


def load_ddcl_chart(
    entry: training_index.TrainingIndexEntry,
    data_root: str,
    *,
    memlen: int,
    cache_features: bool = True,
    spec: np.ndarray | None = None,
    shared_beat_audio: np.ndarray | None = None,
    lazy_beat_audio_loader: Callable[[int], np.ndarray] | None = None,
    pack: prep_models.ParsedSongPack | None = None,
    audio_feature_type: str = constants.AUDIO_FEATURE_LOGMEL,
    n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
    mert_cache_suffix: str = constants.MERT_CACHE_SUFFIX,
) -> DdclChart:
    """Load beat-grid features and ``M-slot48`` targets for one chart.

    Args:
        entry: Manifest row.
        data_root: Prepared output root.
        memlen: Context length for windows.
        cache_features: Write/read ``*.ddc_mel.npy`` beside audio.
        spec: Optional precomputed audio features shared across difficulties.
        shared_beat_audio: Optional per-song beat-aligned MERT tensor sliced
            zero-copy to ``[:n_beats]`` across difficulties of the same song.
        lazy_beat_audio_loader: Optional callable ``(n_beats) -> beat_audio``
            that defers MERT disk loading until ``chart.beat_audio`` is read.
        pack: Optional preloaded parsed song metadata/charts.
        audio_feature_type: Beat audio frontend (``logmel`` or ``mert``).
        n_frames_per_beat: Sub-beat frames resampled per beat.
        mert_cache_suffix: Sibling filename suffix for cached MERT arrays.

    Returns:
        In-memory chart example (or lazy MERT chart when
        ``lazy_beat_audio_loader`` is provided).

    Raises:
        ValueError: If the chart has no onsets or ``audio_feature_type`` is invalid.
    """
    root = pathlib.Path(data_root)
    audio_path = root / entry.audio_relpath
    if pack is None:
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
    beat_audio_loader: Callable[[], np.ndarray] | None = None
    if audio_feature_type == constants.AUDIO_FEATURE_MERT:
        if lazy_beat_audio_loader is not None:
            beat_audio = np.empty(0, dtype=np.float32)

            def _resolve_beats(
                _loader: Callable[[int], np.ndarray] = lazy_beat_audio_loader,
                _n_beats: int = n_beats,
            ) -> np.ndarray:
                return _loader(_n_beats)

            beat_audio_loader = _resolve_beats
        elif shared_beat_audio is not None and shared_beat_audio.shape[0] >= n_beats:
            beat_audio = shared_beat_audio[:n_beats]
        else:
            beat_times = slots.beat_times_sec(
                n_beats,
                pack.metadata.offset_sec,
                pack.metadata.bpm_segments,
            )
            if spec is None:
                spec = features.load_mert_features(
                    audio_path, cache_suffix=mert_cache_suffix
                )
            beat_audio = features.beats_to_mert_tensor(
                spec,
                beat_times,
                n_frames=n_frames_per_beat,
            )
    elif audio_feature_type == constants.AUDIO_FEATURE_LOGMEL:
        beat_times = slots.beat_times_sec(
            n_beats,
            pack.metadata.offset_sec,
            pack.metadata.bpm_segments,
        )
        if spec is None:
            spec = features.load_or_compute_logmel(audio_path, cache=cache_features)
        beat_audio = features.zscore_beats(
            features.beats_to_audio_tensor(
                spec,
                beat_times,
                n_frames=n_frames_per_beat,
            )
        )
    else:
        raise ValueError(f"unsupported audio_feature_type: {audio_feature_type!r}")
    stream = slots.stream_features(
        n_beats,
        int(chart_block.summary.meter),
        pack.metadata.bpm_segments,
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
        memlen=memlen,
        offset_sec=float(pack.metadata.offset_sec),
        bpm_segments=tuple(pack.metadata.bpm_segments),
        gt_times_sec=np.sort(times),
        _beat_audio_loader=beat_audio_loader,
    )


def load_split_charts(
    dataset_config: config.DdclDatasetConfig,
    split: str,
) -> list[DdclChart]:
    """Load all charts for one split, sharing per-song audio features.

    Args:
        dataset_config: Dataset config.
        split: ``train`` or ``val``.

    Returns:
        Loaded charts.
    """
    data_root = resolve_data_root(dataset_config)
    entries = list_split_entries(dataset_config, split)
    grouped: dict[str, list[training_index.TrainingIndexEntry]] = {}
    song_order: list[str] = []
    for entry in entries:
        key = training_index.song_key(entry.normalized_bundle, entry.normalized_id)
        if key not in grouped:
            grouped[key] = []
            song_order.append(key)
        grouped[key].append(entry)
    use_lazy_mert = (
        dataset_config.audio_feature_type == constants.AUDIO_FEATURE_MERT
        and dataset_config.lazy_mert_cache_size > 0
    )
    lazy_cache = (
        _LazyMertBeatCache(dataset_config.lazy_mert_cache_size)
        if use_lazy_mert
        else None
    )
    charts: list[DdclChart] = []
    verbose_songs = len(song_order) <= 100
    for song_idx, song_key in enumerate(song_order, start=1):
        group = grouped[song_key]
        audio_key = str(pathlib.Path(data_root) / group[0].audio_relpath)
        pack = prep_models.load_parsed_song(
            data_root,
            group[0].normalized_bundle,
            group[0].normalized_id,
        )
        spec: np.ndarray | None = None
        shared_beat_audio: np.ndarray | None = None
        lazy_loader: Callable[[int], np.ndarray] | None = None
        if dataset_config.audio_feature_type == constants.AUDIO_FEATURE_MERT:
            max_n_beats = 0
            for entry in group:
                if 0 <= entry.chart_index < len(pack.charts):
                    chart_times = np.asarray(
                        pack.charts[entry.chart_index].times_sec,
                        dtype=np.float64,
                    )
                    if chart_times.size > 0:
                        n_beats_i = slots.times_to_slot_matrix(
                            chart_times,
                            pack.metadata.offset_sec,
                            pack.metadata.bpm_segments,
                        ).shape[0]
                        max_n_beats = max(max_n_beats, n_beats_i)
            if lazy_cache is not None:
                features.validate_mert_features(
                    audio_key, cache_suffix=dataset_config.mert_cache_suffix
                )
                if max_n_beats > 0:
                    song_beat_times = slots.beat_times_sec(
                        max_n_beats,
                        pack.metadata.offset_sec,
                        pack.metadata.bpm_segments,
                    )
                    lazy_loader = _make_lazy_song_loader(
                        cache=lazy_cache,
                        song_key=song_key,
                        audio_key=audio_key,
                        beat_times=song_beat_times,
                        n_frames=dataset_config.n_frames_per_beat,
                        mert_cache_suffix=dataset_config.mert_cache_suffix,
                    )
            else:
                spec = features.load_mert_features(
                    audio_key, cache_suffix=dataset_config.mert_cache_suffix
                )
                if max_n_beats > 0:
                    shared_beat_audio = features.beats_to_mert_tensor(
                        spec,
                        slots.beat_times_sec(
                            max_n_beats,
                            pack.metadata.offset_sec,
                            pack.metadata.bpm_segments,
                        ),
                        n_frames=dataset_config.n_frames_per_beat,
                    )
                    shared_beat_audio.setflags(write=False)
        else:
            spec = features.load_or_compute_logmel(
                audio_key,
                cache=dataset_config.cache_features,
            )
        for entry in group:
            charts.append(
                load_ddcl_chart(
                    entry,
                    data_root,
                    memlen=dataset_config.memlen,
                    cache_features=dataset_config.cache_features,
                    spec=spec,
                    shared_beat_audio=shared_beat_audio,
                    lazy_beat_audio_loader=lazy_loader,
                    pack=pack,
                    audio_feature_type=dataset_config.audio_feature_type,
                    n_frames_per_beat=dataset_config.n_frames_per_beat,
                    mert_cache_suffix=dataset_config.mert_cache_suffix,
                )
            )
        del spec
        if verbose_songs or song_idx % 100 == 0 or song_idx == len(song_order):
            mode_tag = " [lazy]" if lazy_cache is not None else ""
            print(
                f"loaded {split} {song_idx}/{len(song_order)} {song_key}: "
                f"{len(group)} charts, {charts[-1].n_beats} beats{mode_tag}",
                flush=True,
            )
    print(
        f"loaded {len(charts)} {split} charts from {len(song_order)} songs",
        flush=True,
    )
    return charts


def windows_for_indices(
    chart: DdclChart,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build placement inputs for selected beats without a full-chart stack.

    Args:
        chart: Loaded chart.
        indices: Beat indices, shape ``(batch,)``.

    Returns:
        Model input dict with a leading batch axis.

    Raises:
        ValueError: If ``indices`` is empty.
    """
    beat_idxs = np.asarray(indices, dtype=np.int64).reshape(-1)
    if beat_idxs.size < 1:
        raise ValueError("indices must contain at least one beat")
    audio_fwd = []
    audio_bwd = []
    stream_fwd = []
    stream_bwd = []
    memlen = chart.memlen
    audio_pad = 0.0 if chart.beat_audio.ndim == 3 else None
    for beat_idx in beat_idxs:
        idx = int(beat_idx)
        audio_fwd.append(
            features.window_at_beat(
                chart.beat_audio,
                memlen,
                idx,
                reverse=False,
                pad_value=audio_pad,
            )
        )
        audio_bwd.append(
            features.window_at_beat(
                chart.beat_audio,
                memlen,
                idx,
                reverse=True,
                pad_value=audio_pad,
            )
        )
        stream_fwd.append(
            features.window_at_beat(chart.stream, memlen, idx, reverse=False)
        )
        stream_bwd.append(
            np.flip(
                features.window_at_beat(chart.stream, memlen, idx, reverse=True),
                axis=0,
            )
        )
    return {
        "audio_fwd": np.stack(audio_fwd, axis=0).astype(np.float32),
        "audio_bwd": np.stack(audio_bwd, axis=0).astype(np.float32),
        "stream_fwd": np.stack(stream_fwd, axis=0).astype(np.float32),
        "stream_bwd": np.stack(stream_bwd, axis=0).astype(np.float32),
    }


def chart_model_inputs(chart: DdclChart) -> dict[str, np.ndarray]:
    """Return a full-chart batch (one row per beat) for ``model.predict``.

    Args:
        chart: Loaded chart.

    Returns:
        Model input dict with a leading batch axis equal to ``n_beats``.
    """
    return windows_for_indices(chart, np.arange(chart.n_beats, dtype=np.int64))


def iter_window_batches(
    chart: DdclChart,
    *,
    batch_size: int = constants.PREDICT_BEAT_BATCH,
) -> Iterator[np.ndarray]:
    """Yield beat-index batches for GPU-safe eval.

    Args:
        chart: Loaded chart.
        batch_size: Beats per predict call.

    Yields:
        np.ndarray: Integer beat indices of shape ``(batch,)``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    n_beats = chart.n_beats
    for start in range(0, n_beats, batch_size):
        end = min(n_beats, start + batch_size)
        yield np.arange(start, end, dtype=np.int64)


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
        windows = windows_for_indices(chart, np.array([beat_idx], dtype=np.int64))
        audio_fwd.append(windows["audio_fwd"][0])
        audio_bwd.append(windows["audio_bwd"][0])
        stream_fwd.append(windows["stream_fwd"][0])
        stream_bwd.append(windows["stream_bwd"][0])
        labels.append(chart.slots[beat_idx])
    inputs = {
        "audio_fwd": np.stack(audio_fwd, axis=0),
        "audio_bwd": np.stack(audio_bwd, axis=0),
        "stream_fwd": np.stack(stream_fwd, axis=0),
        "stream_bwd": np.stack(stream_bwd, axis=0),
    }
    return inputs, np.stack(labels, axis=0)
