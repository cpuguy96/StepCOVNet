"""Beat tracking and ITGPT-MERT onset inference on raw audio files.

Integrates the neural beat/downbeat tracker ``beat_this`` (Foscarin et al.,
ISMIR 2024) as an optional dependency with a ``librosa`` fallback, plus
beat-grid MERT alignment and 48-slot timestamp decoding for end-to-end chart
generation without a pre-existing ``.sm`` beat grid.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib
from typing import Any

import librosa
import numpy as np
import torch

from stepcovnet import constants, ssl_features
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import features as ddcl_features
from stepcovnet.itgpt import constants as itgpt_constants
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import evaluation as itgpt_eval

TRACKER_AUTO = "auto"
TRACKER_BEAT_THIS = "beat_this"
TRACKER_LIBROSA = "librosa"
VALID_TRACKERS = (TRACKER_AUTO, TRACKER_BEAT_THIS, TRACKER_LIBROSA)

DEFAULT_MERT_LAYERS: tuple[int, ...] = (2, 6, 10, 16)
DEFAULT_MERT_CACHE_SUFFIX = ".mert_l2_6_10_16.npy"
_BEAT_THIS_INSTALL_HINT = (
    "pip install soxr rotary-embedding-torch "
    "https://github.com/CPJKU/beat_this/archive/main.zip"
)


@dataclasses.dataclass(frozen=True)
class BeatGrid:
    """Estimated beat and downbeat timestamps for an audio track.

    Attributes:
        beat_times: Strictly increasing 1D float64 beat timestamps in seconds.
        downbeat_times: 1D float64 downbeat timestamps in seconds (may be empty).
        bpm: Median tempo in beats per minute derived from ``beat_times``.
        tracker: Backend name that produced this grid (``beat_this`` or ``librosa``).
    """

    beat_times: np.ndarray
    downbeat_times: np.ndarray
    bpm: float
    tracker: str

    def __post_init__(self) -> None:
        """Validate that the beat grid contains at least two strictly increasing beats.

        Raises:
            ValueError: If ``beat_times`` has fewer than two entries or ``bpm <= 0``.
        """
        bt = np.asarray(self.beat_times, dtype=np.float64).reshape(-1)
        if bt.size < 2:
            raise ValueError(
                f"BeatGrid requires at least 2 beats, got {bt.size} from {self.tracker!r}"
            )
        if not np.all(np.diff(bt) > 0.0):
            raise ValueError("BeatGrid.beat_times must be strictly increasing")
        if not (self.bpm > 0.0 and np.isfinite(self.bpm)):
            raise ValueError(f"BeatGrid.bpm must be positive finite, got {self.bpm}")

    @property
    def n_beats(self) -> int:
        """Number of beat intervals represented by this grid."""
        return int(self.beat_times.size)

    @property
    def beat_times_with_end(self) -> np.ndarray:
        """Beat start times plus an exclusive end timestamp (length ``n_beats + 1``)."""
        bt = np.asarray(self.beat_times, dtype=np.float64).reshape(-1)
        median_ibi = float(np.median(np.diff(bt)))
        return np.concatenate([bt, [bt[-1] + median_ibi]])

    def per_beat_bpm(
        self,
        *,
        min_bpm: float = 30.0,
        max_bpm: float = 300.0,
    ) -> np.ndarray:
        """Return per-beat local BPM values clamped to ``[min_bpm, max_bpm]``.

        Args:
            min_bpm: Lower BPM clamp.
            max_bpm: Upper BPM clamp.

        Returns:
            Float32 array of shape ``(n_beats,)``.
        """
        ibis = np.diff(self.beat_times_with_end)
        bpms = 60.0 / np.maximum(ibis, 1e-3)
        return np.clip(bpms, min_bpm, max_bpm).astype(np.float32)


def _clean_beat_times(
    raw_beats: np.ndarray, *, min_spacing_sec: float = 0.05
) -> np.ndarray:
    """Sort and deduplicate beat timestamps so consecutive beats are strictly increasing.

    Args:
        raw_beats: Raw beat timestamps in seconds.
        min_spacing_sec: Minimum allowed gap between consecutive beats.

    Returns:
        Sorted, strictly increasing 1D float64 array.
    """
    bt = np.sort(np.asarray(raw_beats, dtype=np.float64).reshape(-1))
    bt = bt[np.isfinite(bt) & (bt >= 0.0)]
    if bt.size < 2:
        return bt
    keep = np.concatenate([[True], np.diff(bt) > float(min_spacing_sec)])
    return bt[keep]


def beat_this_available() -> bool:
    """Return True if the optional ``beat_this`` package can be imported."""
    return importlib.util.find_spec("beat_this") is not None


def estimate_beats_beat_this(
    audio_path: str,
    *,
    device: str | torch.device = "auto",
    checkpoint_path: str = "final0",
    dbn: bool = False,
) -> BeatGrid:
    """Estimate beats and downbeats from raw audio using ``beat_this`` (ISMIR 2024).

    Args:
        audio_path: Path to the input audio file.
        device: Target compute device (``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``).
        checkpoint_path: Pretrained ``beat_this`` checkpoint identifier.
        dbn: Whether to enable optional DBN post-processing.

    Returns:
        :class:`BeatGrid` populated with neural beat and downbeat timestamps.

    Raises:
        ImportError: If ``beat_this`` is not installed.
        ValueError: If fewer than two beats are detected.
    """
    try:
        bt_mod = importlib.import_module("beat_this.inference")
    except ImportError as exc:
        raise ImportError(
            "Optional dependency 'beat_this' is not installed. Install it with: "
            f"{_BEAT_THIS_INSTALL_HINT}"
        ) from exc

    resolved = runtime.resolve_device(None if device == "auto" else device)
    # beat_this expects a torch device string like "cuda" or "cpu".
    bt_device = "cuda" if resolved.type == "cuda" else "cpu"
    tracker = bt_mod.File2Beats(
        checkpoint_path=checkpoint_path,
        device=bt_device,
        dbn=dbn,
    )
    raw_beats, raw_downbeats = tracker(str(audio_path))
    beats = _clean_beat_times(np.asarray(raw_beats, dtype=np.float64))
    downbeats = _clean_beat_times(np.asarray(raw_downbeats, dtype=np.float64))
    if beats.size < 2:
        raise ValueError(
            f"beat_this detected fewer than 2 beats for {audio_path!r} (got {beats.size})."
        )
    median_ibi = float(np.median(np.diff(beats)))
    bpm = 60.0 / max(median_ibi, 1e-6)
    return BeatGrid(
        beat_times=beats,
        downbeat_times=downbeats,
        bpm=bpm,
        tracker=TRACKER_BEAT_THIS,
    )


def estimate_beats_librosa(audio_path: str) -> BeatGrid:
    """Estimate beats from raw audio using ``librosa.beat.beat_track``.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        :class:`BeatGrid` populated with ``librosa`` beat timestamps.

    Raises:
        ValueError: If tempo or beats cannot be estimated.
    """
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = _clean_beat_times(np.asarray(beat_times, dtype=np.float64))
    tempo_f = float(np.atleast_1d(tempo)[0])
    if beats.size < 2 or not (tempo_f > 0.0 and np.isfinite(tempo_f)):
        raise ValueError(
            f"librosa could not estimate a valid beat grid for {audio_path!r}."
        )
    median_ibi = float(np.median(np.diff(beats)))
    bpm = 60.0 / max(median_ibi, 1e-6)
    return BeatGrid(
        beat_times=beats,
        downbeat_times=np.zeros(0, dtype=np.float64),
        bpm=bpm,
        tracker=TRACKER_LIBROSA,
    )


def snap_beats_to_onsets(
    beat_grid: BeatGrid,
    audio_path: str,
    *,
    max_shift_ms: float = 25.0,
    step_ms: float = 1.0,
) -> BeatGrid:
    """Apply a constant song-level phase shift aligning beats to audio transients.

    Neural beat trackers trained on MIR beat annotations often sit ~10-15 ms
    behind sharp percussive attack transients expected by rhythm-game charts.
    This searches a single scalar shift ``delta in [-max_shift_ms, +max_shift_ms]``
    maximizing mean onset strength at ``beat_times + delta`` without altering
    inter-beat spacing.

    Args:
        beat_grid: Input :class:`BeatGrid`.
        audio_path: Path to the source audio file.
        max_shift_ms: Maximum absolute shift in milliseconds.
        step_ms: Grid search resolution in milliseconds.

    Returns:
        New :class:`BeatGrid` shifted by the optimal scalar offset.
    """
    if max_shift_ms <= 0.0 or step_ms <= 0.0:
        return beat_grid
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    hop_length = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if onset_env.size == 0:
        return beat_grid
    env_times = librosa.frames_to_time(
        np.arange(onset_env.size), sr=sr, hop_length=hop_length
    )

    shifts_sec = (
        np.arange(-max_shift_ms, max_shift_ms + 0.5 * step_ms, step_ms) / 1000.0
    )
    best_shift = 0.0
    best_score = -float("inf")
    bt = beat_grid.beat_times
    for shift in shifts_sec:
        shifted = bt + float(shift)
        if np.any(shifted < 0.0):
            continue
        score = float(np.mean(np.interp(shifted, env_times, onset_env)))
        if score > best_score:
            best_score = score
            best_shift = float(shift)

    if abs(best_shift) < 1e-6:
        return beat_grid
    shifted_beats = np.maximum(0.0, beat_grid.beat_times + best_shift)
    shifted_downbeats = (
        np.maximum(0.0, beat_grid.downbeat_times + best_shift)
        if beat_grid.downbeat_times.size > 0
        else beat_grid.downbeat_times
    )
    return BeatGrid(
        beat_times=shifted_beats,
        downbeat_times=shifted_downbeats,
        bpm=beat_grid.bpm,
        tracker=beat_grid.tracker,
    )


def estimate_beats(
    audio_path: str,
    *,
    tracker: str = TRACKER_AUTO,
    device: str | torch.device = "auto",
    snap_to_onsets: bool = False,
) -> BeatGrid:
    """Estimate a :class:`BeatGrid` from raw audio using the selected backend.

    Args:
        audio_path: Path to the input audio file.
        tracker: Backend selector: ``"auto"`` (prefers ``beat_this`` when
            installed, else falls back to ``librosa``), ``"beat_this"``, or
            ``"librosa"``.
        device: Compute device forwarded to ``beat_this``.
        snap_to_onsets: If True, apply :func:`snap_beats_to_onsets` to correct
            constant beat-annotation phase bias.

    Returns:
        Estimated :class:`BeatGrid`.

    Raises:
        ValueError: If ``tracker`` is not one of :data:`VALID_TRACKERS`.
    """
    if tracker not in VALID_TRACKERS:
        raise ValueError(
            f"beat tracker must be one of {VALID_TRACKERS}, got {tracker!r}"
        )
    if tracker == TRACKER_BEAT_THIS:
        grid = estimate_beats_beat_this(audio_path, device=device)
    elif tracker == TRACKER_LIBROSA:
        grid = estimate_beats_librosa(audio_path)
    elif beat_this_available():
        grid = estimate_beats_beat_this(audio_path, device=device)
    else:
        grid = estimate_beats_librosa(audio_path)

    if snap_to_onsets:
        grid = snap_beats_to_onsets(grid, audio_path)
    return grid


def decode_itgpt_slots_to_times(
    probs: np.ndarray,
    beat_grid: BeatGrid,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Decode ``(n_beats, n_slots)`` ITGPT probabilities to sorted onset times.

    Matches :func:`stepcovnet.ddcl.slots.occupied_center_times` by placing each
    occupied slot at its cell center ``(slot + 0.5) / n_slots`` linearly
    interpolated between consecutive beats of ``beat_grid``.

    Args:
        probs: Slot occupancy probabilities of shape ``(n_beats, n_slots)``.
        beat_grid: :class:`BeatGrid` corresponding to ``probs``.
        threshold: Probability cutoff for occupied slots.

    Returns:
        Sorted 1D float64 array of onset timestamps in seconds.

    Raises:
        ValueError: If ``probs`` is not 2D or exceeds ``beat_grid.n_beats``.
    """
    grid = np.asarray(probs, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] < 1:
        raise ValueError(f"probs must be 2D (n_beats, n_slots), got {grid.shape}")
    bt_full = beat_grid.beat_times_with_end
    if grid.shape[0] > bt_full.size - 1:
        raise ValueError(
            f"probs has {grid.shape[0]} beats but beat_grid only has {bt_full.size - 1}"
        )
    n_slots = int(grid.shape[1])
    times: list[float] = []
    for beat_idx, slot_idx in np.argwhere(grid >= float(threshold)):
        t0 = float(bt_full[int(beat_idx)])
        t1 = float(bt_full[int(beat_idx) + 1])
        frac = (float(slot_idx) + 0.5) / float(n_slots)
        times.append(t0 + frac * (t1 - t0))
    if not times:
        return np.zeros(0, dtype=np.float64)
    return np.sort(np.asarray(times, dtype=np.float64))


def _load_or_extract_mert_for_audio(
    audio_path: str,
    *,
    mert_layers: tuple[int, ...] = DEFAULT_MERT_LAYERS,
    mert_cache_suffix: str = DEFAULT_MERT_CACHE_SUFFIX,
    device: str = "cpu",
) -> np.ndarray:
    """Load sibling cached MERT features if present, else extract on the fly."""
    sibling = pathlib.Path(audio_path).with_suffix(mert_cache_suffix)
    if sibling.is_file():
        return ddcl_features.load_mert_features(
            audio_path, cache_suffix=mert_cache_suffix
        )
    return ssl_features.extract_mert_features_from_audio(
        audio_path,
        layers=mert_layers,
        device=device,
    )


def predict_itgpt_onsets_from_audio(
    audio_path: str,
    itgpt_model: Any,
    *,
    beat_grid: BeatGrid | None = None,
    beat_tracker: str = TRACKER_AUTO,
    snap_beats: bool = False,
    target_density: float = 2.0,
    meter: int = 10,
    threshold: float = 0.5,
    max_beats: int | None = None,
    mert_layers: tuple[int, ...] = DEFAULT_MERT_LAYERS,
    mert_cache_suffix: str = DEFAULT_MERT_CACHE_SUFFIX,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, BeatGrid]:
    """Run Beat This! + MERT + ITGPT-MERT onset placement on raw audio.

    Args:
        audio_path: Path to the raw audio file.
        itgpt_model: Trained PyTorch :class:`ItgptPlacementModel`.
        beat_grid: Optional precomputed :class:`BeatGrid` (estimated when None).
        beat_tracker: Beat tracker backend (``"auto"``, ``"beat_this"``, ``"librosa"``).
        snap_beats: Apply song-level onset-strength phase alignment to beats.
        target_density: Desired average steps per beat for density-conditioned
            ITGPT models (e.g. ``0.5`` Beginner, ``1.5`` Medium, ``2.0`` Hard,
            ``3.0`` Challenge).
        meter: Difficulty meter for meter-conditioned ITGPT models.
        threshold: Slot probability cutoff (default ``0.5``).
        max_beats: Maximum beat chunk length (defaults to model's ``max_beats``).
        mert_layers: MERT hidden layers to average when extracting on the fly.
        mert_cache_suffix: Sibling MERT cache suffix checked before extraction.
        device: Optional compute device override.

    Returns:
        Tuple ``(onset_times_sec, beat_grid)``.

    Raises:
        ValueError: If ``target_density <= 0``.
    """
    if target_density <= 0.0:
        raise ValueError(f"target_density must be positive, got {target_density}")

    target_device = runtime.resolve_device(device)
    grid = (
        beat_grid
        if beat_grid is not None
        else estimate_beats(
            audio_path,
            tracker=beat_tracker,
            device=target_device,
            snap_to_onsets=snap_beats,
        )
    )
    mert_100hz = _load_or_extract_mert_for_audio(
        audio_path,
        mert_layers=mert_layers,
        mert_cache_suffix=mert_cache_suffix,
        device="cuda" if target_device.type == "cuda" else "cpu",
    )
    n_frames = int(getattr(itgpt_model, "n_frames_per_beat", 24))
    beat_mert = ddcl_features.beats_to_mert_tensor(
        mert_100hz, grid.beat_times_with_end, n_frames=n_frames
    ).astype(np.float32)

    n_beats = int(beat_mert.shape[0])
    stream = np.zeros((n_beats, 2), dtype=np.float32)
    stream[:, 0] = float(meter)
    stream[:, 1] = grid.per_beat_bpm()[:n_beats]

    # Populate dummy_slots with the exact target total step count so
    # density-conditioned ITGPT models receive mean(steps_per_beat) == target_density.
    dummy_slots = np.zeros((n_beats, itgpt_constants.N_SLOTS), dtype=np.float32)
    target_n_steps = max(1, int(round(float(target_density) * float(n_beats))))
    flat_slots = dummy_slots.reshape(-1)
    flat_slots[: min(target_n_steps, flat_slots.size)] = 1.0

    synth_chart = ddcl_datasets.DdclChart(
        song_key=pathlib.Path(audio_path).stem,
        difficulty="generated",
        meter=int(meter),
        beat_audio=beat_mert,
        stream=stream,
        slots=dummy_slots,
        memlen=0,
    )

    effective_max_beats = int(
        max_beats
        if max_beats is not None
        else getattr(itgpt_model, "max_beats", itgpt_constants.MAX_BEATS)
    )
    probs = itgpt_eval.predict_chart_slots(
        itgpt_model,
        synth_chart,
        max_beats=effective_max_beats,
        device=target_device,
    )
    onset_times = decode_itgpt_slots_to_times(probs, grid, threshold=threshold)
    return onset_times, grid
