"""Beat tracking and ITGPT-MERT onset inference on raw audio files.

Integrates a self-contained pure-PyTorch implementation of the neural
beat/downbeat tracker ``beat_this`` (Foscarin et al., ISMIR 2024;
:mod:`stepcovnet.beat_this_model`) alongside a ``librosa`` fallback, plus
beat-grid MERT alignment and 48-slot timestamp decoding for end-to-end chart
generation without a pre-existing ``.sm`` beat grid.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import pathlib
import warnings
from typing import Any

import librosa
import numpy as np
import torch

from stepcovnet import beat_this_model, constants, ssl_features, trainers
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import features as ddcl_features
from stepcovnet.itgpt import constants as itgpt_constants
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import evaluation as itgpt_eval
from stepcovnet.pytorch.itgpt import trainers as itgpt_trainers

TRACKER_AUTO = "auto"
TRACKER_BEAT_THIS = "beat_this"
TRACKER_LIBROSA = "librosa"
VALID_TRACKERS = (TRACKER_AUTO, TRACKER_BEAT_THIS, TRACKER_LIBROSA)

DEFAULT_MERT_LAYERS: tuple[int, ...] = (2, 6, 10, 16)
DEFAULT_MERT_CACHE_SUFFIX = ".mert_l2_6_10_16.npy"


@dataclasses.dataclass(frozen=True)
class BeatGrid:
    """Estimated beat and downbeat timestamps for an audio track.

    Attributes:
        beat_times: Strictly increasing 1D float64 beat timestamps in seconds.
        downbeat_times: 1D float64 downbeat timestamps in seconds (may be empty).
        bpm: Tempo in beats per minute derived from ``beat_times``.
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

    @property
    def anchor_time_sec(self) -> float:
        """Measure-1 beat-0 timestamp in seconds (first downbeat if available)."""
        if self.downbeat_times.size > 0:
            return float(self.downbeat_times[0])
        return float(self.beat_times[0])

    @property
    def sm_offset_sec(self) -> float:
        """StepMania ``#OFFSET`` in seconds (``-anchor_time_sec``)."""
        return -self.anchor_time_sec

    def times_to_beats(
        self,
        times: np.ndarray,
        *,
        anchor_time_sec: float | None = None,
    ) -> np.ndarray:
        """Map onset timestamps in seconds to continuous beat indices.

        Uses piecewise-linear interpolation across ``self.beat_times`` and
        extrapolates linearly beyond the first/last beat using the boundary
        inter-beat interval. Beat ``0.0`` corresponds to ``anchor_time_sec``
        (defaulting to :attr:`anchor_time_sec`, i.e., the first downbeat).

        Args:
            times: 1D array of timestamps in seconds.
            anchor_time_sec: Optional override for the beat-0 timestamp.

        Returns:
            Float64 array of continuous beat indices with the same shape as
            ``times``.
        """
        t_arr = np.asarray(times, dtype=np.float64)
        bt = np.asarray(self.beat_times, dtype=np.float64).reshape(-1)
        idx_grid = np.arange(bt.size, dtype=np.float64)

        def _interp(vals: np.ndarray) -> np.ndarray:
            out = np.interp(vals, bt, idx_grid)
            first_ibi = max(float(bt[1] - bt[0]), 1e-6)
            last_ibi = max(float(bt[-1] - bt[-2]), 1e-6)
            before = vals < bt[0]
            if np.any(before):
                out = np.where(before, (vals - bt[0]) / first_ibi, out)
            after = vals > bt[-1]
            if np.any(after):
                out = np.where(after, idx_grid[-1] + (vals - bt[-1]) / last_ibi, out)
            return out

        anchor = (
            self.anchor_time_sec if anchor_time_sec is None else float(anchor_time_sec)
        )
        anchor_beat = float(_interp(np.asarray([anchor], dtype=np.float64))[0])
        return _interp(t_arr) - anchor_beat

    def beat_phase_for_times(self, times: np.ndarray) -> np.ndarray:
        """Compute fractional beat phase in ``[0, 1)`` for each timestamp.

        Args:
            times: 1D array of onset timestamps in seconds.

        Returns:
            Float32 array of beat phases in ``[0, 1)``.
        """
        beats = self.times_to_beats(times)
        return np.mod(beats, 1.0).astype(np.float32)


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


def _robust_bpm_from_beats(beats: np.ndarray) -> float:
    """Compute robust tempo in BPM from beat timestamps.

    Averages inter-beat intervals within 15% of the median IBI to eliminate
    20 ms frame-quantization bias (at 50 FPS) while remaining immune to breaks,
    silences, or double-time outliers.

    Args:
        beats: Strictly increasing 1D beat timestamps in seconds.

    Returns:
        Estimated tempo in beats per minute.
    """
    ibis = np.diff(beats)
    median_ibi = float(np.median(ibis))
    inliers = ibis[np.abs(ibis - median_ibi) <= 0.15 * max(median_ibi, 1e-6)]
    effective_ibi = float(np.mean(inliers)) if inliers.size > 0 else median_ibi
    return 60.0 / max(effective_ibi, 1e-6)


def beat_this_available() -> bool:
    """Return True — Beat This! is built into StepCOVNet with no extra packages."""
    return True


def estimate_beats_beat_this(
    audio_path: str,
    *,
    device: str | torch.device = "auto",
    checkpoint_path: str = beat_this_model.DEFAULT_CHECKPOINT,
    float16: bool = False,
) -> BeatGrid:
    """Estimate beats and downbeats using the built-in Beat This! model.

    Args:
        audio_path: Path to the input audio file.
        device: Target compute device (``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``).
        checkpoint_path: Pretrained Beat This! checkpoint identifier, URL, or
            local file path (default ``"final0"``).
        float16: Enable fp16 autocast during transformer inference.

    Returns:
        :class:`BeatGrid` populated with neural beat and downbeat timestamps.

    Raises:
        ValueError: If fewer than two beats are detected or the checkpoint
            cannot be loaded.
    """
    resolved = runtime.resolve_device(None if device == "auto" else device)
    raw_beats, raw_downbeats = beat_this_model.predict_beats_from_audio(
        str(audio_path),
        checkpoint_path=checkpoint_path,
        device=resolved,
        float16=float16,
    )
    beats = _clean_beat_times(np.asarray(raw_beats, dtype=np.float64))
    downbeats = _clean_beat_times(np.asarray(raw_downbeats, dtype=np.float64))
    if beats.size < 2:
        raise ValueError(
            f"beat_this detected fewer than 2 beats for {audio_path!r} (got {beats.size})."
        )
    return BeatGrid(
        beat_times=beats,
        downbeat_times=downbeats,
        bpm=_robust_bpm_from_beats(beats),
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


def _compute_attack_onset_envelope(
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a zero-lag short-window attack-onset envelope and its timestamps.

    Combines a short-window causal log-RMS attack jump (for percussive kicks and
    snares) with short-window centered log-mel spectral flux (for tonal/melodic
    attacks without large broadband RMS jumps). Unlike default
    ``librosa.onset.onset_strength`` (``n_fft=2048`` backward diff), this peaks
    within ~1–2 ms of the physical waveform transient leading edge.

    Args:
        y: Mono 1D audio waveform.
        sr: Sample rate in Hz.

    Returns:
        Tuple ``(envelope, env_times_sec)`` on a ~1.5 ms frame grid.
    """
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if y_arr.size < 256:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    rms_hop = 64
    rms = librosa.feature.rms(
        y=y_arr, frame_length=256, hop_length=rms_hop, center=True
    )[0]
    n_rms = int(rms.size)
    if n_rms < 4:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    log_rms = np.log(rms.astype(np.float64) + 1e-4)
    k = 4  # ~5.8 ms pre/post comparison window at 44.1 kHz
    pad = np.pad(log_rms, (k, k), mode="edge")
    cs = np.concatenate([[0.0], np.cumsum(pad)])
    pre = (cs[k : k + n_rms] - cs[:n_rms]) / float(k)
    post = (cs[2 * k : 2 * k + n_rms] - cs[k : k + n_rms]) / float(k)
    e_rms = np.maximum(0.0, post - pre)
    t_rms = librosa.frames_to_time(np.arange(n_rms), sr=sr, hop_length=rms_hop)

    env = np.zeros(n_rms, dtype=np.float64)
    rms_std = float(np.std(e_rms))
    if rms_std > 1e-12:
        env += 0.5 * (e_rms / rms_std)

    mel_hop = 128
    if y_arr.size >= 512:
        mel = librosa.feature.melspectrogram(
            y=y_arr,
            sr=sr,
            n_fft=512,
            hop_length=mel_hop,
            n_mels=64,
            power=1.0,
        )
        if mel.shape[1] >= 3:
            log_mel = np.log1p(10.0 * mel.astype(np.float64))
            e_mel = np.maximum(0.0, log_mel[:, 2:] - log_mel[:, :-2]).mean(axis=0)
            t_mel = librosa.frames_to_time(
                np.arange(1, 1 + e_mel.size), sr=sr, hop_length=mel_hop
            )
            e_mel_interp = np.interp(t_rms, t_mel, e_mel)
            mel_std = float(np.std(e_mel_interp))
            if mel_std > 1e-12:
                env += 0.5 * (e_mel_interp / mel_std)

    return env, t_rms


def snap_beats_to_onsets(
    beat_grid: BeatGrid,
    audio_path: str,
    *,
    max_shift_ms: float = 35.0,
    step_ms: float = 0.5,
) -> BeatGrid:
    """Apply a constant song-level phase shift aligning beats to audio attacks.

    Neural beat trackers trained on MIR beat annotations often sit ~15-30 ms
    behind sharp percussive attack transients expected by rhythm-game charts.
    This searches a single scalar shift ``delta in [-max_shift_ms, +max_shift_ms]``
    maximizing mean zero-lag attack-onset strength at ``beat_times + delta``
    without altering inter-beat spacing.

    Args:
        beat_grid: Input :class:`BeatGrid`.
        audio_path: Path to the source audio file.
        max_shift_ms: Maximum absolute shift in milliseconds (default ``35.0``).
        step_ms: Grid search resolution in milliseconds (default ``0.5``).

    Returns:
        New :class:`BeatGrid` shifted by the optimal scalar offset.
    """
    if max_shift_ms <= 0.0 or step_ms <= 0.0:
        return beat_grid
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    onset_env, env_times = _compute_attack_onset_envelope(y, int(sr))
    if onset_env.size == 0 or float(np.max(onset_env)) <= 0.0:
        return beat_grid

    max_shift_sec = float(max_shift_ms) / 1000.0
    env_end = float(env_times[-1])
    bt = beat_grid.beat_times
    interior = bt[(bt >= max_shift_sec) & (bt <= env_end - max_shift_sec)]
    score_bt = interior if interior.size > 0 else bt

    shifts_sec = (
        np.arange(-max_shift_ms, max_shift_ms + 0.5 * step_ms, step_ms) / 1000.0
    )
    best_shift = 0.0
    best_score = -float("inf")
    for shift in shifts_sec:
        shifted = score_bt + float(shift)
        score = float(np.mean(np.interp(shifted, env_times, onset_env)))
        if score > best_score:
            best_score = score
            best_shift = float(shift)

    if abs(best_shift) < 1e-6:
        return beat_grid
    raw_beats = beat_grid.beat_times + best_shift
    kept_beats = raw_beats[raw_beats >= 0.0]
    if kept_beats.size < 2:
        # Preserve strict monotonicity if fewer than 2 non-negative beats remain.
        shift_floor = float(np.min(beat_grid.beat_times))
        applied_shift = max(best_shift, -shift_floor)
        shifted_beats = beat_grid.beat_times + applied_shift
        shifted_downbeats = (
            beat_grid.downbeat_times + applied_shift
            if beat_grid.downbeat_times.size > 0
            else beat_grid.downbeat_times
        )
    else:
        shifted_beats = kept_beats
        shifted_downbeats = (
            (beat_grid.downbeat_times + best_shift)[
                beat_grid.downbeat_times + best_shift >= 0.0
            ]
            if beat_grid.downbeat_times.size > 0
            else beat_grid.downbeat_times
        )
    return BeatGrid(
        beat_times=shifted_beats,
        downbeat_times=shifted_downbeats,
        bpm=beat_grid.bpm,
        tracker=beat_grid.tracker,
    )


def regularize_beat_grid(
    beat_grid: BeatGrid,
    *,
    audio_duration_sec: float | None = None,
    inlier_tol_sec: float = 0.035,
    min_inlier_fraction: float = 0.85,
) -> BeatGrid:
    """Fit a drift-free constant-BPM integer beat lattice when the track is steady.

    Neural beat trackers occasionally drop a beat during silent breakdowns or
    emit spurious extra beats during fills. For steady-tempo songs (electronic,
    pop, and most rhythm-game tracks), fitting an integer-indexed lattice
    ``t(k) = t0 + k * ibi`` recovers tempo to ~0.01 BPM, bridges dropped beats,
    removes spurious beats, and aligns downbeats onto exact 4-beat bar lines.
    Songs with genuine tempo changes (fewer than ``min_inlier_fraction`` of
    beats within ``inlier_tol_sec``) are returned unchanged.

    Args:
        beat_grid: Input :class:`BeatGrid`.
        audio_duration_sec: Optional full audio duration to extend the lattice
            through quiet intros/outros.
        inlier_tol_sec: Maximum residual (seconds) for a beat to count as an
            inlier on the constant-BPM lattice.
        min_inlier_fraction: Minimum fraction of beats that must be inliers to
            replace the raw grid with the regularized constant-BPM lattice.

    Returns:
        Regularized :class:`BeatGrid` if the song fits a constant tempo, else
        the original ``beat_grid`` unchanged.
    """
    bt = np.asarray(beat_grid.beat_times, dtype=np.float64).reshape(-1)
    if bt.size < 8:
        return beat_grid

    ibis = np.diff(bt)
    med_ibi = float(np.median(ibis))
    if med_ibi <= 1e-4:
        return beat_grid
    inlier_ibis = ibis[np.abs(ibis - med_ibi) <= 0.15 * med_ibi]
    ibi = float(np.mean(inlier_ibis)) if inlier_ibis.size > 0 else med_ibi
    anchor_t = float(bt[bt.size // 2])

    for _ in range(3):
        k = np.rint((bt - anchor_t) / ibi)
        resid = bt - (anchor_t + k * ibi)
        mask = np.abs(resid) <= max(inlier_tol_sec, 0.12 * ibi)
        if int(np.sum(mask)) < 4 or np.unique(k[mask]).size < 2:
            return beat_grid
        slope, intercept = np.polyfit(k[mask], bt[mask], 1)
        if not (slope > 1e-4 and np.isfinite(slope) and np.isfinite(intercept)):
            return beat_grid
        ibi = float(slope)
        anchor_t = float(intercept)

    resid_all = np.abs(bt - (anchor_t + np.rint((bt - anchor_t) / ibi) * ibi))
    inlier_fraction = float(np.mean(resid_all <= float(inlier_tol_sec)))
    if inlier_fraction < float(min_inlier_fraction):
        return beat_grid

    end_time = (
        max(float(bt[-1]), float(audio_duration_sec) - 0.5 * ibi)
        if audio_duration_sec is not None and audio_duration_sec > 0.0
        else float(bt[-1])
    )
    k_start = int(np.ceil(-anchor_t / ibi))
    k_end = int(np.floor((end_time - anchor_t) / ibi))
    if k_end - k_start < 1:
        return beat_grid
    ks = np.arange(k_start, k_end + 1, dtype=np.int64)
    reg_beats = anchor_t + ks.astype(np.float64) * ibi

    # Align downbeats to the dominant 4-beat bar phase modulo 4.
    db = np.asarray(beat_grid.downbeat_times, dtype=np.float64).reshape(-1)
    if db.size > 0:
        db_k = np.rint((db - anchor_t) / ibi).astype(np.int64)
        db_resid = np.abs(db - (anchor_t + db_k * ibi))
        valid_db_k = db_k[db_resid <= float(inlier_tol_sec)]
        if valid_db_k.size > 0:
            counts = np.bincount(np.mod(valid_db_k, 4), minlength=4)
            bar_phase = int(np.argmax(counts))
            reg_downbeats = reg_beats[np.mod(ks, 4) == bar_phase]
        else:
            reg_downbeats = reg_beats[::4]
    else:
        reg_downbeats = np.zeros(0, dtype=np.float64)

    return BeatGrid(
        beat_times=reg_beats,
        downbeat_times=reg_downbeats,
        bpm=60.0 / max(ibi, 1e-6),
        tracker=beat_grid.tracker,
    )


def estimate_beats(
    audio_path: str,
    *,
    tracker: str = TRACKER_AUTO,
    device: str | torch.device = "auto",
    snap_to_onsets: bool = False,
    regularize_tempo: bool = True,
) -> BeatGrid:
    """Estimate a :class:`BeatGrid` from raw audio using the selected backend.

    In ``"auto"`` mode, uses the built-in ``beat_this`` model and falls back
    gracefully to ``librosa`` with a warning if the Beat This! checkpoint
    cannot be loaded (for example, in an offline environment without cached
    weights).

    Args:
        audio_path: Path to the input audio file.
        tracker: Backend selector: ``"auto"`` (uses the built-in ``beat_this``
            tracker with offline ``librosa`` fallback), ``"beat_this"``, or
            ``"librosa"``.
        device: Compute device forwarded to ``beat_this``.
        snap_to_onsets: If True, apply :func:`snap_beats_to_onsets` to correct
            constant beat-annotation phase bias.
        regularize_tempo: If True (default), apply :func:`regularize_beat_grid`
            to recover exact constant BPM and bridge dropped beats on steady
            songs while leaving variable-tempo songs untouched.

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
        try:
            grid = estimate_beats_beat_this(audio_path, device=device)
        except beat_this_model.BeatThisCheckpointError as exc:
            warnings.warn(
                f"Falling back to librosa beat tracker because Beat This! "
                f"checkpoint could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            grid = estimate_beats_librosa(audio_path)
    else:
        grid = estimate_beats_librosa(audio_path)

    if regularize_tempo:
        duration_sec: float | None = None
        if pathlib.Path(audio_path).is_file():
            try:
                duration_sec = float(librosa.get_duration(path=audio_path))
            except Exception:  # noqa: BLE001
                duration_sec = None
        grid = regularize_beat_grid(grid, audio_duration_sec=duration_sec)

    if snap_to_onsets:
        grid = snap_beats_to_onsets(grid, audio_path)
    return grid


def is_itgpt_checkpoint(checkpoint_path: str | pathlib.Path) -> bool:
    """Return True if ``checkpoint_path`` is a saved PyTorch ITGPT checkpoint.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint file.

    Returns:
        True if the file exists and holds a serialized ITGPT placement config.
    """
    path = pathlib.Path(checkpoint_path)
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(payload, dict):
        return False
    model_cfg = payload.get("model_config")
    return isinstance(model_cfg, dict) and "n_enc_layers" in model_cfg


def load_onset_model_auto(
    checkpoint_path: str | pathlib.Path,
    *,
    device: str | torch.device | None = None,
) -> Any:
    """Load either an ITGPT placement checkpoint or a legacy dense onset model.

    Args:
        checkpoint_path: Path to a ``.pt`` onset model checkpoint.
        device: Optional target device.

    Returns:
        Loaded PyTorch onset model ready for inference.
    """
    target = runtime.resolve_device(device)
    if is_itgpt_checkpoint(checkpoint_path):
        return itgpt_trainers.load_placement_model(str(checkpoint_path), device=target)
    return trainers.load_onset_model(str(checkpoint_path), device=target)


def decode_itgpt_slots_to_times(
    probs: np.ndarray,
    beat_grid: BeatGrid,
    *,
    threshold: float = 0.5,
    use_slot_centers: bool = True,
) -> np.ndarray:
    """Decode ``(n_beats, n_slots)`` ITGPT probabilities to sorted onset times.

    Args:
        probs: Slot occupancy probabilities of shape ``(n_beats, n_slots)``.
        beat_grid: :class:`BeatGrid` corresponding to ``probs``.
        threshold: Probability cutoff for occupied slots.
        use_slot_centers: If True (default), place each occupied slot at its
            cell center ``(slot + 0.5) / n_slots`` (matching
            :func:`stepcovnet.ddcl.slots.occupied_center_times`). If False,
            place notes at exact StepMania grid edges ``slot / n_slots`` so
            quarter/8th/16th/24th notes quantize cleanly without half-slot
            rounding jitter.

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
    slot_offset = 0.5 if use_slot_centers else 0.0
    times: list[float] = []
    for beat_idx, slot_idx in np.argwhere(grid >= float(threshold)):
        t0 = float(bt_full[int(beat_idx)])
        t1 = float(bt_full[int(beat_idx) + 1])
        frac = (float(slot_idx) + slot_offset) / float(n_slots)
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


@dataclasses.dataclass(frozen=True)
class DifficultyPreset:
    """Target StepMania difficulty metadata and step density for pack generation.

    Attributes:
        name: Standard StepMania difficulty name (e.g. ``"Beginner"`` .. ``"Challenge"``).
        meter: StepMania numeric meter written into ``#NOTES``.
        target_density: Target average steps per beat for density-conditioned
            ITGPT models.
    """

    name: str
    meter: int
    target_density: float

    def __post_init__(self) -> None:
        """Validate difficulty name, meter, and positive target density.

        Raises:
            ValueError: If ``name`` is empty, ``meter < 1``, or
                ``target_density <= 0``.
        """
        if not self.name.strip():
            raise ValueError("DifficultyPreset.name must be non-empty")
        if int(self.meter) < 1:
            raise ValueError(f"DifficultyPreset.meter must be >= 1, got {self.meter}")
        if not (float(self.target_density) > 0.0 and np.isfinite(self.target_density)):
            raise ValueError(
                f"DifficultyPreset.target_density must be positive finite, got {self.target_density}"
            )


DEFAULT_DIFFICULTY_PACK: tuple[DifficultyPreset, ...] = (
    DifficultyPreset(name="Beginner", meter=3, target_density=0.4),
    DifficultyPreset(name="Easy", meter=5, target_density=0.8),
    DifficultyPreset(name="Medium", meter=7, target_density=1.3),
    DifficultyPreset(name="Hard", meter=9, target_density=2.0),
    DifficultyPreset(name="Challenge", meter=11, target_density=3.0),
)


def _predict_itgpt_onsets_for_mert(
    song_key: str,
    beat_mert: np.ndarray,
    grid: BeatGrid,
    itgpt_model: Any,
    *,
    difficulty: str,
    target_density: float,
    meter: int,
    threshold: float,
    max_beats: int | None,
    device: torch.device,
    fallback_top_k: bool = False,
) -> np.ndarray:
    """Run ITGPT slot prediction and timestamp decoding on precomputed beat MERT."""
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
        song_key=song_key,
        difficulty=difficulty,
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
        device=device,
    )
    onset_times = decode_itgpt_slots_to_times(
        probs, grid, threshold=threshold, use_slot_centers=False
    )
    if onset_times.size == 0 and fallback_top_k and probs.size > 0:
        flat_probs = probs.reshape(-1)
        k = min(flat_probs.size, max(1, target_n_steps))
        top_idx = np.argpartition(flat_probs, -k)[-k:]
        top_mask = np.zeros_like(flat_probs, dtype=np.float32)
        top_mask[top_idx] = 1.0
        onset_times = decode_itgpt_slots_to_times(
            top_mask.reshape(probs.shape),
            grid,
            threshold=0.5,
            use_slot_centers=False,
        )
    return onset_times


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

    onset_times = _predict_itgpt_onsets_for_mert(
        pathlib.Path(audio_path).stem,
        beat_mert,
        grid,
        itgpt_model,
        difficulty="generated",
        target_density=target_density,
        meter=meter,
        threshold=threshold,
        max_beats=max_beats,
        device=target_device,
    )
    return onset_times, grid


def predict_itgpt_pack_from_audio(
    audio_path: str,
    itgpt_model: Any,
    *,
    difficulties: collections.abc.Sequence[DifficultyPreset] = DEFAULT_DIFFICULTY_PACK,
    beat_grid: BeatGrid | None = None,
    beat_tracker: str = TRACKER_AUTO,
    snap_beats: bool = False,
    threshold: float = 0.5,
    max_beats: int | None = None,
    mert_layers: tuple[int, ...] = DEFAULT_MERT_LAYERS,
    mert_cache_suffix: str = DEFAULT_MERT_CACHE_SUFFIX,
    device: str | torch.device | None = None,
) -> tuple[dict[str, np.ndarray], BeatGrid]:
    """Run Beat This! and MERT once and decode onsets for a full difficulty pack.

    Args:
        audio_path: Path to the raw audio file.
        itgpt_model: Trained PyTorch :class:`ItgptPlacementModel`.
        difficulties: Sequence of :class:`DifficultyPreset` definitions
            (defaults to 5-difficulty Beginner–Challenge pack).
        beat_grid: Optional precomputed :class:`BeatGrid` (estimated when None).
        beat_tracker: Beat tracker backend (``"auto"``, ``"beat_this"``, ``"librosa"``).
        snap_beats: Apply song-level onset-strength phase alignment to beats.
        threshold: Slot probability cutoff (default ``0.5``).
        max_beats: Maximum beat chunk length (defaults to model's ``max_beats``).
        mert_layers: MERT hidden layers to average when extracting on the fly.
        mert_cache_suffix: Sibling MERT cache suffix checked before extraction.
        device: Optional compute device override.

    Returns:
        Tuple ``(onsets_by_difficulty, beat_grid)`` where keys follow the
        ordering of ``difficulties``.

    Raises:
        ValueError: If ``difficulties`` is empty.
    """
    if not difficulties:
        raise ValueError("difficulties must contain at least one DifficultyPreset")

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

    song_key = pathlib.Path(audio_path).stem
    onsets_by_difficulty: dict[str, np.ndarray] = {}
    for preset in difficulties:
        onsets_by_difficulty[preset.name] = _predict_itgpt_onsets_for_mert(
            song_key,
            beat_mert,
            grid,
            itgpt_model,
            difficulty=preset.name,
            target_density=preset.target_density,
            meter=preset.meter,
            threshold=threshold,
            max_beats=max_beats,
            device=target_device,
            fallback_top_k=True,
        )
    return onsets_by_difficulty, grid
