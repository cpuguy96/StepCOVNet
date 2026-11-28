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

from stepcovnet import (
    beat_this_model,
    constants,
    pretrained,
    runtime,
    ssl_features,
    trainers,
)
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import features as ddcl_features
from stepcovnet.itgpt import constants as itgpt_constants
from stepcovnet.itgpt import evaluation as itgpt_eval
from stepcovnet.itgpt import trainers as itgpt_trainers

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

    def sm_bpm_segments(
        self,
        *,
        anchor_time_sec: float | None = None,
        shift_beats: float = 0.0,
        max_drift_sec: float = 0.025,
    ) -> tuple[list[tuple[float, float]], float]:
        """Compute piecewise-constant StepMania ``#BPMS`` segments and ``#OFFSET``.

        Constant-tempo grids (all inter-beat intervals within 5 ms) return a
        single ``[(0.0, self.bpm)]`` segment so steady songs are unaffected.
        Variable-tempo grids are merged greedily into smooth constant-BPM
        plateaus whose cumulative timing error stays within ``max_drift_sec``
        without noisy beat-by-beat scroll-speed jitter.

        Args:
            anchor_time_sec: Beat-0 anchor timestamp before ``shift_beats``
                (defaults to :attr:`anchor_time_sec`).
            shift_beats: Non-negative beat shift added to accommodate pickup
                notes before ``anchor_time_sec`` (typically ``4 * shift_measures``).
            max_drift_sec: Maximum timing deviation (seconds) allowed inside
                each constant-BPM plateau.

        Returns:
            Tuple ``(bpm_segments, sm_offset_sec)`` where ``bpm_segments`` is a
            list of ``(beat, bpm)`` pairs starting at beat ``0.0`` and
            ``sm_offset_sec`` is the matching StepMania ``#OFFSET``.
        """
        bt = np.asarray(self.beat_times, dtype=np.float64).reshape(-1)
        anchor = (
            self.anchor_time_sec if anchor_time_sec is None else float(anchor_time_sec)
        )
        ibis = np.diff(bt)
        if ibis.size == 0 or float(np.max(ibis) - np.min(ibis)) <= 0.005:
            bpm_f = max(float(self.bpm), 1.0)
            t_beat0 = anchor - float(shift_beats) * (60.0 / bpm_f)
            return [(0.0, bpm_f)], -t_beat0

        # Greedy piecewise-linear segmentation over beat_times.
        raw_segs: list[tuple[int, int, float]] = []
        n = int(bt.size)
        i = 0
        while i < n - 1:
            best_j = i + 1
            for j in range(i + 2, n):
                seg_ibi = (bt[j] - bt[i]) / float(j - i)
                ks = np.arange(j - i + 1, dtype=np.float64)
                pred = bt[i] + ks * seg_ibi
                if float(np.max(np.abs(bt[i : j + 1] - pred))) <= float(max_drift_sec):
                    best_j = j
                else:
                    break
            seg_ibi = max((bt[best_j] - bt[i]) / float(best_j - i), 1e-4)
            raw_segs.append((i, best_j, 60.0 / seg_ibi))
            i = best_j

        anchor_idx = float(
            self.times_to_beats(
                np.asarray([anchor], dtype=np.float64),
                anchor_time_sec=float(bt[0]),
            )[0]
        )
        idx_0 = anchor_idx - float(shift_beats)

        # Evaluate audio time and active BPM at continuous beat index idx_0.
        if idx_0 <= 0.0:
            first_bpm = raw_segs[0][2]
            t_beat0 = float(bt[0]) + idx_0 * (60.0 / first_bpm)
            active_bpm = first_bpm
        elif idx_0 >= float(n - 1):
            last_bpm = raw_segs[-1][2]
            t_beat0 = float(bt[-1]) + (idx_0 - float(n - 1)) * (60.0 / last_bpm)
            active_bpm = last_bpm
        else:
            active_bpm = raw_segs[0][2]
            t_beat0 = float(bt[0])
            for s_i, s_j, s_bpm in raw_segs:
                if float(s_i) <= idx_0 <= float(s_j):
                    active_bpm = s_bpm
                    t_beat0 = float(bt[s_i]) + (idx_0 - float(s_i)) * (60.0 / s_bpm)
                    break

        sm_segments: list[tuple[float, float]] = [(0.0, float(active_bpm))]
        for s_i, _s_j, s_bpm in raw_segs:
            if float(s_i) > idx_0 + 1e-6:
                sm_beat = float(s_i) - idx_0
                if abs(float(s_bpm) - sm_segments[-1][1]) >= 0.05:
                    sm_segments.append((sm_beat, float(s_bpm)))

        return sm_segments, -t_beat0


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
    """Check whether Beat This! inference is available.

    Returns:
        Always ``True``, since Beat This! is built into StepCOVNet with no extra
        packages.
    """
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


def _fold_ibi_to_bpm_range(
    ibi: float,
    *,
    min_bpm: float = 110.0,
    max_bpm: float = 240.0,
) -> float:
    """Fold an inter-beat interval by powers of 2 into ``[min_bpm, max_bpm]``.

    Args:
        ibi: Raw inter-beat interval in seconds.
        min_bpm: Lower tempo bound in BPM.
        max_bpm: Upper tempo bound in BPM.

    Returns:
        Octave-folded inter-beat interval in seconds.
    """
    x = float(ibi)
    while x > 1e-6 and 60.0 / x > float(max_bpm):
        x *= 2.0
    while x > 1e-6 and 60.0 / x < float(min_bpm):
        x *= 0.5
    return x


def _octave_folded_ibi_dist(ibi_a: float, ibi_b: float) -> float:
    """Return minimum relative distance between ``ibi_a`` and power-of-2 multiples of ``ibi_b``.

    Args:
        ibi_a: First inter-beat interval in seconds.
        ibi_b: Reference inter-beat interval in seconds.

    Returns:
        Non-negative relative error to the nearest ``{0.25, 0.5, 1, 2, 4}`` multiple.
    """
    if ibi_b <= 1e-6:
        return 1.0
    ratio = float(ibi_a) / float(ibi_b)
    return min(abs(ratio - m) / m for m in (0.25, 0.5, 1.0, 2.0, 4.0))


def _segment_beats_by_tempo(
    bt: np.ndarray,
    *,
    rel_tol: float = 0.06,
    min_section_beats: int = 16,
    octave_folded: bool = True,
) -> list[tuple[int, int]]:
    """Partition beat indices into contiguous constant-tempo sections.

    Smooths inter-beat intervals with a median filter so isolated 1–2 beat
    tracker glitches do not trigger spurious splits, groups adjacent intervals
    within ``rel_tol`` of the section median, merges adjacent runs with
    matching tempo (modulo power-of-2 octaves when ``octave_folded`` is True),
    and absorbs transient runs shorter than ``min_section_beats`` into the
    closer neighboring tempo.

    Args:
        bt: Strictly increasing 1D float64 beat timestamps in seconds.
        rel_tol: Fractional IBI change threshold to start a new tempo section.
        min_section_beats: Minimum beat count for a standalone tempo section.
        octave_folded: When True (default), fold ``0.5× / 2×`` octave flickers
            before smoothing and compare section tempos modulo powers of 2 so
            tracker octave switches inside one medley section do not split it.

    Returns:
        List of non-overlapping ``(start_idx, end_idx)`` half-open index slices
        covering ``[0, bt.size)``.
    """
    n = int(bt.size)
    if n < max(4, int(min_section_beats)):
        return [(0, n)]
    raw_ibis = np.diff(bt)
    ibis = (
        np.asarray(
            [_fold_ibi_to_bpm_range(float(x)) for x in raw_ibis], dtype=np.float64
        )
        if octave_folded
        else raw_ibis
    )
    radius = 4 if octave_folded else 3
    smooth = np.empty_like(ibis)
    for i in range(ibis.size):
        lo = max(0, i - radius)
        hi = min(ibis.size, i + radius + 1)
        smooth[i] = float(np.median(ibis[lo:hi]))

    def _dist(a: float, b: float) -> float:
        if octave_folded:
            return _octave_folded_ibi_dist(a, b)
        return abs(a - b) / max(b, 1e-6)

    runs: list[list[int]] = []
    start = 0
    ref = float(smooth[0])
    for i in range(1, smooth.size):
        if _dist(float(smooth[i]), ref) > float(rel_tol):
            runs.append([start, i])
            start = i
            ref = float(smooth[i])
        elif (i - start) % 4 == 0:
            ref = float(np.median(smooth[start : i + 1]))
    runs.append([start, int(smooth.size)])

    changed = True
    while changed and len(runs) > 1:
        changed = False
        if octave_folded:
            for k in range(len(runs) - 1):
                m1 = float(np.median(ibis[runs[k][0] : runs[k][1]]))
                m2 = float(np.median(ibis[runs[k + 1][0] : runs[k + 1][1]]))
                if _dist(m1, m2) <= float(rel_tol):
                    runs[k][1] = runs[k + 1][1]
                    runs.pop(k + 1)
                    changed = True
                    break
            if changed:
                continue
        lengths = [r[1] - r[0] + 1 for r in runs]
        min_idx = int(np.argmin(lengths))
        if lengths[min_idx] < int(min_section_beats):
            r_lo, r_hi = runs[min_idx]
            r_med = float(np.median(ibis[r_lo:r_hi])) if r_hi > r_lo else 0.5
            left_diff = (
                _dist(
                    r_med,
                    float(np.median(ibis[runs[min_idx - 1][0] : runs[min_idx - 1][1]])),
                )
                if min_idx > 0
                else float("inf")
            )
            right_diff = (
                _dist(
                    r_med,
                    float(np.median(ibis[runs[min_idx + 1][0] : runs[min_idx + 1][1]])),
                )
                if min_idx + 1 < len(runs)
                else float("inf")
            )
            if left_diff <= right_diff:
                runs[min_idx - 1][1] = runs[min_idx][1]
            else:
                runs[min_idx + 1][0] = runs[min_idx][0]
            runs.pop(min_idx)
            changed = True

    sections: list[tuple[int, int]] = []
    for idx, (s, _e_incl) in enumerate(runs):
        e_excl = runs[idx + 1][0] if idx + 1 < len(runs) else n
        if e_excl > s:
            sections.append((s, e_excl))
    return sections if sections else [(0, n)]


def _optimal_attack_shift_sec(
    bt: np.ndarray,
    onset_env: np.ndarray,
    env_times: np.ndarray,
    shifts_sec: np.ndarray,
    max_shift_sec: float,
) -> float:
    """Find the scalar shift maximizing mean attack-onset energy on ``bt``.

    Args:
        bt: 1-D array of beat timestamps in seconds.
        onset_env: 1-D attack-onset envelope values.
        env_times: 1-D timestamp grid corresponding to ``onset_env``.
        shifts_sec: Candidate scalar time shifts in seconds to evaluate.
        max_shift_sec: Maximum absolute shift margin used to select interior beats.

    Returns:
        Optimal shift in seconds from ``shifts_sec``.
    """
    env_end = float(env_times[-1])
    interior = bt[(bt >= max_shift_sec) & (bt <= env_end - max_shift_sec)]
    score_bt = interior if interior.size > 0 else bt
    best_shift = 0.0
    best_score = -float("inf")
    for shift in shifts_sec:
        shifted = score_bt + float(shift)
        score = float(np.mean(np.interp(shifted, env_times, onset_env)))
        if score > best_score:
            best_score = score
            best_shift = float(shift)
    return best_shift


def snap_beats_to_onsets(
    beat_grid: BeatGrid,
    audio_path: str,
    *,
    max_shift_ms: float = 35.0,
    step_ms: float = 0.5,
    per_section: bool = True,
) -> BeatGrid:
    """Align beats to audio attacks via global (steady) or per-section phase snap.

    Neural beat trackers trained on MIR beat annotations often sit ~15–30 ms
    behind sharp percussive attack transients expected by rhythm-game charts.
    For steady-tempo tracks, this searches a single scalar shift
    ``delta in [-max_shift_ms, +max_shift_ms]`` maximizing mean zero-lag
    attack-onset strength without altering inter-beat spacing. For
    variable-tempo tracks and medleys (when ``per_section`` is True), each
    piecewise-constant tempo section is snapped independently so phase jumps
    across mid-song BPM switches are corrected cleanly.

    Args:
        beat_grid: Input :class:`BeatGrid`.
        audio_path: Path to the source audio file.
        max_shift_ms: Maximum absolute shift in milliseconds (default ``35.0``).
        step_ms: Grid search resolution in milliseconds (default ``0.5``).
        per_section: When True (default), snap each detected tempo section
            independently on variable-tempo tracks/medleys.

    Returns:
        New :class:`BeatGrid` shifted onto audio attack transients.
    """
    if max_shift_ms <= 0.0 or step_ms <= 0.0:
        return beat_grid
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    onset_env, env_times = _compute_attack_onset_envelope(y, int(sr))
    if onset_env.size == 0 or float(np.max(onset_env)) <= 0.0:
        return beat_grid

    max_shift_sec = float(max_shift_ms) / 1000.0
    shifts_sec = (
        np.arange(-max_shift_ms, max_shift_ms + 0.5 * step_ms, step_ms) / 1000.0
    )
    bt = beat_grid.beat_times
    ibis = np.diff(bt)
    is_variable = ibis.size > 0 and float(np.max(ibis) - np.min(ibis)) > 0.005

    if per_section and is_variable:
        sections = _segment_beats_by_tempo(bt, rel_tol=0.06, min_section_beats=12)
        if len(sections) > 1:
            new_bt = bt.copy()
            new_db_parts: list[np.ndarray] = []
            db = beat_grid.downbeat_times
            any_shift = False
            for s, e_idx in sections:
                sec_bt = bt[s:e_idx]
                best_shift = _optimal_attack_shift_sec(
                    sec_bt, onset_env, env_times, shifts_sec, max_shift_sec
                )
                if abs(best_shift) >= 1e-6:
                    any_shift = True
                new_bt[s:e_idx] = sec_bt + best_shift
                if db.size > 0:
                    t_lo = float(sec_bt[0]) - 1e-4
                    t_hi = (
                        float(bt[e_idx]) - 1e-4
                        if e_idx < bt.size
                        else float(sec_bt[-1]) + 1.0
                    )
                    sec_db = db[(db >= t_lo) & (db < t_hi)] + best_shift
                    sec_db = sec_db[sec_db >= 0.0]
                    if sec_db.size > 0:
                        new_db_parts.append(sec_db)
            if not any_shift:
                return beat_grid
            cleaned_bt = _clean_beat_times(new_bt[new_bt >= 0.0])
            if cleaned_bt.size >= 2:
                cleaned_db = (
                    _clean_beat_times(np.concatenate(new_db_parts))
                    if new_db_parts
                    else np.zeros(0, dtype=np.float64)
                )
                return BeatGrid(
                    beat_times=cleaned_bt,
                    downbeat_times=cleaned_db,
                    bpm=beat_grid.bpm,
                    tracker=beat_grid.tracker,
                )

    best_shift = _optimal_attack_shift_sec(
        bt, onset_env, env_times, shifts_sec, max_shift_sec
    )
    if abs(best_shift) < 1e-6:
        return beat_grid
    raw_beats = beat_grid.beat_times + best_shift
    # Preserve a genuine Beat 0 sitting right at t=0 that shifted slightly
    # negative (within 5 ms), while still dropping stray negative pre-beats.
    raw_beats = np.where((raw_beats < 0.0) & (raw_beats >= -0.005), 0.0, raw_beats)
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
        raw_db = beat_grid.downbeat_times + best_shift
        raw_db = np.where((raw_db < 0.0) & (raw_db >= -0.005), 0.0, raw_db)
        shifted_downbeats = (
            raw_db[raw_db >= 0.0]
            if beat_grid.downbeat_times.size > 0
            else beat_grid.downbeat_times
        )
    return BeatGrid(
        beat_times=shifted_beats,
        downbeat_times=shifted_downbeats,
        bpm=beat_grid.bpm,
        tracker=beat_grid.tracker,
    )


def _harmonic_ibi_fraction(
    ibis: np.ndarray,
    base_ibi: float,
    *,
    rel_tol: float = 0.12,
) -> float:
    """Return fraction of raw IBIs matching a power-of-2 multiple of ``base_ibi``.

    Args:
        ibis: 1D array of raw inter-beat intervals in seconds.
        base_ibi: Candidate reference inter-beat interval in seconds.
        rel_tol: Maximum relative error from ``{0.25, 0.5, 1, 2, 4} * base_ibi``.

    Returns:
        Fraction of intervals in ``[0.0, 1.0]`` matching a power-of-2 harmonic.
    """
    if ibis.size == 0 or base_ibi <= 1e-6:
        return 0.0
    ratios = ibis / float(base_ibi)
    harmonics = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    rel_err = np.min(
        np.abs(ratios[:, None] - harmonics[None, :]) / harmonics[None, :], axis=1
    )
    return float(np.mean(rel_err <= float(rel_tol)))


def _choose_musical_octave_and_phase(
    bt: np.ndarray,
    db: np.ndarray,
    fine_ibi: float,
    fine_anchor: float,
    *,
    max_bpm: float = 250.0,
) -> tuple[float, float]:
    """Fold an ultra-fast fine lattice into ``<= max_bpm`` using beat/downbeat parity.

    Args:
        bt: Raw beat timestamps in seconds.
        db: Raw downbeat timestamps in seconds (may be empty).
        fine_ibi: Fine-lattice inter-beat interval in seconds.
        fine_anchor: Fine-lattice anchor timestamp in seconds.
        max_bpm: Upper bound of the natural musical tempo octave (default ``250.0``).

    Returns:
        Tuple ``(folded_ibi, folded_anchor)`` aligned to the dominant beat parity.
    """
    stride = 1
    while 60.0 / max(fine_ibi * float(stride), 1e-6) > float(max_bpm):
        stride *= 2
    if stride == 1:
        return float(fine_ibi), float(fine_anchor)

    tol = min(0.035, 0.28 * float(fine_ibi))
    scores = np.zeros(stride, dtype=np.float64)
    k_bt = np.rint((bt - fine_anchor) / fine_ibi).astype(np.int64)
    r_bt = np.abs(bt - (fine_anchor + k_bt * fine_ibi))
    valid_bt = k_bt[r_bt <= tol]
    if valid_bt.size > 0:
        for p in range(stride):
            scores[p] += float(np.sum(np.mod(valid_bt, stride) == p))
    if db.size > 0:
        k_db = np.rint((db - fine_anchor) / fine_ibi).astype(np.int64)
        r_db = np.abs(db - (fine_anchor + k_db * fine_ibi))
        valid_db = k_db[r_db <= tol]
        for p in range(stride):
            scores[p] += 4.0 * float(np.sum(np.mod(valid_db, stride) == p))
    best_p = int(np.argmax(scores))
    return float(fine_ibi) * float(stride), float(fine_anchor) + float(best_p) * float(
        fine_ibi
    )


def _fit_octave_aware_global_lattice(
    bt: np.ndarray,
    db: np.ndarray,
    *,
    inlier_tol_sec: float = 0.035,
    min_inlier_fraction: float = 0.85,
    min_harmonic_fraction: float = 0.80,
    max_bpm: float = 250.0,
) -> tuple[float, float] | None:
    """Fit a constant-BPM lattice across ``0.5× / 1× / 2×`` tracker octaves.

    Uses progressive span expansion (``2.5 s -> 8 s -> 24 s -> 72 s -> full``)
    to avoid phase-wrapping on long or fast tracks and enforces a power-of-2
    IBI consistency guard so genuine non-harmonic medleys are never collapsed
    into a single tempo.

    Args:
        bt: Raw beat timestamps in seconds.
        db: Raw downbeat timestamps in seconds (may be empty).
        inlier_tol_sec: Maximum beat residual in seconds.
        min_inlier_fraction: Minimum fraction of raw beats on the sub-lattice.
        min_harmonic_fraction: Minimum fraction of raw IBIs matching a power-of-2
            harmonic of the fitted tempo (default ``0.80`` for whole-song fits).
        max_bpm: Upper bound when folding a fine sub-lattice to musical BPM.

    Returns:
        Tuple ``(final_ibi, final_anchor)`` if an octave-consistent lattice
        passes all guards, else ``None``.
    """
    if bt.size < 8:
        return None
    ibis = np.diff(bt)
    q_ibis = [float(np.percentile(ibis, p)) for p in (25, 40, 50, 60, 75)]
    raw_cands: list[float] = []
    for base in q_ibis:
        if base <= 1e-4:
            continue
        for mult in (1.0, 0.5, 0.25):
            ci = base * mult
            bpm_c = 60.0 / max(ci, 1e-6)
            if 60.0 <= bpm_c <= 520.0:
                raw_cands.append(ci)
    # Deduplicate near-identical seeds, preferring coarsest (largest IBI) first.
    raw_cands.sort(reverse=True)
    seeds: list[float] = []
    for ci in raw_cands:
        if not seeds or abs(ci - seeds[-1]) / max(seeds[-1], 1e-6) > 0.015:
            seeds.append(ci)

    best_fit: tuple[float, float, float, float] | None = None
    for seed_ibi in seeds:
        for q in (0.5, 0.25, 0.75):
            anchor_t = float(bt[int(q * (bt.size - 1))])
            cur_ibi = float(seed_ibi)
            ok = True
            for span_sec in (2.5, 8.0, 24.0, 72.0, 1e9):
                local = bt[np.abs(bt - anchor_t) <= span_sec]
                if local.size < 4:
                    local = bt
                k = np.rint((local - anchor_t) / cur_ibi)
                resid = local - (anchor_t + k * cur_ibi)
                tol = min(float(inlier_tol_sec), 0.26 * cur_ibi)
                mask = np.abs(resid) <= tol
                if int(np.sum(mask)) < 4 or np.unique(k[mask]).size < 2:
                    ok = False
                    break
                slope, intercept = np.polyfit(k[mask], local[mask], 1)
                if not (slope > 1e-4 and np.isfinite(slope) and np.isfinite(intercept)):
                    ok = False
                    break
                cur_ibi = float(slope)
                anchor_t = float(intercept)
            if not ok:
                continue

            tol_eval = min(float(inlier_tol_sec), 0.24 * cur_ibi)
            k_all = np.rint((bt - anchor_t) / cur_ibi)
            resid_all = np.abs(bt - (anchor_t + k_all * cur_ibi))
            inlier_mask = resid_all <= tol_eval
            frac = float(np.mean(inlier_mask))
            if frac < float(min_inlier_fraction):
                continue
            med_err = float(np.median(resid_all[inlier_mask]))
            med_limit = (
                min(0.012, 0.08 * cur_ibi)
                if min_harmonic_fraction >= 0.75
                else min(0.015, 0.10 * cur_ibi)
            )
            if med_err > med_limit:
                continue
            # Do not hallucinate a finer 2x/4x sub-lattice unless the raw tracker
            # actually emitted intervals at that finer spacing (prevents doubling
            # noisy single-octave grids like librosa 124 -> 248 BPM).
            if cur_ibi < 0.75 * float(np.median(ibis)) and (
                float(np.mean(ibis <= 1.35 * cur_ibi)) < 0.10
            ):
                continue

            final_ibi, final_anchor = _choose_musical_octave_and_phase(
                bt, db, cur_ibi, anchor_t, max_bpm=max_bpm
            )
            if _harmonic_ibi_fraction(ibis, final_ibi) < float(min_harmonic_fraction):
                continue
            if (
                best_fit is None
                or frac > best_fit[0] + 0.03
                or (abs(frac - best_fit[0]) <= 0.03 and med_err < best_fit[1] - 0.0005)
            ):
                best_fit = (frac, med_err, final_ibi, final_anchor)

    if best_fit is None:
        return None
    return best_fit[2], best_fit[3]


def regularize_beat_grid(
    beat_grid: BeatGrid,
    *,
    audio_duration_sec: float | None = None,
    inlier_tol_sec: float = 0.035,
    min_inlier_fraction: float = 0.85,
    integer_bpm_tol: float = 0.03,
    piecewise_sections: bool = True,
    octave_aware: bool = True,
) -> BeatGrid:
    """Fit a drift-free constant-BPM lattice (globally or per tempo section).

    Neural beat trackers occasionally drop a beat during silent breakdowns,
    emit spurious extra beats during fills, or flicker between tempo octaves
    (``0.5× / 1× / 2×``). For steady-tempo songs, fitting an integer-indexed
    lattice ``t(k) = t0 + k * ibi`` recovers tempo to ~0.01 BPM, bridges
    dropped beats, removes spurious beats, and aligns downbeats onto exact
    4-beat bar lines. When a single 1× fit fails because the tracker flickered
    between power-of-2 octaves (when ``octave_aware`` is True), an octave-aware
    global lattice unifies the song onto one musical constant-BPM grid. For
    genuine medleys and variable-tempo songs (when ``piecewise_sections`` is
    True), each contiguous constant-tempo section is regularized independently
    (including per-section octave-flicker unification and ``> 250 BPM``
    double-time folding) while leaving unstructured rubato sections untouched.

    Args:
        beat_grid: Input :class:`BeatGrid`.
        audio_duration_sec: Optional full audio duration to extend the lattice
            through quiet intros/outros.
        inlier_tol_sec: Maximum residual (seconds) for a beat to count as an
            inlier on the constant-BPM lattice.
        min_inlier_fraction: Minimum fraction of beats that must be inliers to
            replace the raw grid with the regularized constant-BPM lattice.
        integer_bpm_tol: Maximum distance (BPM) from an integer tempo to test
            an exact integer-BPM lock (``0.0`` disables integer locking).
        piecewise_sections: When True (default), fall back to per-section
            constant-BPM regularization on variable-tempo songs and medleys.
        octave_aware: When True (default), unify power-of-2 octave flickers
            (``0.5× / 1× / 2×``) onto a single musical BPM grid and fold
            ``> 250 BPM`` double-time sections.

    Returns:
        Regularized :class:`BeatGrid` (global or piecewise-constant), or the
        original ``beat_grid`` if no section fits a clean tempo lattice.
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

    global_fit_ok = True
    for _ in range(3):
        k = np.rint((bt - anchor_t) / ibi)
        resid = bt - (anchor_t + k * ibi)
        mask = np.abs(resid) <= max(inlier_tol_sec, 0.12 * ibi)
        if int(np.sum(mask)) < 4 or np.unique(k[mask]).size < 2:
            global_fit_ok = False
            break
        slope, intercept = np.polyfit(k[mask], bt[mask], 1)
        if not (slope > 1e-4 and np.isfinite(slope) and np.isfinite(intercept)):
            global_fit_ok = False
            break
        ibi = float(slope)
        anchor_t = float(intercept)

    k_all = np.rint((bt - anchor_t) / ibi)
    resid_all = np.abs(bt - (anchor_t + k_all * ibi))
    inlier_mask = resid_all <= float(inlier_tol_sec)
    inlier_fraction = float(np.mean(inlier_mask))
    octave_global_used = False

    if not global_fit_ok or inlier_fraction < float(min_inlier_fraction):
        db_all = np.asarray(beat_grid.downbeat_times, dtype=np.float64).reshape(-1)
        oct_fit = (
            _fit_octave_aware_global_lattice(
                bt,
                db_all,
                inlier_tol_sec=inlier_tol_sec,
                min_inlier_fraction=min_inlier_fraction,
            )
            if octave_aware
            else None
        )
        if oct_fit is not None:
            ibi, anchor_t = oct_fit
            k_all = np.rint((bt - anchor_t) / ibi)
            resid_all = np.abs(bt - (anchor_t + k_all * ibi))
            inlier_mask = resid_all <= float(inlier_tol_sec)
            octave_global_used = True
        else:
            if not piecewise_sections:
                return beat_grid
            min_sec = min(24, max(12, int(bt.size) // 16))
            sections = _segment_beats_by_tempo(
                bt,
                rel_tol=0.07 if octave_aware else 0.06,
                min_section_beats=min_sec,
                octave_folded=octave_aware,
            )
            if len(sections) <= 1:
                return beat_grid
            merged_beats_parts: list[np.ndarray] = []
            merged_db_parts: list[np.ndarray] = []
            any_section_regularized = False
            for s, e_idx in sections:
                sec_bt = bt[s:e_idx]
                sec_db = (
                    db_all[(db_all >= sec_bt[0] - 0.05) & (db_all <= sec_bt[-1] + 0.05)]
                    if db_all.size > 0
                    else np.zeros(0, dtype=np.float64)
                )
                if sec_bt.size >= 8:
                    sec_grid = BeatGrid(
                        beat_times=sec_bt,
                        downbeat_times=sec_db,
                        bpm=_robust_bpm_from_beats(sec_bt),
                        tracker=beat_grid.tracker,
                    )
                    sec_min_inlier = min(0.75, float(min_inlier_fraction))
                    sec_reg = regularize_beat_grid(
                        sec_grid,
                        inlier_tol_sec=inlier_tol_sec,
                        min_inlier_fraction=sec_min_inlier,
                        integer_bpm_tol=integer_bpm_tol,
                        piecewise_sections=False,
                        octave_aware=False,
                    )
                    sec_oct_fit: tuple[float, float] | None = None
                    if octave_aware and sec_reg is sec_grid:
                        sec_oct_fit = _fit_octave_aware_global_lattice(
                            sec_bt,
                            sec_db,
                            inlier_tol_sec=inlier_tol_sec,
                            min_inlier_fraction=sec_min_inlier,
                            min_harmonic_fraction=0.65,
                        )
                    if sec_oct_fit is not None:
                        s_ibi, s_anc = sec_oct_fit
                        s_bpm = 60.0 / max(s_ibi, 1e-6)
                        if (
                            integer_bpm_tol > 0.0
                            and round(s_bpm) >= 40
                            and abs(s_bpm - float(round(s_bpm)))
                            <= max(float(integer_bpm_tol), 0.05)
                        ):
                            ibi_i = 60.0 / float(round(s_bpm))
                            k_i = np.rint((sec_bt - s_anc) / ibi_i)
                            anc_i = float(np.median(sec_bt - k_i * ibi_i))
                            if (
                                float(
                                    np.mean(
                                        np.abs(sec_bt - (anc_i + k_i * ibi_i))
                                        <= float(inlier_tol_sec)
                                    )
                                )
                                >= 0.70
                            ):
                                s_ibi, s_anc = ibi_i, anc_i
                        k_lo = int(np.ceil((sec_bt[0] - 0.05 - s_anc) / s_ibi))
                        k_hi = int(np.floor((sec_bt[-1] + 0.05 - s_anc) / s_ibi))
                        if k_hi - k_lo >= 1:
                            ks_s = np.arange(k_lo, k_hi + 1, dtype=np.int64)
                            sb = s_anc + ks_s.astype(np.float64) * s_ibi
                            any_section_regularized = True
                            if sec_db.size > 0:
                                db_ks = np.rint((sec_db - s_anc) / s_ibi).astype(
                                    np.int64
                                )
                                db_rs = np.abs(sec_db - (s_anc + db_ks * s_ibi))
                                v_db = db_ks[db_rs <= float(inlier_tol_sec)]
                                if v_db.size > 0:
                                    bp = int(
                                        np.argmax(
                                            np.bincount(np.mod(v_db, 4), minlength=4)
                                        )
                                    )
                                    sd = sb[np.mod(ks_s, 4) == bp]
                                else:
                                    sd = sb[::4]
                            else:
                                sd = np.zeros(0, dtype=np.float64)
                        else:
                            sb, sd = sec_bt, sec_db
                    else:
                        sb = sec_reg.beat_times
                        sb = sb[(sb >= sec_bt[0] - 0.05) & (sb <= sec_bt[-1] + 0.05)]
                        if sb.size >= 2:
                            if sec_reg is not sec_grid:
                                any_section_regularized = True
                            if octave_aware and sec_reg.bpm > 250.0:
                                sec_ibi = 60.0 / max(float(sec_reg.bpm), 1e-6)
                                folded_ibi, folded_anchor = (
                                    _choose_musical_octave_and_phase(
                                        sec_bt,
                                        sec_db,
                                        sec_ibi,
                                        float(sb[0]),
                                        max_bpm=250.0,
                                    )
                                )
                                k_lo = int(
                                    np.ceil(
                                        (sec_bt[0] - 0.05 - folded_anchor) / folded_ibi
                                    )
                                )
                                k_hi = int(
                                    np.floor(
                                        (sec_bt[-1] + 0.05 - folded_anchor) / folded_ibi
                                    )
                                )
                                if k_hi - k_lo >= 1:
                                    ks_f = np.arange(k_lo, k_hi + 1, dtype=np.int64)
                                    sb = (
                                        folded_anchor
                                        + ks_f.astype(np.float64) * folded_ibi
                                    )
                                    any_section_regularized = True
                                    if sec_db.size > 0:
                                        db_kf = np.rint(
                                            (sec_db - folded_anchor) / folded_ibi
                                        ).astype(np.int64)
                                        db_rf = np.abs(
                                            sec_db
                                            - (folded_anchor + db_kf * folded_ibi)
                                        )
                                        v_db = db_kf[db_rf <= float(inlier_tol_sec)]
                                        if v_db.size > 0:
                                            bp = int(
                                                np.argmax(
                                                    np.bincount(
                                                        np.mod(v_db, 4), minlength=4
                                                    )
                                                )
                                            )
                                            sd = sb[np.mod(ks_f, 4) == bp]
                                        else:
                                            sd = sb[::4]
                                    else:
                                        sd = np.zeros(0, dtype=np.float64)
                                else:
                                    sd = sec_reg.downbeat_times
                            else:
                                sd = sec_reg.downbeat_times
                            sd = (
                                sd[(sd >= sb[0] - 1e-4) & (sd <= sb[-1] + 1e-4)]
                                if sd.size > 0
                                else np.zeros(0, dtype=np.float64)
                            )
                        else:
                            sb, sd = sec_bt, sec_db
                else:
                    sb, sd = sec_bt, sec_db

                if merged_beats_parts and sb.size > 0:
                    min_t = float(merged_beats_parts[-1][-1]) + 0.05
                    sb = sb[sb >= min_t]
                    sd = sd[sd >= min_t]
                if sb.size > 0:
                    merged_beats_parts.append(sb)
                if sd.size > 0:
                    merged_db_parts.append(sd)

            if not any_section_regularized or not merged_beats_parts:
                return beat_grid
            merged_bt = _clean_beat_times(np.concatenate(merged_beats_parts))
            if merged_bt.size < 2:
                return beat_grid
            merged_db = (
                _clean_beat_times(np.concatenate(merged_db_parts))
                if merged_db_parts
                else np.zeros(0, dtype=np.float64)
            )
            return BeatGrid(
                beat_times=merged_bt,
                downbeat_times=merged_db,
                bpm=_robust_bpm_from_beats(merged_bt),
                tracker=beat_grid.tracker,
            )

    # Safe integer-BPM lock: eliminate ~0.01–0.02 BPM estimation drift on
    # steady studio/electronic tracks while preserving genuinely fractional BPMs.
    cont_bpm = 60.0 / max(ibi, 1e-6)
    nearest_int_bpm = round(cont_bpm)
    if (
        integer_bpm_tol > 0.0
        and nearest_int_bpm >= 40
        and abs(cont_bpm - float(nearest_int_bpm)) <= float(integer_bpm_tol)
        and int(np.sum(inlier_mask)) >= 4
    ):
        ibi_int = 60.0 / float(nearest_int_bpm)
        inlier_bt = bt[inlier_mask]
        inlier_k = k_all[inlier_mask]
        anchor_int = float(np.median(inlier_bt - inlier_k * ibi_int))
        resid_int_all = np.abs(
            bt - (anchor_int + np.rint((bt - anchor_int) / ibi_int) * ibi_int)
        )
        signed_int = inlier_bt - (anchor_int + inlier_k * ibi_int)
        q_n = max(1, signed_int.size // 4)
        int_drift = abs(
            float(np.median(signed_int[-q_n:])) - float(np.median(signed_int[:q_n]))
        )
        req_frac = (
            float(np.mean(inlier_mask)) - 0.02
            if octave_global_used
            else float(min_inlier_fraction)
        )
        if (
            float(np.mean(resid_int_all <= float(inlier_tol_sec))) >= req_frac
            and float(np.median(np.abs(signed_int)))
            <= float(np.median(resid_all[inlier_mask])) + 0.005
            and int_drift <= 0.015
        ):
            ibi = ibi_int
            anchor_t = anchor_int

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


SLOTS_8TH: frozenset[int] = frozenset({0, 24})
SLOTS_16TH: frozenset[int] = frozenset(range(0, 48, 12))
SLOTS_16TH_24TH: frozenset[int] = frozenset(
    s for s in range(48) if s % 12 == 0 or s % 8 == 0
)


def decode_itgpt_slots_to_times(
    probs: np.ndarray,
    beat_grid: BeatGrid,
    *,
    threshold: float = 0.5,
    use_slot_centers: bool = True,
    allowed_slots: collections.abc.Collection[int] | None = None,
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
        allowed_slots: Optional subset of valid slot indices (e.g.
            :data:`SLOTS_16TH_24TH`) to suppress noisy non-musical slots that
            would otherwise trigger 192-row StepMania measure blocks.

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
    if allowed_slots is not None:
        allowed_set = {int(s) for s in allowed_slots if 0 <= int(s) < n_slots}
        slot_mask = np.array([s in allowed_set for s in range(n_slots)], dtype=bool)
        grid = np.where(slot_mask[None, :], grid, 0.0)
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
    use_mert_cache: bool = True,
    mert_cache_dir: str | pathlib.Path | None = None,
) -> np.ndarray:
    """Load sibling or audio-hash cached MERT features if present, else extract.

    Args:
        audio_path: Path to the source audio file.
        mert_layers: Tuple of MERT transformer layer indices to extract/average.
        mert_cache_suffix: Filename suffix for sibling ``.npy`` MERT caches.
        device: Device string for MERT extraction when uncached.
        use_mert_cache: Whether to read/write the shared audio-hash MERT cache.
        mert_cache_dir: Optional override directory for the shared MERT cache.

    Returns:
        2-D float32 MERT feature array of shape ``(n_frames, hidden_dim)``.
    """
    sibling = pathlib.Path(audio_path).with_suffix(mert_cache_suffix)
    if sibling.is_file():
        return ddcl_features.load_mert_features(
            audio_path, cache_suffix=mert_cache_suffix
        )
    cache_dir: str | pathlib.Path | None = None
    if use_mert_cache:
        cache_dir = (
            mert_cache_dir
            if mert_cache_dir is not None
            else pretrained.get_default_mert_cache_dir()
        )
    return ssl_features.load_or_extract_mert_features_cached(
        audio_path,
        cache_dir=cache_dir,
        layers=mert_layers,
        device=device,
    )


def estimate_sample_preview(
    audio_path: str,
    *,
    beat_grid: BeatGrid | None = None,
    sample_length_sec: float = 15.0,
) -> tuple[float, float]:
    """Estimate an energetic chorus-like preview window for StepMania ``#SAMPLESTART``.

    Scores sliding ``sample_length_sec`` windows using a blend of RMS loudness
    and onset strength, gently favoring the first major chorus zone (22%–55% of
    the track) when multiple choruses have similar energy, and snaps the start
    time to the nearest downbeat (or beat) when ``beat_grid`` is provided.

    Args:
        audio_path: Path to the source audio file.
        beat_grid: Optional :class:`BeatGrid` for downbeat snapping.
        sample_length_sec: Desired preview duration in seconds (default ``15.0``).

    Returns:
        Tuple ``(sample_start_sec, sample_length_sec)`` rounded to milliseconds.
    """
    req_len = max(1.0, float(sample_length_sec))
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    sr_f = float(sr)
    duration = float(y.size) / sr_f if sr_f > 0.0 else 0.0
    if y.size == 0 or duration <= 0.0:
        return 0.0, round(req_len, 3)

    eff_len = min(req_len, duration)
    if duration <= eff_len + 1.0:
        return 0.0, round(max(1.0, eff_len), 3)

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0].astype(
        np.float64
    )
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(
        np.float64
    )
    n_frames = min(rms.size, onset_env.size)
    if n_frames < 4:
        return 0.0, round(max(1.0, eff_len), 3)

    rms = rms[:n_frames]
    onset_env = onset_env[:n_frames]
    rms_z = (rms - float(np.mean(rms))) / max(float(np.std(rms)), 1e-8)
    onset_z = (onset_env - float(np.mean(onset_env))) / max(
        float(np.std(onset_env)), 1e-8
    )
    energy = 0.7 * rms_z + 0.3 * onset_z

    frame_sec = float(hop_length) / sr_f
    win_frames = max(1, min(n_frames, int(round(eff_len / frame_sec))))
    cumsum = np.concatenate(([0.0], np.cumsum(energy)))
    n_starts = n_frames - win_frames + 1
    if n_starts <= 1:
        return 0.0, round(max(1.0, eff_len), 3)

    win_means = (cumsum[win_frames:] - cumsum[:-win_frames]) / float(win_frames)
    start_times = np.arange(n_starts, dtype=np.float64) * frame_sec

    # Skip intro and final fadeout on full-length tracks so previews jump right
    # into the main section.
    if duration >= eff_len + 10.0:
        min_start = 0.15 * duration
        max_start = max(min_start, duration - eff_len - 0.05 * duration)
    else:
        min_start = 0.0
        max_start = max(0.0, duration - eff_len)

    valid_mask = (start_times >= min_start) & (start_times <= max_start + 1e-6)
    if not bool(np.any(valid_mask)):
        valid_mask = np.ones(n_starts, dtype=bool)

    # Mild bonus for the first major chorus zone (22%–55%) to break ties cleanly
    # against repeated outro choruses without overriding a louder late drop.
    chorus_bonus = np.where(
        (start_times >= 0.22 * duration) & (start_times <= 0.55 * duration),
        0.10,
        0.0,
    )
    scores = np.where(valid_mask, win_means + chorus_bonus, -np.inf)
    best_start = float(start_times[int(np.argmax(scores))])

    if beat_grid is not None:
        anchors = (
            beat_grid.downbeat_times
            if beat_grid.downbeat_times.size > 0
            else beat_grid.beat_times
        )
        if anchors.size > 0:
            feasible = anchors[
                (anchors >= 0.0) & (anchors <= max(0.0, duration - eff_len + 0.5))
            ]
            if feasible.size > 0:
                nearest_idx = int(np.argmin(np.abs(feasible - best_start)))
                nearest_t = float(feasible[nearest_idx])
                if abs(nearest_t - best_start) <= 4.0:
                    best_start = nearest_t

    best_start = float(np.clip(best_start, 0.0, max(0.0, duration - 1.0)))
    return round(best_start, 3), round(max(1.0, eff_len), 3)


@dataclasses.dataclass(frozen=True)
class DifficultyPreset:
    """Target StepMania difficulty metadata and step density for pack generation.

    Supports either fixed steps-per-beat (``target_density``) or BPM-adaptive
    physical foot speed (``target_nps``, converted via ``target_nps * 60 / bpm``
    and clamped to ``[min_density, max_density]``).

    Attributes:
        name: Standard StepMania difficulty name (e.g. ``"Beginner"`` .. ``"Challenge"``).
        meter: StepMania numeric meter written into ``#NOTES``.
        target_density: Fallback/reference average steps per beat for
            density-conditioned ITGPT models.
        target_nps: Optional target physical steps per second (NPS) used to
            adapt steps-per-beat automatically across song tempos.
        min_density: Lower clamp for BPM-adapted steps per beat.
        max_density: Upper clamp for BPM-adapted steps per beat.
        allowed_slots: Optional subset of valid 48-slot beat subdivisions
            (e.g. :data:`SLOTS_8TH` for Beginner, :data:`SLOTS_16TH_24TH` for
            Hard/Challenge).
    """

    name: str
    meter: int
    target_density: float
    target_nps: float | None = None
    min_density: float = 0.15
    max_density: float = 3.5
    allowed_slots: frozenset[int] | None = None

    def __post_init__(self) -> None:
        """Validate difficulty name, meter, and positive target densities.

        Raises:
            ValueError: If ``name`` is empty, ``meter < 1``,
                ``target_density <= 0``, ``target_nps <= 0``, or density clamps
                are invalid.
        """
        if not self.name.strip():
            raise ValueError("DifficultyPreset.name must be non-empty")
        if int(self.meter) < 1:
            raise ValueError(f"DifficultyPreset.meter must be >= 1, got {self.meter}")
        if not (float(self.target_density) > 0.0 and np.isfinite(self.target_density)):
            raise ValueError(
                f"DifficultyPreset.target_density must be positive finite, got {self.target_density}"
            )
        if self.target_nps is not None and not (
            float(self.target_nps) > 0.0 and np.isfinite(self.target_nps)
        ):
            raise ValueError(
                f"DifficultyPreset.target_nps must be positive finite, got {self.target_nps}"
            )
        if not (
            float(self.min_density) > 0.0
            and float(self.max_density) >= float(self.min_density)
        ):
            raise ValueError(
                "DifficultyPreset density bounds must satisfy 0 < min_density <= max_density"
            )

    def resolve_target_density(self, bpm: float) -> float:
        """Resolve effective steps-per-beat target for a given song ``bpm``.

        Args:
            bpm: Song tempo in beats per minute.

        Returns:
            Effective target average steps per beat.
        """
        if self.target_nps is not None and float(bpm) > 0.0 and np.isfinite(bpm):
            raw_spb = float(self.target_nps) * 60.0 / float(bpm)
            return float(np.clip(raw_spb, self.min_density, self.max_density))
        return float(self.target_density)


DEFAULT_DIFFICULTY_PACK: tuple[DifficultyPreset, ...] = (
    DifficultyPreset(
        name="Beginner",
        meter=2,
        target_density=0.30,
        target_nps=0.65,
        min_density=0.18,
        max_density=0.55,
        allowed_slots=SLOTS_8TH,
    ),
    DifficultyPreset(
        name="Easy",
        meter=4,
        target_density=0.67,
        target_nps=1.45,
        min_density=0.35,
        max_density=1.05,
        allowed_slots=SLOTS_16TH,
    ),
    DifficultyPreset(
        name="Medium",
        meter=6,
        target_density=1.08,
        target_nps=2.35,
        min_density=0.65,
        max_density=1.70,
        allowed_slots=SLOTS_16TH,
    ),
    DifficultyPreset(
        name="Hard",
        meter=9,
        target_density=1.66,
        target_nps=3.60,
        min_density=1.00,
        max_density=2.50,
        allowed_slots=SLOTS_16TH_24TH,
    ),
    DifficultyPreset(
        name="Challenge",
        meter=11,
        target_density=2.26,
        target_nps=4.90,
        min_density=1.45,
        max_density=3.30,
        allowed_slots=SLOTS_16TH_24TH,
    ),
)


def _align_inputs_to_downbeat_phase(
    beat_mert: np.ndarray,
    grid: BeatGrid,
) -> tuple[np.ndarray, BeatGrid, float | None]:
    """Align beat-indexed MERT and grid so index ``0`` sits on a 4-beat downbeat.

    ITGPT chunks beats into 4-beat bars (``level2``) and 16-beat phrases
    (``level3``) starting strictly at index ``0``, matching StepMania charts
    whose Measure 1 Beat 0 is always a downbeat. When raw-audio beat tracking
    prepends ``1..3`` partial-measure lead-in beats before the first downbeat,
    trimming those leading beats aligns every ``level2``/``level3`` chunk to
    true bar boundaries; when the first detected downbeat lies deeper in the
    track (``db0_idx >= 4``), left-padding ``1..3`` silent beats aligns bar
    phase without discarding early audio.

    Args:
        beat_mert: Beat-aligned MERT array of shape ``(n_beats, n_frames, D)``.
        grid: Corresponding :class:`BeatGrid`.

    Returns:
        Tuple ``(aligned_mert, aligned_grid, min_time_sec)`` where
        ``min_time_sec`` filters out any decoded onset inside synthetic
        left-padding (or ``None`` if no padding was added).
    """
    n_beats = int(beat_mert.shape[0])
    if grid.downbeat_times.size == 0 or n_beats < 8 or grid.n_beats < n_beats:
        return beat_mert, grid, None
    bt = grid.beat_times[:n_beats]
    db0_idx = int(np.argmin(np.abs(bt - float(grid.downbeat_times[0]))))
    rem = db0_idx % 4
    if rem == 0:
        return beat_mert, grid, None
    if 1 <= db0_idx <= 3 and n_beats - db0_idx >= 4:
        trimmed_grid = BeatGrid(
            beat_times=grid.beat_times[db0_idx:],
            downbeat_times=grid.downbeat_times,
            bpm=grid.bpm,
            tracker=grid.tracker,
        )
        return beat_mert[db0_idx:], trimmed_grid, None
    pad = 4 - rem
    ibi0 = max(float(grid.beat_times[1] - grid.beat_times[0]), 1e-3)
    pad_bt = float(grid.beat_times[0]) - np.arange(pad, 0, -1, dtype=np.float64) * ibi0
    pad_grid = BeatGrid(
        beat_times=np.concatenate([pad_bt, grid.beat_times]),
        downbeat_times=grid.downbeat_times,
        bpm=grid.bpm,
        tracker=grid.tracker,
    )
    pad_mert = np.concatenate(
        [
            np.zeros(
                (pad, beat_mert.shape[1], beat_mert.shape[2]), dtype=beat_mert.dtype
            ),
            beat_mert,
        ],
        axis=0,
    )
    return pad_mert, pad_grid, float(grid.beat_times[0]) - 1e-6


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
    allowed_slots: collections.abc.Collection[int] | None = SLOTS_16TH_24TH,
    align_downbeat: bool = True,
) -> np.ndarray:
    """Run ITGPT slot prediction and timestamp decoding on precomputed beat MERT.

    Args:
        song_key: Identifier for the synthetic chart metadata.
        beat_mert: Beat-aligned MERT features of shape ``(n_beats, frames, dim)``.
        grid: Estimated or regularized beat grid for the song.
        itgpt_model: Loaded ITGPT placement model.
        difficulty: Chart difficulty label (for metadata only).
        target_density: Desired average steps per beat for density conditioning.
        meter: StepMania integer difficulty rating for metadata conditioning.
        threshold: Sigmoid probability threshold for slot activation.
        max_beats: Optional cap on beats passed to the model.
        device: Torch device for inference.
        fallback_top_k: If True and threshold yields zero onsets, fall back to top-k.
        allowed_slots: Optional subset of 48-slot indices permitted during decoding.
        align_downbeat: If True, align beat 0 to ``grid.downbeat_times[0]`` before
            ITGPT bar/phrase chunking.

    Returns:
        Sorted 1-D float64 array of predicted onset timestamps in seconds.
    """
    orig_n_beats = int(beat_mert.shape[0])
    min_time_sec: float | None = None
    if align_downbeat:
        beat_mert, grid, min_time_sec = _align_inputs_to_downbeat_phase(beat_mert, grid)
    n_beats = int(beat_mert.shape[0])
    stream = np.zeros((n_beats, 2), dtype=np.float32)
    stream[:, 0] = float(meter)
    stream[:, 1] = grid.per_beat_bpm()[:n_beats]

    # Populate dummy_slots with the exact target total step count so
    # density-conditioned ITGPT models receive mean(steps_per_beat) == target_density.
    dummy_slots = np.zeros((n_beats, itgpt_constants.N_SLOTS), dtype=np.float32)
    target_n_steps = max(
        1, int(round(float(target_density) * float(min(orig_n_beats, n_beats))))
    )
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
        probs,
        grid,
        threshold=threshold,
        use_slot_centers=False,
        allowed_slots=allowed_slots,
    )
    if min_time_sec is not None and onset_times.size > 0:
        onset_times = onset_times[onset_times >= min_time_sec]
    if onset_times.size == 0 and fallback_top_k and probs.size > 0:
        masked_probs = np.asarray(probs, dtype=np.float64)
        if allowed_slots is not None and masked_probs.ndim == 2:
            n_slots = int(masked_probs.shape[1])
            allowed_set = {int(s) for s in allowed_slots if 0 <= int(s) < n_slots}
            slot_mask = np.array([s in allowed_set for s in range(n_slots)], dtype=bool)
            masked_probs = np.where(slot_mask[None, :], masked_probs, -1.0)
        flat_probs = masked_probs.reshape(-1)
        valid_count = int(np.sum(flat_probs >= 0.0))
        k = min(max(1, valid_count), max(1, target_n_steps))
        top_idx = np.argpartition(flat_probs, -k)[-k:]
        top_mask = np.zeros_like(flat_probs, dtype=np.float32)
        top_mask[top_idx] = 1.0
        onset_times = decode_itgpt_slots_to_times(
            top_mask.reshape(probs.shape),
            grid,
            threshold=0.5,
            use_slot_centers=False,
            allowed_slots=allowed_slots,
        )
        if min_time_sec is not None and onset_times.size > 0:
            onset_times = onset_times[onset_times >= min_time_sec]
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
    use_mert_cache: bool = True,
    device: str | torch.device | None = None,
    allowed_slots: collections.abc.Collection[int] | None = SLOTS_16TH_24TH,
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
        use_mert_cache: Cache extracted MERT features by audio SHA-256 hash.
        device: Optional compute device override.
        allowed_slots: Optional valid 48-slot beat subdivisions (defaults to
            :data:`SLOTS_16TH_24TH` to avoid stray 192-row StepMania measures).

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
        use_mert_cache=use_mert_cache,
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
        allowed_slots=allowed_slots,
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
    use_mert_cache: bool = True,
    device: str | torch.device | None = None,
) -> tuple[dict[str, np.ndarray], BeatGrid]:
    """Run Beat This! and MERT once and decode onsets for a full difficulty pack.

    Args:
        audio_path: Path to the raw audio file.
        itgpt_model: Trained PyTorch :class:`ItgptPlacementModel`.
        difficulties: Sequence of :class:`DifficultyPreset` definitions
            (defaults to BPM-adaptive 5-difficulty Beginner–Challenge pack).
        beat_grid: Optional precomputed :class:`BeatGrid` (estimated when None).
        beat_tracker: Beat tracker backend (``"auto"``, ``"beat_this"``, ``"librosa"``).
        snap_beats: Apply song-level onset-strength phase alignment to beats.
        threshold: Slot probability cutoff (default ``0.5``).
        max_beats: Maximum beat chunk length (defaults to model's ``max_beats``).
        mert_layers: MERT hidden layers to average when extracting on the fly.
        mert_cache_suffix: Sibling MERT cache suffix checked before extraction.
        use_mert_cache: Cache extracted MERT features by audio SHA-256 hash.
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
        use_mert_cache=use_mert_cache,
    )
    n_frames = int(getattr(itgpt_model, "n_frames_per_beat", 24))
    beat_mert = ddcl_features.beats_to_mert_tensor(
        mert_100hz, grid.beat_times_with_end, n_frames=n_frames
    ).astype(np.float32)

    song_key = pathlib.Path(audio_path).stem
    onsets_by_difficulty: dict[str, np.ndarray] = {}
    for preset in difficulties:
        eff_density = preset.resolve_target_density(grid.bpm)
        onsets_by_difficulty[preset.name] = _predict_itgpt_onsets_for_mert(
            song_key,
            beat_mert,
            grid,
            itgpt_model,
            difficulty=preset.name,
            target_density=eff_density,
            meter=preset.meter,
            threshold=threshold,
            max_beats=max_beats,
            device=target_device,
            fallback_top_k=True,
            allowed_slots=preset.allowed_slots,
        )
    return onsets_by_difficulty, grid
