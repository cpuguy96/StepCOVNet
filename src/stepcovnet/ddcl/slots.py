"""Beat-grid 48-slot labels for DDCL placement (`omalley2025ddcl`).

Inspired by ``create_beat_dicts`` in
https://github.com/miguelomalley/DDCL/blob/5b1375c642bb708b3c66baf5d880fbf865b85097/smfiler.py
and ``label_to_vect_dict`` in ``util.py``. Labels are a length-48 binary vector
per integer beat (``M-slot48``). Chart ``#BPMS`` / ``#OFFSET`` come from our
``.chart.json`` (same SM timing as training), not ArrowVortex.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import constants


def beat_to_time_sec(
    beat: float,
    offset_sec: float,
    segments: Sequence[prep_models.BpmSegment],
) -> float:
    """Convert a beat index to audio seconds using the SM BPM ladder.

    Beat 0 is at ``-offset_sec`` in the music file, matching StepMania
    ``#OFFSET`` / ``#BPMS``.

    Args:
        beat: Beat position (may be fractional).
        offset_sec: Simfile ``#OFFSET``.
        segments: Non-empty BPM ladder.

    Returns:
        Time in seconds.

    Raises:
        ValueError: If ``segments`` is empty or a BPM is not positive.
    """
    if not segments:
        raise ValueError("bpm_segments must be non-empty")
    ordered = tuple(sorted(segments, key=lambda item: item.start_beat))
    time_sec = -float(offset_sec)
    cursor = 0.0
    bpm = float(ordered[0].bpm)
    next_idx = 0
    while next_idx < len(ordered) and ordered[next_idx].start_beat <= cursor + 1e-12:
        bpm = float(ordered[next_idx].bpm)
        next_idx += 1
    if bpm <= 0.0:
        raise ValueError(f"BPM must be positive, got {bpm}")
    target = float(beat)
    while cursor < target - 1e-12:
        next_change = (
            ordered[next_idx].start_beat if next_idx < len(ordered) else target
        )
        next_beat = min(target, float(next_change))
        time_sec += (next_beat - cursor) * 60.0 / bpm
        cursor = next_beat
        if (
            next_idx < len(ordered)
            and abs(cursor - ordered[next_idx].start_beat) < 1e-9
        ):
            bpm = float(ordered[next_idx].bpm)
            if bpm <= 0.0:
                raise ValueError(f"BPM must be positive, got {bpm}")
            next_idx += 1
    return time_sec


def time_to_beat(
    time_sec: float,
    offset_sec: float,
    segments: Sequence[prep_models.BpmSegment],
) -> float:
    """Invert :func:`beat_to_time_sec`.

    Args:
        time_sec: Audio time in seconds.
        offset_sec: Simfile ``#OFFSET``.
        segments: Non-empty BPM ladder.

    Returns:
        Beat position.

    Raises:
        ValueError: If ``segments`` is empty or a BPM is not positive.
    """
    if not segments:
        raise ValueError("bpm_segments must be non-empty")
    ordered = tuple(sorted(segments, key=lambda item: item.start_beat))
    remaining = float(time_sec) + float(offset_sec)
    cursor = 0.0
    bpm = float(ordered[0].bpm)
    next_idx = 0
    while next_idx < len(ordered) and ordered[next_idx].start_beat <= cursor + 1e-12:
        bpm = float(ordered[next_idx].bpm)
        next_idx += 1
    if bpm <= 0.0:
        raise ValueError(f"BPM must be positive, got {bpm}")
    while remaining > 1e-12:
        next_change = (
            float(ordered[next_idx].start_beat) if next_idx < len(ordered) else np.inf
        )
        span = next_change - cursor
        span_sec = span * 60.0 / bpm
        if remaining <= span_sec + 1e-12:
            return cursor + remaining * bpm / 60.0
        remaining -= span_sec
        cursor = next_change
        bpm = float(ordered[next_idx].bpm)
        if bpm <= 0.0:
            raise ValueError(f"BPM must be positive, got {bpm}")
        next_idx += 1
    return cursor


def bpm_at_beat(
    beat: float,
    segments: Sequence[prep_models.BpmSegment],
) -> float:
    """Return the BPM in force at ``beat``.

    Args:
        beat: Beat position.
        segments: Non-empty BPM ladder.

    Returns:
        BPM value.

    Raises:
        ValueError: If ``segments`` is empty.
    """
    if not segments:
        raise ValueError("bpm_segments must be non-empty")
    ordered = tuple(sorted(segments, key=lambda item: item.start_beat))
    bpm = float(ordered[0].bpm)
    for segment in ordered:
        if beat + 1e-12 < segment.start_beat:
            break
        bpm = float(segment.bpm)
    return bpm


def upsample_rhythm_bits(
    bits: Sequence[int],
    n_slots: int = constants.N_SLOTS,
) -> np.ndarray:
    """Upsample a variable-length 0/1 beat pattern to ``n_slots``.

    Port of ``label_to_vect_dict(..., force_max_len=48)`` in DDCL ``util.py``:
    ``step = n_slots / len(bits)`` and copies ``bits[j / step]`` onto the
    coarse grid.

    Args:
        bits: 0/1 occupancy at the chart's native subdivision.
        n_slots: Output length (paper: 48).

    Returns:
        Float32 vector of shape ``(n_slots,)``.

    Raises:
        ValueError: If ``n_slots`` is not positive.
    """
    if n_slots < 1:
        raise ValueError(f"n_slots must be at least 1, got {n_slots}")
    out = np.zeros((n_slots,), dtype=np.float32)
    values = [int(bit) for bit in bits]
    if not values:
        return out
    step = n_slots / len(values)
    stride = int(step)
    if stride < 1:
        stride = 1
    for index in range(0, n_slots, stride):
        source = int(index / step)
        if source < len(values):
            out[index] = float(values[source])
    return out


def slot_index(fractional_beat: float, n_slots: int = constants.N_SLOTS) -> int:
    """Map a fractional beat into ``[0, n_slots)``.

    Args:
        fractional_beat: Beat position (integer part is the beat index).
        n_slots: Slots per beat.

    Returns:
        Slot index.

    Raises:
        ValueError: If ``n_slots`` is not positive.
    """
    if n_slots < 1:
        raise ValueError(f"n_slots must be at least 1, got {n_slots}")
    frac = float(fractional_beat) - np.floor(float(fractional_beat))
    if frac < 0.0:
        frac += 1.0
    return int(np.floor(frac * n_slots + 1e-9)) % n_slots


def times_to_slot_matrix(
    times_sec: np.ndarray,
    offset_sec: float,
    segments: Sequence[prep_models.BpmSegment],
    *,
    n_slots: int = constants.N_SLOTS,
) -> np.ndarray:
    """Build a ``(n_beats, n_slots)`` binary occupancy grid.

    Each encoded chart row (tap / hold head / tail) sets one slot. Beat 0 is
    included so silent intro beats stay aligned with the audio, matching
    ``create_beat_dicts`` iterating from beat 0.

    Args:
        times_sec: Onset times in seconds.
        offset_sec: Simfile ``#OFFSET``.
        segments: BPM ladder.
        n_slots: Slots per beat.

    Returns:
        Float32 matrix of 0/1 slot labels.

    Raises:
        ValueError: If there are no onsets.
    """
    times = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times_sec must contain at least one onset")
    beats = np.array(
        [time_to_beat(float(time), offset_sec, segments) for time in times],
        dtype=np.float64,
    )
    last_beat = int(np.floor(float(np.max(beats)) + 1e-9))
    n_beats = last_beat + 1
    slots = np.zeros((n_beats, n_slots), dtype=np.float32)
    for beat in beats:
        beat_idx = int(np.floor(beat + 1e-9))
        if beat_idx < 0:
            continue
        if beat_idx >= n_beats:
            beat_idx = n_beats - 1
        slots[beat_idx, slot_index(beat, n_slots=n_slots)] = 1.0
    return slots


def beat_times_sec(
    n_beats: int,
    offset_sec: float,
    segments: Sequence[prep_models.BpmSegment],
) -> np.ndarray:
    """Return start times for integer beats ``0 .. n_beats`` (inclusive end).

    The extra endpoint is the start of beat ``n_beats``, used as the exclusive
    end of the last labeled beat when resampling audio (DDCL ``time_dict``).

    Args:
        n_beats: Number of labeled integer beats.
        offset_sec: Simfile ``#OFFSET``.
        segments: BPM ladder.

    Returns:
        Float64 times of length ``n_beats + 1``.

    Raises:
        ValueError: If ``n_beats`` is not positive.
    """
    if n_beats < 1:
        raise ValueError(f"n_beats must be at least 1, got {n_beats}")
    return np.array(
        [
            beat_to_time_sec(float(beat), offset_sec, segments)
            for beat in range(n_beats + 1)
        ],
        dtype=np.float64,
    )


def stream_features(
    n_beats: int,
    meter: int,
    segments: Sequence[prep_models.BpmSegment],
) -> np.ndarray:
    """Per-beat ``[meter, bpm]`` stream features.

    DDCL ``create_beat_dicts`` stores ``[difficulty_fine, cur_bpm, coarse]``;
    ``generatorify_from_fp_list_onset`` keeps only the first two
    (``models.py``).

    Args:
        n_beats: Number of integer beats.
        meter: Chart ``#METER`` (fine difficulty).
        segments: BPM ladder.

    Returns:
        Float32 array of shape ``(n_beats, 2)``.

    Raises:
        ValueError: If ``n_beats`` is not positive.
    """
    if n_beats < 1:
        raise ValueError(f"n_beats must be at least 1, got {n_beats}")
    meter_value = float(meter)
    rows = np.zeros((n_beats, constants.STREAM_DIM), dtype=np.float32)
    for beat_idx in range(n_beats):
        rows[beat_idx, 0] = meter_value
        rows[beat_idx, 1] = float(bpm_at_beat(float(beat_idx), segments))
    return rows
