"""Generates step chart data from audio using PyTorch StepCOVNet models."""

from __future__ import annotations

import collections.abc
import dataclasses
import math
from typing import Any

import librosa
import numpy as np
import scipy.signal
import torch

from stepcovnet import beat_tracking, constants, datasets

_SM_TICKS_PER_MEASURE = 192
_SM_MEASURE_DIVISORS = (4, 8, 12, 16, 24, 32, 48, 64, 192)


def _format_sm_measures(
    note_rows: list[tuple[str, str]],
    *,
    bpm: float,
    anchor_time_sec: float,
    beat_grid: beat_tracking.BeatGrid | None,
) -> tuple[str, float]:
    """Quantize chart notes into StepMania measure blocks and return ``(notes_body, sm_offset)``."""
    if not note_rows:
        return "0000\n0000\n0000\n0000", -anchor_time_sec

    times = np.asarray([float(t) for t, _ in note_rows], dtype=np.float64)
    arrows = [str(a).zfill(4)[:4] for _, a in note_rows]
    bpm_f = max(float(bpm), 1.0)
    measure_sec = 4.0 * 60.0 / bpm_f

    raw_beats = (
        beat_grid.times_to_beats(times, anchor_time_sec=anchor_time_sec)
        if beat_grid is not None
        else (times - anchor_time_sec) * (bpm_f / 60.0)
    )
    min_beat = float(np.min(raw_beats))
    shift_measures = int(math.ceil(-min_beat / 4.0)) if min_beat < -1e-6 else 0
    beats = raw_beats + 4.0 * float(shift_measures)
    effective_anchor = anchor_time_sec - float(shift_measures) * measure_sec
    sm_offset = -effective_anchor

    ticks = np.maximum(0, np.rint(beats * (_SM_TICKS_PER_MEASURE / 4.0)).astype(int))
    max_measure = int(np.max(ticks)) // _SM_TICKS_PER_MEASURE

    by_measure: dict[int, dict[int, str]] = {}
    for tick, arrow in zip(ticks, arrows, strict=True):
        m_idx = int(tick) // _SM_TICKS_PER_MEASURE
        pos = int(tick) % _SM_TICKS_PER_MEASURE
        by_measure.setdefault(m_idx, {})[pos] = arrow

    measure_blocks: list[str] = []
    for m_idx in range(max_measure + 1):
        events = by_measure.get(m_idx, {})
        chosen_div = _SM_TICKS_PER_MEASURE
        for div in _SM_MEASURE_DIVISORS:
            step = _SM_TICKS_PER_MEASURE // div
            if all(pos % step == 0 for pos in events):
                chosen_div = div
                break
        step = _SM_TICKS_PER_MEASURE // chosen_div
        rows = [events.get(r * step, "0000") for r in range(chosen_div)]
        measure_blocks.append("\n".join(rows))

    return "\n,\n".join(measure_blocks), sm_offset


@dataclasses.dataclass(frozen=True)
class OutputData:
    """Generated step chart metadata and note list for a song.

    Attributes:
        title: Song title.
        bpm: Beats per minute of the song.
        notes: Mapping from difficulty name to a list of ``(timestamp_str, arrow_str)`` tuples.
        offset: StepMania ``#OFFSET`` in seconds (``-anchor_time_sec`` where
            Measure 1 Beat 0 starts at ``anchor_time_sec``).
        beat_grid: Optional :class:`stepcovnet.beat_tracking.BeatGrid` carrying
            tracked beats and downbeats from raw audio.
        meters: Optional mapping from difficulty name to numeric StepMania meter.
    """

    title: str
    bpm: int
    notes: dict[str, list[tuple[str, str]]]
    offset: float = 0.0
    beat_grid: beat_tracking.BeatGrid | None = None
    meters: dict[str, int] | None = None

    def generate_txt_output(self) -> str:
        """Generate a formatted string representation of the step chart.

        Returns:
            Formatted plain-text chart string.
        """
        title = f"TITLE {self.title}\n"
        bpm = f"BPM {self.bpm}\n"
        notes = "NOTES\n"
        for difficulty in self.notes:
            notes += f"DIFFICULTY {difficulty}\n"
            for timing, arrow in self.notes[difficulty]:
                notes += f"{arrow} {timing}\n"
        return "".join((title, bpm, notes))

    def generate_sm_output(
        self,
        *,
        music_filename: str = "",
        meter: int = 10,
    ) -> str:
        """Generate a StepMania ``.sm`` simfile aligned to tracked downbeats.

        Uses :attr:`beat_grid` (when available) to anchor Measure 1 Beat 0 on
        the detected downbeat via ``#OFFSET`` and quantizes notes cleanly onto
        the StepMania 192-tick measure grid.

        Args:
            music_filename: Audio filename written to ``#MUSIC``.
            meter: Default difficulty meter written to ``#NOTES`` blocks when
                :attr:`meters` does not provide a per-difficulty meter.

        Returns:
            Formatted StepMania ``.sm`` file content.
        """
        bpm_val = (
            float(self.beat_grid.bpm) if self.beat_grid is not None else float(self.bpm)
        )
        anchor_time = (
            self.beat_grid.anchor_time_sec
            if self.beat_grid is not None
            else -float(self.offset)
        )

        # Compute a shared #OFFSET across all difficulty charts so every chart
        # aligns to the same Measure 1 Beat 0.
        all_rows = [row for rows in self.notes.values() for row in rows]
        _, shared_offset = _format_sm_measures(
            all_rows,
            bpm=bpm_val,
            anchor_time_sec=anchor_time,
            beat_grid=self.beat_grid,
        )
        shared_anchor = -shared_offset

        header = (
            f"#TITLE:{self.title};\n"
            f"#SUBTITLE:;\n"
            f"#ARTIST:;\n"
            f"#CREDIT:StepCOVNet;\n"
            f"#MUSIC:{music_filename};\n"
            f"#OFFSET:{shared_offset:.6f};\n"
            f"#SAMPLESTART:0.000;\n"
            f"#SAMPLELENGTH:15.000;\n"
            f"#SELECTABLE:YES;\n"
            f"#BPMS:0.000={bpm_val:.3f};\n"
            f"#STOPS:;\n"
        )
        note_blocks: list[str] = []
        for difficulty, rows in self.notes.items():
            chart_meter = (
                int(self.meters[difficulty])
                if self.meters is not None and difficulty in self.meters
                else int(meter)
            )
            body, _ = _format_sm_measures(
                rows,
                bpm=bpm_val,
                anchor_time_sec=shared_anchor,
                beat_grid=self.beat_grid,
            )
            note_blocks.append(
                "\n#NOTES:\n"
                "     dance-single:\n"
                "     :\n"
                f"     {difficulty}:\n"
                f"     {chart_meter}:\n"
                "     0.000,0.000,0.000,0.000,0.000:\n"
                f"{body}\n"
                ";\n"
            )
        return header + "".join(note_blocks)


def _int_to_base4_string(n: int, min_digits: int = 1) -> str:
    """Convert a non-negative integer to its base-4 string representation.

    Args:
        n: Non-negative integer to convert.
        min_digits: Minimum number of output digits (left zero-padded).

    Returns:
        Base-4 string representation of ``n``.

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("Input integer must be non-negative.")
    if n == 0:
        return "0".zfill(min_digits)
    base4_digits: list[str] = []
    temp_n = int(n)
    while temp_n > 0:
        base4_digits.append(str(temp_n % 4))
        temp_n //= 4
    return "".join(reversed(base4_digits)).zfill(min_digits)


def _post_process_predictions(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_distance_ms: float = 50.0,
) -> np.ndarray:
    """Peak-pick raw per-frame onset probabilities into timestamps in seconds.

    Args:
        probabilities: Raw sigmoid output for one song, shape ``(T,)`` or ``(T, 1)``.
        threshold: Minimum peak height.
        min_distance_ms: Minimum time separation between peaks in milliseconds.

    Returns:
        Numpy array of detected onset timestamps in seconds.
    """
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    min_distance_frames = max(
        1, int(round((min_distance_ms / 1000.0) / datasets.HOP_COEFF))
    )
    peak_indices, _ = scipy.signal.find_peaks(
        probs, height=threshold, distance=min_distance_frames
    )
    return peak_indices * datasets.HOP_COEFF


def _create_txt_mapping(onsets: list, arrows: list) -> list[tuple[str, str]]:
    """Map onset timestamps to base-4 arrow strings, skipping padding arrows.

    Args:
        onsets: Sequence of onset timestamps.
        arrows: Sequence of integer arrow pattern IDs.

    Returns:
        List of ``(timestamp_str, base4_arrow_str)`` tuples.
    """
    note_data: list[tuple[str, str]] = []
    assert len(onsets) == len(arrows)
    for onset, arrow in zip(onsets, arrows, strict=False):
        int_arrow = int(arrow)
        if int_arrow == constants.ARROW_PADDING_CLASS:
            continue
        binary_arrow = _int_to_base4_string(int_arrow, min_digits=4)
        note_data.append((str(onset), str(binary_arrow)))
    return note_data


def _estimate_bpm_from_audio(
    audio_path: str,
    *,
    beat_tracker: str = beat_tracking.TRACKER_LIBROSA,
) -> int:
    """Estimate BPM from an audio file using the configured beat tracker.

    Args:
        audio_path: Path to the input audio file.
        beat_tracker: Beat tracker backend (``"librosa"``, ``"beat_this"``, or
            ``"auto"``). Defaults to ``"librosa"`` when called directly for
            backwards compatibility.

    Returns:
        Estimated integer BPM clamped to ``[1, 9999]``.

    Raises:
        ValueError: If tempo cannot be estimated.
    """
    if beat_tracker != beat_tracking.TRACKER_LIBROSA:
        grid = beat_tracking.estimate_beats(audio_path, tracker=beat_tracker)
        return max(1, min(9999, int(round(grid.bpm))))
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_f = float(np.atleast_1d(tempo)[0])
    if not (tempo_f > 0 and np.isfinite(tempo_f)):
        raise ValueError("Could not estimate BPM from audio; please provide --bpm.")
    return max(1, min(9999, int(round(tempo_f))))


def _is_itgpt_placement_model(model: Any) -> bool:
    """Return True if ``model`` is a beat-slot ITGPT placement model."""
    return (
        callable(getattr(model, "forward_logits", None))
        and hasattr(model, "n_enc_layers")
        and hasattr(model, "audio_feature_type")
    )


def _model_device(model: Any) -> torch.device:
    if isinstance(model, torch.nn.Module):
        param = next(model.parameters(), None)
        if param is not None:
            return param.device
    return torch.device("cpu")


def _run_onset_inference(onset_model: Any, normalized_spec: np.ndarray) -> np.ndarray:
    batch_np = np.expand_dims(normalized_spec.astype(np.float32), axis=0)
    if isinstance(onset_model, torch.nn.Module):
        device = _model_device(onset_model)
        onset_model.eval()
        with torch.no_grad():
            out = onset_model(torch.from_numpy(batch_np).to(device))
        return out.detach().cpu().numpy()
    if hasattr(onset_model, "predict"):
        return np.asarray(onset_model.predict(batch_np))
    return np.asarray(onset_model(batch_np))


def _required_arrow_inputs(arrow_model: Any) -> list[str]:
    req_fn = getattr(arrow_model, "required_input_names", None)
    if callable(req_fn):
        names: Any = req_fn()
        return [str(name) for name in names]
    return ["timing_input"]


def _run_arrow_inference(
    arrow_model: Any,
    arrow_feed_np: dict[str, np.ndarray],
) -> np.ndarray:
    if isinstance(arrow_model, torch.nn.Module):
        device = _model_device(arrow_model)
        arrow_model.eval()
        feed_t = {
            k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
            for k, v in arrow_feed_np.items()
        }
        with torch.no_grad():
            out = arrow_model(feed_t)
        if isinstance(out, dict):
            out = out["output_probabilities"]
        return out.detach().cpu().numpy()
    if hasattr(arrow_model, "predict"):
        if len(arrow_feed_np) == 1:
            return np.asarray(arrow_model.predict(next(iter(arrow_feed_np.values()))))
        return np.asarray(arrow_model.predict(arrow_feed_np))
    out = arrow_model(arrow_feed_np)
    if isinstance(out, dict):
        out = out["output_probabilities"]
    return np.asarray(out)


def _predict_arrows_for_onsets(
    onsets: np.ndarray,
    *,
    arrow_model: Any,
    required_arrow: list[str],
    bpm: int,
    beat_grid: beat_tracking.BeatGrid | None,
    normalized_spec: np.ndarray | None,
) -> list[tuple[str, str]]:
    """Build arrow-model feature inputs for ``onsets`` and decode arrow rows."""
    max_onset = float(np.max(onsets))
    denom = max_onset if max_onset > 0.0 else 1.0
    normalized_onsets = np.expand_dims(
        (onsets / denom).astype(np.float32), axis=(0, -1)
    )

    arrow_feed: dict[str, np.ndarray] = {}
    for base in required_arrow:
        if base == "timing_input":
            arrow_feed[base] = normalized_onsets
        elif base == "interval_input":
            intervals_norm = datasets.normalized_intervals_from_times(onsets)
            arrow_feed[base] = np.expand_dims(intervals_norm, axis=(0, -1))
        elif base == "interval_log_input":
            interval_log = datasets.log_normalized_intervals_from_times(onsets)
            arrow_feed[base] = np.expand_dims(interval_log, axis=(0, -1))
        elif base == "interval_next_input":
            interval_next = datasets.next_interval_normalized_from_times(onsets)
            arrow_feed[base] = np.expand_dims(interval_next, axis=(0, -1))
        elif base == "step_index_input":
            step_idx = datasets.step_index_normalized(len(onsets))
            arrow_feed[base] = np.expand_dims(step_idx, axis=(0, -1))
        elif base == "beat_phase_input":
            beat_ph = (
                beat_grid.beat_phase_for_times(onsets)
                if beat_grid is not None
                else datasets.beat_phase_from_times_bpm(onsets, float(bpm))
            )
            arrow_feed[base] = np.expand_dims(beat_ph, axis=(0, -1))
        elif base == "snippet_input":
            assert normalized_spec is not None
            in_opts = getattr(arrow_model, "input_options", None)
            half_frames = int(getattr(in_opts, "snippet_half_frames", 0))
            snippets = datasets.extract_snippets_from_spec(
                normalized_spec, onsets, half_frames
            )
            arrow_feed[base] = np.expand_dims(snippets, axis=0)

    arrows_pred = _run_arrow_inference(arrow_model, arrow_feed)
    arrows = np.argmax(arrows_pred[0], axis=-1)
    return _create_txt_mapping(onsets=list(onsets), arrows=list(arrows))


def _decode_dense_onsets_for_preset(
    probs_1d: np.ndarray,
    preset: beat_tracking.DifficultyPreset,
    *,
    base_threshold: float,
    use_post_processing: bool,
) -> np.ndarray:
    """Decode legacy frame-level onset probabilities for one difficulty preset."""
    probs = np.asarray(probs_1d, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return np.zeros(0, dtype=np.float64)

    ratio = min(1.0, max(0.1, float(preset.target_density) / 3.0))
    eff_threshold = min(
        0.95,
        float(base_threshold) + (1.0 - float(base_threshold)) * (1.0 - ratio) * 0.5,
    )
    min_dist_ms = 50.0 / ratio

    if use_post_processing:
        onsets = _post_process_predictions(
            probs, threshold=eff_threshold, min_distance_ms=min_dist_ms
        )
    else:
        min_distance_frames = max(
            1, int(round((min_dist_ms / 1000.0) / datasets.HOP_COEFF))
        )
        raw_indices = np.where(probs > eff_threshold)[0]
        kept: list[int] = []
        last_idx = -min_distance_frames
        for idx in raw_indices:
            idx_i = int(idx)
            if idx_i - last_idx >= min_distance_frames:
                kept.append(idx_i)
                last_idx = idx_i
        onsets = np.asarray(kept, dtype=np.float64) * datasets.HOP_COEFF

    if onsets.size == 0 and float(np.max(probs)) > 0.0:
        best_idx = int(np.argmax(probs))
        onsets = np.asarray([best_idx * datasets.HOP_COEFF], dtype=np.float64)
    return onsets


def generate_output_data(
    *,
    audio_path: str,
    song_title: str,
    bpm: int | None = None,
    onset_model: Any,
    arrow_model: Any,
    use_post_processing: bool = False,
    beat_tracker: str = beat_tracking.TRACKER_AUTO,
    snap_beats: bool = False,
    target_density: float = 2.0,
    meter: int = 10,
    threshold: float = 0.5,
    all_difficulties: bool = False,
    difficulties: collections.abc.Sequence[beat_tracking.DifficultyPreset]
    | None = None,
) -> OutputData:
    """Generate step chart data for an audio file using PyTorch models.

    Supports both legacy frame-level dense onset models and beat-slot
    ITGPT-MERT placement models (using the built-in pure-PyTorch ``beat_this``
    tracker by default, with a ``librosa`` fallback, to extract the beat grid
    from raw audio).

    Args:
        audio_path: Path to the input audio file.
        song_title: Song title written into the output header.
        bpm: Optional song BPM; estimated from audio when omitted.
        onset_model: Trained PyTorch dense onset or ITGPT placement model.
        arrow_model: Trained PyTorch arrow sequence model.
        use_post_processing: Whether to apply peak picking to dense onset probabilities.
        beat_tracker: Beat tracker backend (``"auto"``, ``"beat_this"``, ``"librosa"``).
        snap_beats: Align estimated beats to audio onset-strength peaks via a
            constant song-level phase offset.
        target_density: Target steps per beat for density-conditioned ITGPT models.
        meter: Difficulty meter for meter-conditioned ITGPT models.
        threshold: Onset probability threshold.
        all_difficulties: When True, generate a full 5-difficulty pack
            (``Beginner`` through ``Challenge``) in a single beat-tracking and
            MERT pass.
        difficulties: Optional custom sequence of
            :class:`stepcovnet.beat_tracking.DifficultyPreset` overrides (implies
            multi-difficulty pack generation).

    Returns:
        Populated :class:`OutputData`.

    Raises:
        ValueError: If no onsets are predicted, ``difficulties`` is empty, or BPM
            cannot be estimated.
    """
    required_arrow = _required_arrow_inputs(arrow_model)
    normalized_spec: np.ndarray | None = None
    beat_grid: beat_tracking.BeatGrid | None = None
    pack_mode = bool(all_difficulties or difficulties is not None)

    if pack_mode:
        presets = (
            tuple(difficulties)
            if difficulties is not None
            else beat_tracking.DEFAULT_DIFFICULTY_PACK
        )
        if not presets:
            raise ValueError("difficulties must contain at least one DifficultyPreset")

        if _is_itgpt_placement_model(onset_model):
            onsets_by_diff, beat_grid = beat_tracking.predict_itgpt_pack_from_audio(
                audio_path,
                onset_model,
                difficulties=presets,
                beat_tracker=beat_tracker,
                snap_beats=snap_beats,
                threshold=threshold,
                device=_model_device(onset_model),
            )
            if bpm is None:
                bpm = max(1, min(9999, int(round(beat_grid.bpm))))
            if "snippet_input" in required_arrow:
                spec = datasets.audio_to_spectrogram(audio_path).T
                normalized_spec = datasets.normalize_onset_spectrogram(spec)
        else:
            if bpm is None:
                if beat_tracker != beat_tracking.TRACKER_LIBROSA:
                    beat_grid = beat_tracking.estimate_beats(
                        audio_path,
                        tracker=beat_tracker,
                        snap_to_onsets=snap_beats,
                    )
                    bpm = max(1, min(9999, int(round(beat_grid.bpm))))
                else:
                    bpm = _estimate_bpm_from_audio(
                        audio_path, beat_tracker=beat_tracker
                    )
            spec = datasets.audio_to_spectrogram(audio_path).T
            normalized_spec = datasets.normalize_onset_spectrogram(spec)
            onset_pred = _run_onset_inference(onset_model, normalized_spec)
            probs_1d = np.asarray(onset_pred[0]).reshape(-1)
            onsets_by_diff = {
                preset.name: _decode_dense_onsets_for_preset(
                    probs_1d,
                    preset,
                    base_threshold=threshold,
                    use_post_processing=use_post_processing,
                )
                for preset in presets
            }

        notes_by_diff: dict[str, list[tuple[str, str]]] = {}
        for preset in presets:
            diff_onsets = onsets_by_diff[preset.name]
            if not diff_onsets.shape[0]:
                raise ValueError("Failed to predict any onsets for the audio file.")
            notes_by_diff[preset.name] = _predict_arrows_for_onsets(
                diff_onsets,
                arrow_model=arrow_model,
                required_arrow=required_arrow,
                bpm=int(bpm),
                beat_grid=beat_grid,
                normalized_spec=normalized_spec,
            )

        return OutputData(
            title=song_title,
            bpm=int(bpm),
            notes=notes_by_diff,
            offset=beat_grid.sm_offset_sec if beat_grid is not None else 0.0,
            beat_grid=beat_grid,
            meters={preset.name: int(preset.meter) for preset in presets},
        )

    if _is_itgpt_placement_model(onset_model):
        onsets, beat_grid = beat_tracking.predict_itgpt_onsets_from_audio(
            audio_path,
            onset_model,
            beat_tracker=beat_tracker,
            snap_beats=snap_beats,
            target_density=target_density,
            meter=meter,
            threshold=threshold,
            device=_model_device(onset_model),
        )
        if bpm is None:
            bpm = max(1, min(9999, int(round(beat_grid.bpm))))
        if "snippet_input" in required_arrow:
            spec = datasets.audio_to_spectrogram(audio_path).T
            normalized_spec = datasets.normalize_onset_spectrogram(spec)
    else:
        if bpm is None:
            if beat_tracker != beat_tracking.TRACKER_LIBROSA:
                beat_grid = beat_tracking.estimate_beats(
                    audio_path,
                    tracker=beat_tracker,
                    snap_to_onsets=snap_beats,
                )
                bpm = max(1, min(9999, int(round(beat_grid.bpm))))
            else:
                bpm = _estimate_bpm_from_audio(audio_path, beat_tracker=beat_tracker)
        spec = datasets.audio_to_spectrogram(audio_path).T
        normalized_spec = datasets.normalize_onset_spectrogram(spec)
        onset_pred = _run_onset_inference(onset_model, normalized_spec)

        if use_post_processing:
            onsets = _post_process_predictions(onset_pred[0], threshold=threshold)
        else:
            probs_1d = np.asarray(onset_pred[0]).reshape(-1)
            onsets = np.where(probs_1d > threshold)[0] * datasets.HOP_COEFF

    if not onsets.shape[0]:
        raise ValueError("Failed to predict any onsets for the audio file.")

    return OutputData(
        title=song_title,
        bpm=int(bpm),
        notes={
            "Challenge": _predict_arrows_for_onsets(
                onsets,
                arrow_model=arrow_model,
                required_arrow=required_arrow,
                bpm=int(bpm),
                beat_grid=beat_grid,
                normalized_spec=normalized_spec,
            ),
        },
        offset=beat_grid.sm_offset_sec if beat_grid is not None else 0.0,
        beat_grid=beat_grid,
    )


__all__ = [
    "OutputData",
    "_create_txt_mapping",
    "_estimate_bpm_from_audio",
    "_int_to_base4_string",
    "_post_process_predictions",
    "generate_output_data",
]
