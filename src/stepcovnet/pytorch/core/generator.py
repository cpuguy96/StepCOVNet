"""Zero-TF PyTorch step-chart generator matching ``stepcovnet.generator``."""

from __future__ import annotations

import dataclasses

import librosa
import numpy as np
import scipy.signal
import torch

from stepcovnet import constants
from stepcovnet.pytorch.core import arrow_models, onset_models


@dataclasses.dataclass(frozen=True)
class OutputData:
    """Generated step chart metadata and note list for a song.

    Attributes:
        title: Song title.
        bpm: Estimated or provided integer BPM.
        notes: Mapping from difficulty name to ``(timestamp_str, base4_arrow_str)`` tuples.
    """

    title: str
    bpm: int
    notes: dict[str, list[tuple[str, str]]]

    def generate_txt_output(self) -> str:
        """Format chart data as StepCOVNet plain-text note output.

        Returns:
            Formatted chart string.
        """
        lines = [f"TITLE {self.title}", f"BPM {self.bpm}", "NOTES"]
        for difficulty, entries in self.notes.items():
            lines.append(f"DIFFICULTY {difficulty}")
            lines.extend(f"{arrow} {timing}" for timing, arrow in entries)
        return "\n".join(lines) + "\n"


def _int_to_base4_string(n: int, min_digits: int = 4) -> str:
    if n < 0:
        raise ValueError("Input integer must be non-negative.")
    if n == 0:
        return "0".zfill(min_digits)
    digits: list[str] = []
    cur = int(n)
    while cur > 0:
        digits.append(str(cur % 4))
        cur //= 4
    return "".join(reversed(digits)).zfill(min_digits)


def post_process_predictions(
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    min_distance_ms: float = 50.0,
) -> np.ndarray:
    """Peak-pick per-frame onset probabilities into timestamps in seconds.

    Args:
        probabilities: Sigmoid onset probabilities ``(T,)`` or ``(T, 1)``.
        threshold: Minimum peak height.
        min_distance_ms: Minimum separation between peaks in milliseconds.

    Returns:
        Float64 array of onset timestamps in seconds.
    """
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    min_dist = max(1, int(round((min_distance_ms / 1000.0) / constants.HOP_COEFF)))
    peak_indices, _ = scipy.signal.find_peaks(
        probs, height=threshold, distance=min_dist
    )
    return peak_indices.astype(np.float64) * constants.HOP_COEFF


def audio_to_normalized_spectrogram(audio_path: str) -> np.ndarray:
    """Load audio and return a z-scored log-mel spectrogram ``(T, N_MELS)``.

    Args:
        audio_path: Path to input audio file.

    Returns:
        Float32 array of shape ``(n_frames, constants.N_MELS)``.
    """
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=constants.WAVEFORM_SAMPLES_PER_FRAME,
        n_mels=constants.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    mean = float(np.mean(log_mel))
    std = float(np.std(log_mel))
    return ((log_mel - mean) / max(std, 1e-6)).astype(np.float32)


def _estimate_bpm_from_audio(audio_path: str) -> int:
    y, sr = librosa.load(audio_path, sr=constants.TARGET_SR, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_f = float(np.atleast_1d(tempo)[0])
    if not (tempo_f > 0.0 and np.isfinite(tempo_f)):
        raise ValueError("Could not estimate BPM from audio; please provide bpm.")
    return max(1, min(9999, int(round(tempo_f))))


def _build_arrow_feed(
    model: arrow_models.ArrowSequenceModel,
    onsets: np.ndarray,
    normalized_spec: np.ndarray,
    bpm: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_t = max(float(np.max(onsets)), 1e-6)
    norm_onsets = (onsets / max_t).astype(np.float32)
    intervals = np.diff(onsets, prepend=onsets[0]).astype(np.float32)
    max_iv = max(float(np.max(intervals)), 1e-6)
    next_iv = np.diff(onsets, append=onsets[-1]).astype(np.float32)
    max_next = max(float(np.max(next_iv)), 1e-6)

    feed: dict[str, torch.Tensor] = {}
    for name in model.required_input_names():
        if name == "timing_input":
            arr = norm_onsets[np.newaxis, :, np.newaxis]
        elif name == "interval_input":
            arr = (intervals / max_iv)[np.newaxis, :, np.newaxis]
        elif name == "interval_log_input":
            log_iv = np.log1p(np.maximum(intervals, 0.0))
            arr = (log_iv / max(float(np.max(log_iv)), 1e-6))[np.newaxis, :, np.newaxis]
        elif name == "interval_next_input":
            arr = (next_iv / max_next)[np.newaxis, :, np.newaxis]
        elif name == "step_index_input":
            idx = np.linspace(0.0, 1.0, num=len(onsets), dtype=np.float32)
            arr = idx[np.newaxis, :, np.newaxis]
        elif name == "beat_phase_input":
            beats = onsets * (bpm / 60.0)
            arr = (beats - np.floor(beats)).astype(np.float32)[
                np.newaxis, :, np.newaxis
            ]
        elif name == "snippet_input":
            half = int(model.input_options.snippet_half_frames)
            width = 2 * half + 1
            n_frames, n_mels = normalized_spec.shape
            padded = np.pad(normalized_spec, ((half, half), (0, 0)), mode="constant")
            snippets = np.zeros((len(onsets), width, n_mels), dtype=np.float32)
            for i, t_sec in enumerate(onsets):
                center = int(
                    np.clip(round(float(t_sec) / constants.HOP_COEFF), 0, n_frames - 1)
                )
                snippets[i] = padded[center : center + width]
            arr = snippets[np.newaxis, ...]
        else:
            continue
        feed[name] = torch.from_numpy(arr.astype(np.float32)).to(device)
    return feed


@torch.no_grad()
def generate_output_data_from_spec(
    *,
    normalized_spec: np.ndarray,
    song_title: str,
    bpm: int,
    onset_model: onset_models.DenseOnsetTorchModel,
    arrow_model: arrow_models.ArrowSequenceModel,
    use_post_processing: bool = True,
    threshold: float = 0.5,
    difficulty: str = "Challenge",
    device: torch.device | None = None,
) -> OutputData:
    """Run PyTorch onset and arrow models on a precomputed spectrogram.

    Args:
        normalized_spec: Normalized spectrogram ``(T, F)``.
        song_title: Title written into the chart header.
        bpm: Integer song BPM.
        onset_model: Trained PyTorch dense onset model.
        arrow_model: Trained PyTorch arrow sequence model.
        use_post_processing: Apply SciPy peak picking when True.
        threshold: Onset threshold.
        difficulty: Difficulty label key for output notes.
        device: Optional compute device.

    Returns:
        Populated :class:`OutputData`.

    Raises:
        ValueError: If no onsets are predicted above threshold.
    """
    target_device = device or next(onset_model.parameters()).device
    onset_model.eval()
    arrow_model.eval()

    spec_t = (
        torch.from_numpy(np.asarray(normalized_spec, dtype=np.float32))
        .unsqueeze(0)
        .to(target_device)
    )
    onset_probs = onset_model(spec_t)[0].detach().cpu().numpy().reshape(-1)
    if use_post_processing:
        onsets = post_process_predictions(onset_probs, threshold=threshold)
    else:
        onsets = (
            np.where(onset_probs > threshold)[0].astype(np.float64)
            * constants.HOP_COEFF
        )

    if onsets.size == 0:
        raise ValueError("Failed to predict any onsets for the audio.")

    arrow_feed = _build_arrow_feed(
        arrow_model, onsets, normalized_spec, float(bpm), target_device
    )
    arrow_out = arrow_model(arrow_feed)
    probs = (
        arrow_out["output_probabilities"] if isinstance(arrow_out, dict) else arrow_out
    )
    arrow_ids = torch.argmax(probs[0], dim=-1).detach().cpu().numpy().tolist()
    note_rows: list[tuple[str, str]] = []
    for onset_sec, arrow_id in zip(onsets.tolist(), arrow_ids, strict=True):
        if int(arrow_id) == constants.ARROW_PADDING_CLASS:
            continue
        note_rows.append((str(onset_sec), _int_to_base4_string(int(arrow_id), 4)))

    return OutputData(
        title=song_title,
        bpm=int(bpm),
        notes={difficulty: note_rows},
    )


def generate_output_data(
    *,
    audio_path: str,
    song_title: str,
    bpm: int | None = None,
    onset_model: onset_models.DenseOnsetTorchModel,
    arrow_model: arrow_models.ArrowSequenceModel,
    use_post_processing: bool = True,
    threshold: float = 0.5,
    difficulty: str = "Challenge",
    device: torch.device | None = None,
) -> OutputData:
    """Generate step chart data from an audio file using PyTorch models.

    Args:
        audio_path: Path to input audio file.
        song_title: Title written into the chart header.
        bpm: Optional BPM (estimated with librosa if omitted).
        onset_model: Trained PyTorch dense onset model.
        arrow_model: Trained PyTorch arrow model.
        use_post_processing: Apply SciPy peak picking when True.
        threshold: Onset probability threshold.
        difficulty: Output difficulty key.
        device: Optional compute device.

    Returns:
        Populated :class:`OutputData`.
    """
    resolved_bpm = _estimate_bpm_from_audio(audio_path) if bpm is None else int(bpm)
    norm_spec = audio_to_normalized_spectrogram(audio_path)
    return generate_output_data_from_spec(
        normalized_spec=norm_spec,
        song_title=song_title,
        bpm=resolved_bpm,
        onset_model=onset_model,
        arrow_model=arrow_model,
        use_post_processing=use_post_processing,
        threshold=threshold,
        difficulty=difficulty,
        device=device,
    )
