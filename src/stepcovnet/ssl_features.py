"""Self-supervised audio feature extraction and loading for onset training.

Provides offline MERT feature extraction (PyTorch + transformers) and numpy-based
loading/resampling aligned to the onset model frame grid (HOP_COEFF).
"""

from __future__ import annotations

import os
import pathlib

import librosa
import numpy as np
from scipy import interpolate

from stepcovnet import constants

MERT_SAMPLE_RATE = 24000
DEFAULT_MERT_MODEL = "m-a-p/MERT-v1-330M"
DEFAULT_MERT_LAYER = 6
MERT_FILE_SUFFIX = ".mert.npy"
MERT_CHUNK_SECONDS = 30.0
MIN_MERT_CHUNK_SAMPLES = 400


def mert_npy_path(
    audio_path: str,
    features_dir: str = "",
    data_root: str = "",
) -> str:
    """Resolve the path where precomputed MERT features are stored for an audio file.

    Args:
        audio_path: Path to the source audio file.
        features_dir: Root directory for ``.mert.npy`` files. When empty, features
            are expected beside the audio file.
        data_root: When ``features_dir`` is set, preserve relative paths under this
            root so nested training layouts do not collide on song stem alone.

    Returns:
        Absolute or relative path to the ``.mert.npy`` file.
    """
    audio = pathlib.Path(audio_path)
    if not features_dir:
        return str(audio.with_suffix(MERT_FILE_SUFFIX))
    rel = os.path.relpath(audio_path, data_root) if data_root else audio.name
    rel_npy = str(pathlib.Path(rel).with_suffix(MERT_FILE_SUFFIX))
    return os.path.join(features_dir, rel_npy)


def resample_features_to_hop_grid(
    features: np.ndarray,
    audio_duration_sec: float,
    hop_sec: float = constants.HOP_COEFF,
) -> np.ndarray:
    """Linearly resample frame-level features onto the onset detection time grid.

    Args:
        features: Source features with shape ``(time_steps, feature_dim)``.
        audio_duration_sec: Duration of the source audio in seconds.
        hop_sec: Target frame spacing in seconds (defaults to datasets.HOP_COEFF).

    Returns:
        Resampled features with shape ``(n_target_frames, feature_dim)``, float32.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D (time, dim); got shape {features.shape!r}"
        )
    n_src, feature_dim = features.shape
    n_target = max(1, int(round(audio_duration_sec / hop_sec)))
    if n_src == n_target:
        return features.astype(np.float32)
    if n_src == 1:
        return np.tile(features, (n_target, 1)).astype(np.float32)

    src_times = np.linspace(0.0, audio_duration_sec, n_src, endpoint=False)
    tgt_times = np.arange(n_target, dtype=np.float64) * hop_sec
    resampled = interpolate.interp1d(
        src_times,
        features,
        axis=0,
        kind="linear",
        fill_value="extrapolate",  # type: ignore[arg-type]
    )(tgt_times)
    return np.asarray(resampled, dtype=np.float32).reshape(n_target, feature_dim)


def resample_features_to_frame_count(
    features: np.ndarray,
    n_frames: int,
    hop_sec: float = constants.HOP_COEFF,
) -> np.ndarray:
    """Linearly resample features onto a fixed number of onset time steps.

    Frame ``j`` is placed at ``j * hop_sec`` seconds, matching chart target
    indexing used by the mel spectrogram pipeline.

    Args:
        features: Source features with shape ``(time_steps, feature_dim)``.
        n_frames: Target number of time steps (for example mel STFT frame count).
        hop_sec: Target frame spacing in seconds.

    Returns:
        Resampled features with shape ``(n_frames, feature_dim)``, float32.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D (time, dim); got shape {features.shape!r}"
        )
    n_frames = max(1, int(n_frames))
    n_src, feature_dim = features.shape
    if n_src == n_frames:
        return features.astype(np.float32)
    if n_src == 1:
        return np.tile(features, (n_frames, 1)).astype(np.float32)
    if n_frames == 1:
        return features[:1].astype(np.float32)

    span_sec = (n_frames - 1) * hop_sec
    src_times = np.linspace(0.0, span_sec, n_src, endpoint=False)
    tgt_times = np.arange(n_frames, dtype=np.float64) * hop_sec
    resampled = interpolate.interp1d(
        src_times,
        features,
        axis=0,
        kind="linear",
        fill_value="extrapolate",  # type: ignore[arg-type]
    )(tgt_times)
    return np.asarray(resampled, dtype=np.float32).reshape(n_frames, feature_dim)


def load_mert_features(
    audio_path: str,
    features_dir: str = "",
    data_root: str = "",
) -> np.ndarray:
    """Load precomputed MERT features aligned to the onset frame grid.

    Args:
        audio_path: Path to the source audio file (used to locate the ``.npy`` file).
        features_dir: Directory containing ``.mert.npy`` files, or empty to load beside audio.
        data_root: Root training directory when preserving nested paths in ``features_dir``.

    Returns:
        Feature array with shape ``(time_steps, feature_dim)``, float32.

    Raises:
        FileNotFoundError: If the expected ``.mert.npy`` file does not exist.
        ValueError: If the loaded array is not 2D.
    """
    npy_path = mert_npy_path(audio_path, features_dir, data_root)
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(
            f"MERT features not found at {npy_path!r}. "
            "Run scripts/extract_mert_features.py on your training data first."
        )
    features = np.load(npy_path)
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D MERT features in {npy_path!r}; got shape {features.shape!r}"
        )
    return features.astype(np.float32)


def _require_ssl_deps():
    """Import optional PyTorch/transformers dependencies for MERT extraction.

    Returns:
        Tuple ``(torch_module, transformers_module)``.

    Raises:
        ImportError: If torch or transformers is not installed.
    """
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "MERT extraction requires optional dependencies. "
            "Install with: pip install '.[ssl]'"
        ) from exc
    return torch, transformers


def _load_mert_model(model_name: str, device: str):
    """Load a MERT model and processor from Hugging Face.

    Args:
        model_name: Hugging Face model id.
        device: Torch device string (for example ``cpu`` or ``cuda``).

    Returns:
        Tuple ``(model, processor)`` ready for inference.
    """
    torch, transformers = _require_ssl_deps()
    processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = transformers.AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model, processor


def _mert_hidden_states_for_chunk(
    waveform: np.ndarray,
    *,
    model,
    processor,
    layer: int,
    device: str,
) -> np.ndarray:
    """Run MERT on one waveform chunk and return hidden states for one layer.

    Args:
        waveform: 1D float waveform at MERT_SAMPLE_RATE.
        model: Loaded MERT model.
        processor: Matching Wav2Vec2 feature extractor.
        layer: Hidden-state layer index (0 = embedding output).
        device: Torch device string.

    Returns:
        Hidden states with shape ``(time_steps, hidden_dim)``, float32.
    """
    if waveform.size == 0:
        return np.zeros((0, constants.MERT_HIDDEN_SIZE), dtype=np.float32)
    torch, _ = _require_ssl_deps()
    inputs = processor(
        waveform,
        sampling_rate=MERT_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    model_kwargs = {"input_values": input_values}
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        outputs = model(**model_kwargs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(
            f"MERT layer index {layer} out of range (model has {len(hidden_states)} layers)"
        )
    hidden = hidden_states[layer].squeeze(0).cpu().numpy()
    return hidden.astype(np.float32)


def extract_mert_features_from_audio(
    audio_path: str,
    *,
    model_name: str = DEFAULT_MERT_MODEL,
    layer: int = DEFAULT_MERT_LAYER,
    device: str = "cpu",
    chunk_seconds: float = MERT_CHUNK_SECONDS,
) -> np.ndarray:
    """Extract MERT hidden states from an audio file, resampled to the onset grid.

    Args:
        audio_path: Path to an audio file readable by librosa.
        model_name: Hugging Face MERT model id.
        layer: Hidden-state layer index to extract.
        device: Torch device for inference.
        chunk_seconds: Maximum chunk length in seconds for long files.

    Returns:
        Feature array with shape ``(time_steps, hidden_dim)`` on the HOP_COEFF grid.
    """
    waveform, _ = librosa.load(audio_path, sr=MERT_SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path!r}")
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    model, processor = _load_mert_model(model_name, device)
    chunk_samples = max(MIN_MERT_CHUNK_SAMPLES, int(round(chunk_seconds * MERT_SAMPLE_RATE)))
    chunks: list[np.ndarray] = []
    for start in range(0, waveform.size, chunk_samples):
        chunk = waveform[start : start + chunk_samples]
        if 0 < chunk.size < MIN_MERT_CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, MIN_MERT_CHUNK_SAMPLES - chunk.size))
        chunks.append(
            _mert_hidden_states_for_chunk(
                chunk,
                model=model,
                processor=processor,
                layer=layer,
                device=device,
            )
        )
    if not chunks:
        merged = np.zeros((0, constants.MERT_HIDDEN_SIZE), dtype=np.float32)
    elif len(chunks) == 1:
        merged = chunks[0]
    else:
        merged = np.concatenate(chunks, axis=0)

    from stepcovnet import datasets  # noqa: PLC0415

    n_frames = datasets.onset_frame_count(audio_path)
    return resample_features_to_frame_count(merged, n_frames)


def save_mert_features(
    features: np.ndarray,
    output_path: str,
) -> str:
    """Save MERT features to a ``.mert.npy`` file.

    Args:
        features: Array with shape ``(time_steps, feature_dim)``.
        output_path: Destination ``.npy`` path.

    Returns:
        The output path written.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D (time, dim); got shape {features.shape!r}"
        )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, features.astype(np.float32))
    return output_path


def extract_and_save_mert_features(
    audio_path: str,
    output_path: str,
    *,
    model_name: str = DEFAULT_MERT_MODEL,
    layer: int = DEFAULT_MERT_LAYER,
    device: str = "cpu",
    chunk_seconds: float = MERT_CHUNK_SECONDS,
) -> str:
    """Extract MERT features from audio and write them to disk.

    Args:
        audio_path: Source audio path.
        output_path: Destination ``.mert.npy`` path.
        model_name: Hugging Face MERT model id.
        layer: Hidden-state layer index.
        device: Torch device for inference.
        chunk_seconds: Chunk length for long audio.

    Returns:
        The output path written.
    """
    features = extract_mert_features_from_audio(
        audio_path,
        model_name=model_name,
        layer=layer,
        device=device,
        chunk_seconds=chunk_seconds,
    )
    return save_mert_features(features, output_path)
