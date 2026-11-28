"""Self-supervised audio feature extraction and loading for onset training.

Provides offline MERT feature extraction (PyTorch + transformers) and numpy-based
loading/resampling aligned to the onset model frame grid (HOP_COEFF).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pathlib
import typing

import librosa
import numpy as np
from scipy import interpolate

from stepcovnet import constants, mel_onset

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
    *,
    suffix: str = MERT_FILE_SUFFIX,
) -> str:
    """Resolve the path where precomputed MERT features are stored for an audio file.

    Args:
        audio_path: Path to the source audio file.
        features_dir: Root directory for ``.mert.npy`` files. When empty, features
            are expected beside the audio file.
        data_root: When ``features_dir`` is set, preserve relative paths under this
            root so nested training layouts do not collide on song stem alone.
        suffix: Output filename suffix (default ``.mert.npy``).

    Returns:
        Absolute or relative path to the MERT ``.npy`` file.
    """
    audio = pathlib.Path(audio_path)
    if not features_dir:
        return str(audio.with_suffix(suffix))
    rel = os.path.relpath(audio_path, data_root) if data_root else audio.name
    rel_npy = str(pathlib.Path(rel).with_suffix(suffix))
    return str(pathlib.Path(features_dir) / rel_npy)


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

    Raises:
        ValueError: If ``features`` is not 2D.
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

    Raises:
        ValueError: If ``features`` is not 2D.
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


def resample_stacked_features_to_frame_count(
    features: np.ndarray,
    n_frames: int,
    hop_sec: float = constants.HOP_COEFF,
) -> np.ndarray:
    """Linearly resample stacked multi-layer features onto ``n_frames`` steps.

    Args:
        features: Source features with shape ``(time_steps, n_layers, feature_dim)``.
        n_frames: Target number of time steps.
        hop_sec: Target frame spacing in seconds.

    Returns:
        Resampled float32 array with shape ``(n_frames, n_layers, feature_dim)``.

    Raises:
        ValueError: If ``features`` is not 3D.
    """
    if features.ndim != 3:
        raise ValueError(
            f"stacked features must be 3D (time, layers, dim); got shape {features.shape!r}"
        )
    n_src, n_layers, feature_dim = features.shape
    flat = features.reshape(n_src, n_layers * feature_dim)
    resampled = resample_features_to_frame_count(flat, n_frames, hop_sec=hop_sec)
    return resampled.reshape(max(1, int(n_frames)), n_layers, feature_dim)


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
    if not pathlib.Path(npy_path).is_file():
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
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
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


def stack_mert_layers(layer_arrays: list[np.ndarray]) -> np.ndarray:
    """Per-layer standardize and stack MERT hidden states along a layer axis.

    Normalizing each layer across the hidden axis keeps all layers on a common
    scale for learnable softmax layer mixing (SUPERB protocol).

    Args:
        layer_arrays: Non-empty list of ``(time_steps, hidden_dim)`` arrays.

    Returns:
        Float32 array of shape ``(time_steps, n_layers, hidden_dim)``.

    Raises:
        ValueError: If ``layer_arrays`` is empty.
    """
    if not layer_arrays:
        raise ValueError("layer_arrays must be non-empty")
    normalized: list[np.ndarray] = []
    for arr in layer_arrays:
        x = np.asarray(arr, dtype=np.float32)
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized.append((x - mean) / np.maximum(std, 1e-6))
    return np.stack(normalized, axis=1).astype(np.float32)


def combine_mert_layers(layer_arrays: list[np.ndarray]) -> np.ndarray:
    """Average multiple MERT hidden-state layers after per-layer normalization.

    Normalizing each layer across the hidden axis prevents deeper layers with
    larger activation norms from drowning out early percussive layers.

    Args:
        layer_arrays: Non-empty list of ``(time_steps, hidden_dim)`` arrays.

    Returns:
        Float32 array of shape ``(time_steps, hidden_dim)``.

    Raises:
        ValueError: If ``layer_arrays`` is empty.
    """
    if not layer_arrays:
        raise ValueError("layer_arrays must be non-empty")
    if len(layer_arrays) == 1:
        return np.asarray(layer_arrays[0], dtype=np.float32)
    return np.mean(stack_mert_layers(layer_arrays), axis=1).astype(np.float32)


def _mert_hidden_states_for_chunk(
    waveform: np.ndarray,
    *,
    model: typing.Any,
    processor: typing.Any,
    layer: int,
    device: str,
    layers: tuple[int, ...] | None = None,
    stack_layers: bool = False,
) -> np.ndarray:
    """Run MERT on one waveform chunk and return single- or multi-layer states.

    Args:
        waveform: 1D float waveform at MERT_SAMPLE_RATE.
        model: Loaded MERT model.
        processor: Matching Wav2Vec2 feature extractor.
        layer: Hidden-state layer index (0 = embedding output) used when
            ``layers`` is None.
        device: Torch device string.
        layers: Optional tuple of layer indices to layer-normalize and average
            (or stack when ``stack_layers`` is True).
        stack_layers: If True, return ``(time_steps, n_layers, hidden_dim)``
            instead of averaging layers.

    Returns:
        Hidden states with shape ``(time_steps, hidden_dim)`` (or
        ``(time_steps, n_layers, hidden_dim)`` when ``stack_layers`` is True),
        float32.

    Raises:
        ValueError: If any requested layer index is out of range.
    """
    selected = layers if layers is not None else (layer,)
    if not selected:
        raise ValueError("layers must be non-empty when provided")
    if waveform.size == 0:
        if stack_layers:
            return np.zeros(
                (0, len(selected), constants.MERT_HIDDEN_SIZE), dtype=np.float32
            )
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
    extracted: list[np.ndarray] = []
    for idx in selected:
        if idx < 0 or idx >= len(hidden_states):
            raise ValueError(
                f"MERT layer index {idx} out of range (model has {len(hidden_states)} layers)"
            )
        extracted.append(hidden_states[idx].squeeze(0).cpu().numpy().astype(np.float32))
    if stack_layers:
        return stack_mert_layers(extracted)
    return combine_mert_layers(extracted)


def extract_mert_features_from_audio(
    audio_path: str,
    *,
    model_name: str = DEFAULT_MERT_MODEL,
    layer: int = DEFAULT_MERT_LAYER,
    layers: tuple[int, ...] | None = None,
    stack_layers: bool = False,
    device: str = "cpu",
    chunk_seconds: float = MERT_CHUNK_SECONDS,
    model: object = None,
    processor: object = None,
) -> np.ndarray:
    """Extract MERT hidden states from an audio file, resampled to the onset grid.

    Args:
        audio_path: Path to an audio file readable by librosa.
        model_name: Hugging Face MERT model id.
        layer: Hidden-state layer index to extract when ``layers`` is None.
        layers: Optional tuple of layer indices to normalize and combine/stack.
        stack_layers: If True, keep the layer axis ``(time, n_layers, dim)``.
        device: Torch device for inference.
        chunk_seconds: Maximum chunk length in seconds for long files.
        model: Optional pre-loaded MERT model (avoids reloading weights per file).
        processor: Optional matching feature extractor.

    Returns:
        Feature array on the HOP_COEFF grid with shape
        ``(time_steps, hidden_dim)`` or ``(time_steps, n_layers, hidden_dim)``.

    Raises:
        ValueError: If the audio file is empty.
    """
    waveform, _ = librosa.load(audio_path, sr=MERT_SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path!r}")
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    if model is None or processor is None:
        model, processor = _load_mert_model(model_name, device)
    chunk_samples = max(
        MIN_MERT_CHUNK_SAMPLES, int(round(chunk_seconds * MERT_SAMPLE_RATE))
    )
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
                layers=layers,
                stack_layers=stack_layers,
                device=device,
            )
        )
    if not chunks:
        n_layers = len(layers) if layers is not None else 1
        merged = (
            np.zeros((0, n_layers, constants.MERT_HIDDEN_SIZE), dtype=np.float32)
            if stack_layers
            else np.zeros((0, constants.MERT_HIDDEN_SIZE), dtype=np.float32)
        )
    elif len(chunks) == 1:
        merged = chunks[0]
    else:
        merged = np.concatenate(chunks, axis=0)

    n_frames = mel_onset.onset_frame_count(audio_path)
    if stack_layers:
        return resample_stacked_features_to_frame_count(merged, n_frames)
    return resample_features_to_frame_count(merged, n_frames)


def save_mert_features(
    features: np.ndarray,
    output_path: str,
    *,
    allow_3d: bool = False,
    save_fp16: bool = False,
) -> str:
    """Save MERT features to a ``.mert.npy`` file.

    Args:
        features: Array with shape ``(time_steps, feature_dim)``, or
            ``(time_steps, n_layers, feature_dim)`` when ``allow_3d`` is True.
        output_path: Destination ``.npy`` path.
        allow_3d: If True, permit 3D stacked multi-layer arrays.
        save_fp16: If True, cast to float16 on disk to halve cache size.

    Returns:
        The output path written.

    Raises:
        ValueError: If ``features`` has an unsupported rank.
    """
    valid_dims = (2, 3) if allow_3d else (2,)
    if features.ndim not in valid_dims:
        expected = "2D or 3D" if allow_3d else "2D (time, dim)"
        raise ValueError(f"features must be {expected}; got shape {features.shape!r}")
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_dtype = np.float16 if save_fp16 else np.float32
    np.save(out_path, features.astype(target_dtype))
    return str(out_path)


def extract_and_save_mert_features(
    audio_path: str,
    output_path: str,
    *,
    model_name: str = DEFAULT_MERT_MODEL,
    layer: int = DEFAULT_MERT_LAYER,
    layers: tuple[int, ...] | None = None,
    stack_layers: bool = False,
    save_fp16: bool = False,
    device: str = "cpu",
    chunk_seconds: float = MERT_CHUNK_SECONDS,
    model: object = None,
    processor: object = None,
) -> str:
    """Extract MERT features from audio and write them to disk.

    Args:
        audio_path: Source audio path.
        output_path: Destination ``.mert.npy`` path.
        model_name: Hugging Face MERT model id.
        layer: Hidden-state layer index when ``layers`` is None.
        layers: Optional tuple of layer indices to normalize and combine/stack.
        stack_layers: If True, save stacked ``(time, n_layers, dim)`` arrays.
        save_fp16: If True, save array as float16.
        device: Torch device for inference.
        chunk_seconds: Chunk length for long audio.
        model: Optional pre-loaded MERT model.
        processor: Optional matching feature extractor.

    Returns:
        The output path written.
    """
    features = extract_mert_features_from_audio(
        audio_path,
        model_name=model_name,
        layer=layer,
        layers=layers,
        stack_layers=stack_layers,
        device=device,
        chunk_seconds=chunk_seconds,
        model=model,
        processor=processor,
    )
    return save_mert_features(
        features,
        output_path,
        allow_3d=stack_layers,
        save_fp16=save_fp16,
    )


def audio_file_sha256(audio_path: str | pathlib.Path) -> str:
    """Compute a streaming SHA-256 hex digest of an audio file's raw bytes.

    Args:
        audio_path: Path to an existing audio file.

    Returns:
        Lowercase 64-character SHA-256 hexadecimal digest.
    """
    hasher = hashlib.sha256()
    with pathlib.Path(audio_path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def mert_hash_cache_path(
    audio_path: str | pathlib.Path,
    *,
    cache_dir: str | pathlib.Path,
    model_name: str = DEFAULT_MERT_MODEL,
    layer: int = DEFAULT_MERT_LAYER,
    layers: tuple[int, ...] | None = None,
    stack_layers: bool = False,
) -> pathlib.Path:
    """Resolve the deterministic audio-hash cache path for extracted MERT features.

    Keyed by the audio file's SHA-256 digest and the MERT extraction signature
    (model ID, selected layers, and stacking mode) so re-runs on the same audio
    skip MERT inference without colliding across audio edits or model configs.

    Args:
        audio_path: Path to the source audio file.
        cache_dir: Directory where cached ``.mert.npy`` files are stored.
        model_name: Hugging Face MERT model id.
        layer: Single hidden-state layer used when ``layers`` is None.
        layers: Optional tuple of hidden-state layer indices.
        stack_layers: Whether layers are kept stacked rather than averaged.

    Returns:
        Path to the cache file under ``cache_dir``.
    """
    audio_hash = audio_file_sha256(audio_path)
    selected = layers if layers is not None else (int(layer),)
    cfg_payload = f"{model_name}|layers={','.join(str(int(i)) for i in selected)}|stack={int(bool(stack_layers))}"
    cfg_hash = hashlib.sha256(cfg_payload.encode("utf-8")).hexdigest()
    return (
        pathlib.Path(cache_dir) / f"{audio_hash[:16]}_{cfg_hash[:12]}{MERT_FILE_SUFFIX}"
    )


def save_mert_features_atomic(
    features: np.ndarray,
    output_path: str | pathlib.Path,
    *,
    allow_3d: bool = False,
) -> pathlib.Path:
    """Atomically write ``float32`` MERT features to disk via temp file + rename.

    Always stores ``float32`` (never ``float16``) to preserve numerical fidelity
    across onset inference runs and prevent partial writes on interruption.

    Args:
        features: 2D ``(time, dim)`` or 3D ``(time, layers, dim)`` feature array.
        output_path: Destination ``.npy`` path.
        allow_3d: If True, permit 3D stacked multi-layer arrays.

    Returns:
        Path written.

    Raises:
        ValueError: If ``features`` has an unsupported rank.
    """
    valid_dims = (2, 3) if allow_3d else (2,)
    if features.ndim not in valid_dims:
        expected = "2D or 3D" if allow_3d else "2D (time, dim)"
        raise ValueError(f"features must be {expected}; got shape {features.shape!r}")
    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / f".{out_path.name}.{os.getpid()}.tmp"
    arr = np.asarray(features, dtype=np.float32)
    try:
        with tmp_path.open("wb") as f:
            np.save(f, arr)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
    return out_path


def load_or_extract_mert_features_cached(
    audio_path: str,
    *,
    cache_dir: str | pathlib.Path | None = None,
    model_name: str = DEFAULT_MERT_MODEL,
    layer: int = DEFAULT_MERT_LAYER,
    layers: tuple[int, ...] | None = None,
    stack_layers: bool = False,
    device: str = "cpu",
    chunk_seconds: float = MERT_CHUNK_SECONDS,
    model: object = None,
    processor: object = None,
) -> np.ndarray:
    """Load MERT features from an audio-hash cache, or extract and cache them.

    Args:
        audio_path: Path to an audio file readable by librosa.
        cache_dir: Directory for audio-hash-keyed ``.mert.npy`` files. When
            ``None``, caching is bypassed and features are extracted directly.
        model_name: Hugging Face MERT model id.
        layer: Hidden-state layer index when ``layers`` is None.
        layers: Optional tuple of layer indices to normalize and combine/stack.
        stack_layers: If True, keep the layer axis ``(time, n_layers, dim)``.
        device: Torch device for inference.
        chunk_seconds: Maximum chunk length in seconds for long files.
        model: Optional pre-loaded MERT model.
        processor: Optional matching feature extractor.

    Returns:
        Float32 feature array on the HOP_COEFF grid.
    """
    cache_path: pathlib.Path | None = None
    expected_ndim = 3 if stack_layers else 2
    if cache_dir is not None and pathlib.Path(audio_path).is_file():
        with contextlib.suppress(OSError):
            cache_path = mert_hash_cache_path(
                audio_path,
                cache_dir=cache_dir,
                model_name=model_name,
                layer=layer,
                layers=layers,
                stack_layers=stack_layers,
            )
            if cache_path.is_file():
                try:
                    cached = np.load(cache_path)
                    if (
                        isinstance(cached, np.ndarray)
                        and cached.ndim == expected_ndim
                        and cached.size > 0
                        and bool(np.all(np.isfinite(cached)))
                    ):
                        return cached.astype(np.float32, copy=False)
                except (OSError, ValueError):
                    pass

    features = extract_mert_features_from_audio(
        audio_path,
        model_name=model_name,
        layer=layer,
        layers=layers,
        stack_layers=stack_layers,
        device=device,
        chunk_seconds=chunk_seconds,
        model=model,
        processor=processor,
    )
    if cache_path is not None:
        with contextlib.suppress(OSError):
            save_mert_features_atomic(features, cache_path, allow_3d=stack_layers)
    return features
