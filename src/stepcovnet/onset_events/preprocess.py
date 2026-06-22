"""Pre-processing stage: raw audio or cached features for the event onset encoder."""

import numpy as np

from stepcovnet import config as project_config
from stepcovnet import constants
from stepcovnet import datasets as dense_datasets
from stepcovnet.onset_events import audio, frontend

FRONTEND_CONV1D = "conv1d"
FRONTEND_MEL = "mel"
FRONTEND_MERT = "mert"
SUPPORTED_FRONTENDS = (FRONTEND_CONV1D, FRONTEND_MEL, FRONTEND_MERT)


def validate_frontend(name: str) -> str:
    """Return ``name`` when it is a supported frontend identifier."""
    if name not in SUPPORTED_FRONTENDS:
        raise ValueError(
            f"unsupported frontend {name!r}; expected one of {SUPPORTED_FRONTENDS}"
        )
    return name


def encoder_feature_dim(frontend_name: str) -> int:
    """Return the channel dimension for one encoder time step."""
    frontend_name = validate_frontend(frontend_name)
    if frontend_name == FRONTEND_CONV1D:
        return 1
    if frontend_name == FRONTEND_MEL:
        return constants.N_MELS
    return constants.MERT_HIDDEN_SIZE


def max_encoder_frames(
    max_audio_seconds: float,
    frame_hop_sec: float = frontend.DEFAULT_FRAME_HOP_SEC,
) -> int:
    """Return padded encoder time-step count for a duration cap."""
    return frontend.target_encoder_frames(max_audio_seconds, frame_hop_sec)


def _duration_from_truncated_waveform(
    waveform: np.ndarray,
    target_sample_rate: int,
) -> float:
    if waveform.size == 0:
        return 0.0
    return float(waveform.size) / float(target_sample_rate)


def _pad_feature_frames(
    features: np.ndarray,
    max_frames: int,
) -> np.ndarray:
    """Pad or truncate ``(T, D)`` features to ``(max_frames, D)``."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    feature_dim = features.shape[1]
    padded = np.zeros((max_frames, feature_dim), dtype=np.float32)
    copy_frames = min(max_frames, features.shape[0])
    if copy_frames > 0:
        padded[:copy_frames] = features[:copy_frames].astype(np.float32, copy=False)
    return padded


def load_preprocessed_encoder_input(
    audio_path: str,
    *,
    frontend_name: str,
    target_sample_rate: int,
    max_samples: int,
    max_audio_seconds: float,
    frame_hop_sec: float = frontend.DEFAULT_FRAME_HOP_SEC,
    mert_features_dir: str = "",
    data_root: str = "",
) -> tuple[np.ndarray, np.int32, np.float32]:
    """Load the encoder input tensor and metadata for one audio file.

    Args:
        audio_path: Path to the audio file.
        frontend_name: ``conv1d``, ``mel``, or ``mert``.
        target_sample_rate: Sample rate in Hz for waveform I/O.
        max_samples: Padded waveform sample count (``conv1d`` only).
        max_audio_seconds: Maximum clip duration in seconds.
        frame_hop_sec: Encoder frame spacing in seconds for cached features.
        mert_features_dir: Root directory for ``.mert.npy`` files.
        data_root: Training data root for nested MERT paths.

    Returns:
        Tuple of ``(encoder_input, audio_length, duration)``. ``encoder_input`` is
        a 1D waveform for ``conv1d`` or 2D ``(max_frames, feature_dim)`` for
        ``mel`` / ``mert``.
    """
    frontend_name = validate_frontend(frontend_name)
    waveform = audio.load_waveform(audio_path, target_sample_rate=target_sample_rate)
    waveform = audio.truncate_waveform(waveform, max_samples)
    audio_length = np.int32(waveform.size)
    duration = np.float32(
        _duration_from_truncated_waveform(waveform, target_sample_rate)
    )

    if frontend_name == FRONTEND_CONV1D:
        encoder_input = audio.pad_waveform(waveform, max_samples)
        return encoder_input, audio_length, duration

    max_frames = max_encoder_frames(max_audio_seconds, frame_hop_sec)
    active_frames = min(
        max_frames,
        max(0, int(round(float(duration) / frame_hop_sec))),
    )

    if frontend_name == FRONTEND_MEL:
        features = dense_datasets.load_onset_features(
            audio_path,
            project_config.FeatureSource.MEL,
        )
        features = dense_datasets.normalize_onset_spectrogram(features)
    else:
        features = dense_datasets.load_onset_features(
            audio_path,
            project_config.FeatureSource.MERT,
            mert_features_dir,
            data_root,
        )

    features = features[:active_frames] if active_frames > 0 else features[:0]

    encoder_input = _pad_feature_frames(features, max_frames)
    return encoder_input, audio_length, duration
