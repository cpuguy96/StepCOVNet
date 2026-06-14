"""Keras audio frontend: raw waveform to temporal embeddings."""

import keras
import tensorflow as tf

TARGET_SAMPLE_RATE = 44100
MAX_AUDIO_SECONDS = 300.0
DEFAULT_FRAME_HOP_SEC = 0.01
DEFAULT_BASE_FILTERS = 32


def max_waveform_samples(
    max_audio_seconds: float = MAX_AUDIO_SECONDS,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> int:
    """Return padded waveform length for a duration cap.

    Args:
        max_audio_seconds: Maximum audio duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Sample count ``round(max_audio_seconds * sample_rate)``.
    """
    return int(round(max_audio_seconds * sample_rate))


def target_encoder_frames(
    max_audio_seconds: float,
    frame_hop_sec: float = DEFAULT_FRAME_HOP_SEC,
) -> int:
    """Return the target number of frontend time steps for a duration cap.

    Spacing follows ``frame_hop_sec`` (default 0.01 s ~ 10 ms). This is an
    internal downsampling target only; it is not tied to dense-onset hop constants.

    Args:
        max_audio_seconds: Maximum audio duration in seconds.
        frame_hop_sec: Target spacing between encoder frames in seconds.

    Returns:
        ``round(max_audio_seconds / frame_hop_sec)``.
    """
    return int(round(max_audio_seconds / frame_hop_sec))


def _apply_conv1d_frontend(
    audio_input: keras.KerasTensor,
    *,
    sample_rate: int,
    max_audio_seconds: float,
    frame_hop_sec: float,
    base_filters: int,
) -> keras.KerasTensor:
    """Strided Conv1D stack mapping waveform to ``(B, T_enc, base_filters)``."""
    target_frames = target_encoder_frames(max_audio_seconds, frame_hop_sec)
    num_samples = int(round(sample_rate * max_audio_seconds))
    stride = max(1, num_samples // max(target_frames, 1))
    x = keras.layers.Reshape((-1, 1), name="audio_channel")(audio_input)
    x = keras.layers.Conv1D(
        base_filters,
        kernel_size=7,
        strides=stride,
        padding="same",
        activation="gelu",
        name="frontend_proj",
    )(x)
    x = keras.layers.Conv1D(
        base_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="gelu",
        name="frontend_refine",
    )(x)
    return x


def build_cached_feature_frontend(
    *,
    input_features: int,
    output_features: int,
    max_frames: int,
    name: str = "onset_event_cached_frontend",
) -> keras.Model:
    """Project cached ``(T, D_in)`` features to ``(T, D_out)`` encoder embeddings."""
    features_input = keras.Input(
        shape=(max_frames, input_features),
        name="features",
        dtype=tf.float32,
    )
    x = keras.layers.Conv1D(
        output_features,
        kernel_size=1,
        padding="same",
        activation="gelu",
        name="cached_feature_proj",
    )(features_input)
    x = keras.layers.Conv1D(
        output_features,
        kernel_size=3,
        padding="same",
        activation="gelu",
        name="cached_feature_refine",
    )(x)
    return keras.Model(inputs=features_input, outputs=x, name=name)


def build_audio_frontend(
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_audio_seconds: float = MAX_AUDIO_SECONDS,
    frame_hop_sec: float = DEFAULT_FRAME_HOP_SEC,
    base_filters: int = DEFAULT_BASE_FILTERS,
    name: str = "onset_event_frontend",
) -> keras.Model:
    """Build a Keras model: mono waveform in, temporal embeddings out.

    Input shape is ``(num_samples,)`` with
    ``num_samples = round(max_audio_seconds * target_sample_rate)``.
    Output shape is ``(T_enc, base_filters)`` per batch item, where ``T_enc`` is
    chosen by strided Conv1D to approximate one step every ``frame_hop_sec``.

    Args:
        target_sample_rate: Audio sample rate in Hz.
        max_audio_seconds: Fixed input duration cap in seconds.
        frame_hop_sec: Target frontend frame spacing in seconds.
        base_filters: Embedding dimension ``D`` (Conv1D channel count).
        name: Keras model name.

    Returns:
        Keras ``Model`` with input ``audio`` and output embeddings.
    """
    num_samples = max_waveform_samples(max_audio_seconds, target_sample_rate)
    audio_input = keras.Input(
        shape=(num_samples,),
        name="audio",
        dtype=tf.float32,
    )
    embeddings = _apply_conv1d_frontend(
        audio_input,
        sample_rate=target_sample_rate,
        max_audio_seconds=max_audio_seconds,
        frame_hop_sec=frame_hop_sec,
        base_filters=base_filters,
    )
    return keras.Model(
        inputs=audio_input,
        outputs=embeddings,
        name=name,
    )
