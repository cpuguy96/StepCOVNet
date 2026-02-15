"""Data collection and preprocessing for StepCovNet.

This module provides functionality to load audio and StepMania chart files,
process them into spectrograms and target vectors, and create a TensorFlow
dataset for training.
"""
import os
import pathlib

import librosa
import numpy as np
import tensorflow as tf

_DIFFICULTY_MAP = {'beginner': 0, 'easy': 1, 'medium': 2, 'hard': 3, 'challenge': 4}
_N_MELS = 128
_N_TARGET = 1
_F_MIN = 27.5
_F_MAX = 16000
HOP_COEFF = 0.01
_WIN_COEFF = 0.025


def _load_and_pair_files(data_dir: str) -> list[tuple[str, str]]:
    """Find paired audio files and StepMania chart files."""
    pairs = []
    for root, _, files in os.walk(data_dir):
        audio_files = [f for f in files if f.endswith(('.mp3', '.ogg', '.wav'))]
        chart_files = [f for f in files if f.endswith(('.txt'))]

        # Pair files with same stem (e.g., 'song.mp3' and 'song.sm')
        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching_charts = [f for f in chart_files if f.startswith(stem)]
            if matching_charts:
                pairs.append((
                    os.path.join(root, audio_file),
                    os.path.join(root, matching_charts[0])
                ))
    return pairs


def _parse_step_chart(chart_path: str) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Parse StepMania .sm file to extract step timings, BPM, and difficulty."""
    with open(chart_path, 'r') as f:
        f.readline()  # TITLE
        bpm = float(f.readline().removeprefix("BPM").strip())
        f.readline()  # NOTES
        difficulty_level = f.readline().strip().lower().split(" ")[1]
        difficulty_idx = _DIFFICULTY_MAP.get(difficulty_level, 2)
        times = []
        cols = []
        for line in f:
            if line.startswith('DIFFICULTY'):
                # TODO: Read off of multiple difficulties
                break
            # TODO: Use the type of note played and not just the presence
            _, timing = line.strip().split(" ")
            times.append(float(timing))
            cols.append(0)

    return np.array(times), np.array(cols, dtype=np.int32), bpm, difficulty_idx


def _audio_to_spectrogram(audio_path: str, target_sr: int = 44100) -> np.ndarray:
    """Convert audio to mel-spectrogram using LibROSA."""

    y, sr = librosa.load(audio_path, sr=target_sr)

    # Sample rate conversion (if necessary)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Normalize audio data
    y = y / np.max(np.abs(y))

    hop_length = int(round(target_sr * HOP_COEFF))
    win_length = int(round(target_sr * _WIN_COEFF))
    n_fft = 2 ** int(np.ceil(np.log(win_length) / np.log(2.0)))

    S = librosa.feature.melspectrogram(
        y=y,
        sr=target_sr,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        fmin=_F_MIN,
        fmax=_F_MAX,
        n_mels=_N_MELS,
    )

    val = librosa.power_to_db(S, ref=np.max)
    return val


def _temporal_augment(spec: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random time warping augmentation."""
    original_length = spec.shape[1]
    warp_factor = np.random.uniform(0.85, 1.15)  # Keep reasonable warp range

    # Calculate new length while maintaining original size
    new_length = int(original_length * warp_factor)

    print("new length after temporal augment:", new_length)

    # Resize spectrogram
    spec_resized = tf.image.resize(
        spec[np.newaxis],
        [_N_MELS, new_length],
        method='bilinear'
    ).numpy()[0]

    # Resize labels with nearest neighbor to preserve binary values
    labels_resized = tf.image.resize(
        labels[np.newaxis, ..., np.newaxis],
        [new_length, _N_TARGET],
        method='nearest'
    ).numpy()[0, ..., 0]

    # Maintain original length with padding/cropping
    if new_length > original_length:
        spec = spec_resized[:, :original_length]
        labels = labels_resized[:original_length, :]
    else:
        pad_width = original_length - new_length
        spec = np.pad(spec_resized, ((0, 0), (0, pad_width)), mode='edge')
        labels = np.pad(labels_resized, ((0, pad_width), (0, 0)), mode='constant')

    return spec, labels


def _create_target(times: np.ndarray, cols: np.ndarray, spec_length: int) -> np.ndarray:
    """Create target vector from step times and columns."""
    time_resolution = 0.1  # 100ms per frame
    target = np.zeros((spec_length, _N_TARGET), dtype=np.float32)
    for time, col in zip(times, cols):
        frame_idx = int(time / time_resolution)
        if frame_idx < spec_length:
            target[frame_idx, col] = 1.0
    return target


def _process_pair(audio_path: str, chart_path: str) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Process audio-chart pair into input features, target labels, and difficulty."""
    # Parse step chart
    times, cols, bpm, difficulty = tf.py_function(
        lambda p: _parse_step_chart(p.numpy().decode()),
        [chart_path],
        (tf.float32, tf.int32, tf.float32, tf.int32)
    )

    # Process audio
    spec = tf.py_function(
        lambda p: _audio_to_spectrogram(p.numpy().decode()),
        [audio_path],
        tf.float32
    )

    # Reshape spec to have n_mels in the last dimension
    spec = tf.transpose(spec, perm=[1, 0])  # Transpose spec here

    # Create time-aligned target vector
    spec_length = tf.shape(spec)[0]
    target = tf.py_function(
        lambda t, c, sl: _create_target(t.numpy(), c.numpy(), sl.numpy()),
        [times, cols, spec_length],
        tf.float32
    )

    # Apply temporal augmentation
    # Disabled by default since it causes OOM issues
    # spec, target = tf.py_function(
    #     temporal_augment,
    #     [spec, atarget],
    #     (tf.float32, tf.float32)
    # )

    # Enforce fixed shape after augmentation
    spec = tf.ensure_shape(spec, (None, _N_MELS))  # Preserve mel bands dimension
    target = tf.ensure_shape(target, (None, _N_TARGET))
    difficulty_tensor = tf.convert_to_tensor(difficulty, dtype=tf.int32)
    # difficulty_tensor = tf.expand_dims(difficulty_tensor, axis=0)  # Ensure difficulty is a scalar tensor

    return (spec, difficulty_tensor), target


def create_dataset(data_dir: str, batch_size: int = 16) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset pipeline for training StepCovNet.

    Args:
        data_dir: Path to the directory containing audio and .sm files.
        batch_size: Number of samples per training batch.

    Returns:
        A tf.data.Dataset yielding ((spectrogram, difficulty), target) tuples.
    """
    # Get list of audio-chart pairs
    pairs = _load_and_pair_files(data_dir)

    if not pairs:
        raise ValueError("No audio-chart pairs found in the specified directory.")

    # Create dataset from pairs
    ds = tf.data.Dataset.from_generator(
        lambda: pairs,
        output_types=(tf.string, tf.string)
    )

    # Process pairs in parallel
    ds = ds.map(
        lambda audio, chart: _process_pair(audio, chart),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and optimize
    ds = ds.padded_batch(batch_size,
                         padded_shapes=(
                             ((None, _N_MELS), ()),  # Spectrogram (mel bands x time), Difficulty (scalar)
                             (None, 1),  # Target (time x columns)
                         ),
                         padding_values=((0.0, -1), 0.0,))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
