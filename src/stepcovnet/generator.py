"""Generates step chart data from audio using StepCovNet models."""

import dataclasses

import numpy as np
import scipy
from keras import models

from stepcovnet import datasets


@dataclasses.dataclass(frozen=True)
class OutputData:
    """A data class representing the generated step chart information for a song.

    Attributes:
        title: The title of the song.
        bpm: The beats per minute of the song.
        notes: A dictionary mapping difficulty levels to a list of note tuples,
               where each tuple contains a timestamp (str) and an arrow pattern (str).
    """

    title: str
    bpm: int
    notes: dict[str, list[tuple[str, str]]]

    def generate_txt_output(self) -> str:
        """Generates a formatted string representation of the step chart.

        Returns:
            A string containing the song title, BPM, and note data for all difficulties.
        """
        title = "TITLE %s\n" % self.title
        bpm = "BPM %s\n" % str(self.bpm)
        notes = "NOTES\n"
        for difficulty in self.notes:
            notes += "DIFFICULTY %s\n" % difficulty
            for timing, arrow in self.notes[difficulty]:
                notes += f"{arrow} {timing}" + "\n"

        return "".join((title, bpm, notes))


def _int_to_base4_string(n: int, min_digits: int = 1) -> str:
    """
    Converts a non-negative integer (base 10) to its base 4 string representation.

    Args:
      n: The non-negative integer to convert.
      min_digits: The minimum number of digits the output string should have.
                  Leading zeros will be added if necessary. Defaults to 1.

    Returns:
      The string representation of the number in base 4.

    Raises:
      ValueError: If the input integer `n` is negative.
    """
    if n < 0:
        raise ValueError("Input integer must be non-negative.")
    if n == 0:
        # Handle the zero case separately for padding
        return "0".zfill(min_digits)

    base4_digits = []
    temp_n = n
    while temp_n > 0:
        remainder = temp_n % 4
        base4_digits.append(str(remainder))  # Convert remainder to string
        temp_n //= 4  # Integer division

    # The digits are generated in reverse order, so reverse the list
    base4_string = "".join(reversed(base4_digits))

    # Apply zero-padding if needed
    return base4_string.zfill(min_digits)


def _post_process_predictions(
    probabilities: np.ndarray, threshold: float = 0.5, min_distance_ms: float = 50.0
) -> np.ndarray:
    """Cleans up model predictions using peak picking to find precise onset times.

    Args:
        probabilities: The raw sigmoid output from the model for a single song.
                       Shape should be (time_steps, 1).
        threshold: The minimum probability value to consider for a peak (height).
        min_distance_ms: The minimum time in milliseconds between two onsets.

    Returns:
        A numpy array of timestamps (in seconds) for the detected onsets.
    """
    if probabilities.ndim > 1:
        probabilities = probabilities.flatten()

    # 1. Calculate the minimum distance in frames
    min_distance_frames = int(round((min_distance_ms / 1000.0) / datasets.HOP_COEFF))

    # 2. Use SciPy to find the peaks
    # The 'height' parameter acts as our probability threshold.
    # The 'distance' parameter enforces the temporal separation.
    peak_indices, _ = scipy.signal.find_peaks(
        probabilities, height=threshold, distance=min_distance_frames
    )

    # 3. Convert peak frame indices back to timestamps in seconds
    onset_times = peak_indices * datasets.HOP_COEFF

    return onset_times


def _create_txt_mapping(onsets: list, arrows: list) -> list[tuple[str, str]]:
    """
    Maps onset timestamps to their corresponding arrow patterns in base-4 string format.

    Args:
        onsets: A list of timestamps (floats or strings) representing note timings.
        arrows: A list of integer representations of arrow patterns.

    Returns:
        A list of tuples, each containing a timestamp string and a 4-digit base-4 arrow string.
    """
    note_data = []
    assert len(onsets) == len(arrows)
    for onset, arrow in zip(onsets, arrows):
        if not (int_arrow := int(arrow)):
            # Remove all padding arrows from the output.
            continue
        binary_arrow = _int_to_base4_string(int_arrow, min_digits=4)
        note_data.append((str(onset), str(binary_arrow)))
    return note_data


def generate_output_data(
    *,
    audio_path: str,
    song_title: str,
    bpm: int,
    onset_model: models.Model,
    arrow_model: models.Model,
    use_post_processing: bool = False,
) -> OutputData:
    """Generates step chart data for a given audio file using trained models.

    This function processes the audio into a spectrogram, predicts onset timings,
    and then predicts the arrow patterns for those onsets.

    Note: Spectrogram normalization is always applied in both training and
    inference for consistent results.

    Args:
        audio_path: Path to the input audio file.
        song_title: The title of the song.
        bpm: The beats per minute of the song.
        onset_model: A Keras model used to predict note onsets.
        arrow_model: A Keras model used to predict arrow types for given onsets.
        use_post_processing: Whether to use peak picking to refine onset timings.

    Returns:
        An OutputData object containing the song metadata and generated notes.

    Raises:
        ValueError: If failed to predict any onsets for the audio file.
    """
    spec = datasets.audio_to_spectrogram(audio_path).T
    normalized_spec = datasets.normalize_onset_spectrogram(spec)
    onset_pred = onset_model.predict(np.expand_dims(normalized_spec, axis=0))

    if use_post_processing:
        onsets = _post_process_predictions(onset_pred[0])
    else:
        onsets = np.where(onset_pred[0] > 0.5)[0] * datasets.HOP_COEFF

    if not onsets.shape[0]:
        raise ValueError("Failed to predict any onsets for the audio file.")

    normalized_onsets = np.expand_dims(onsets / np.max(onsets), axis=(0, -1))
    arrows_pred = arrow_model.predict(normalized_onsets)
    arrows = np.argmax(arrows_pred[0], axis=1)

    return OutputData(
        title=song_title,
        bpm=bpm,
        notes={
            # TODO(cpuguy96) - Support more difficulties
            "Challenge": _create_txt_mapping(onsets=list(onsets), arrows=list(arrows)),
        },
    )
