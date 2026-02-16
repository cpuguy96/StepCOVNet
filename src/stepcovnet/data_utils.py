import os
import pathlib

import numpy as np

_DIFFICULTY_MAP = {"beginner": 0, "easy": 1, "medium": 2, "hard": 3, "challenge": 4}


def _base4_to_int(base4_string: str) -> int:
    """
    Converts a string representation of a base 4 number to its base 10 integer equivalent.

    Args:
      base4_string: The string representing the number in base 4.
                    Should only contain characters '0', '1', '2', '3'.

    Returns:
      The integer (base 10) equivalent of the input base 4 string.
    """
    if not base4_string:
        raise ValueError("Input string cannot be empty.")

    # Check for invalid characters (optional but good practice)
    valid_chars = set('0123')
    if not set(base4_string).issubset(valid_chars):
        raise ValueError(
            f"Invalid character found in base 4 string: '{base4_string}'. Only '0', '1', '2', '3' are allowed.")

    return int(base4_string, 4)


def load_and_pair_files(data_dir: str) -> list[tuple[str, str]]:
    """Find paired audio files and StepMania chart files."""
    pairs = []
    for root, _, files in os.walk(data_dir):
        audio_files = [f for f in files if f.endswith((".mp3", ".ogg", ".wav"))]
        chart_files = [f for f in files if f.endswith((".txt"))]

        # Pair files with same stem (e.g., 'song.mp3' and 'song.sm')
        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching_charts = [f for f in chart_files if f.startswith(stem)]
            if matching_charts:
                pairs.append(
                    (
                        os.path.join(root, audio_file),
                        os.path.join(root, matching_charts[0]),
                    )
                )
    return pairs


def parse_step_chart(chart_path: str, binary_timings: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Parse StepMania .sm file to extract step timings and note encodings.

    Args:
        chart_path: Path to the StepMania .sm file.
        binary_timings: If True, returns 0 for all note encodings, effectively
                        treating the output as binary (step vs. no step).

    Returns:
        A tuple containing an array of step timings and an array of note encodings.
    """
    with open(chart_path, "r") as f:
        f.readline()  # TITLE
        _ = float(f.readline().removeprefix("BPM").strip())  # BPM
        f.readline()  # NOTES
        difficulty_level = f.readline().strip().lower().split(" ")[1]
        _ = _DIFFICULTY_MAP.get(difficulty_level, 2)
        times = []
        cols = []
        for line in f:
            if line.startswith("DIFFICULTY"):
                # TODO: Read off of multiple difficulties
                break
            # TODO: Use the type of note played and not just the presence
            arrows, timing = line.strip().split(" ")
            times.append(float(timing))
            if binary_timings:
                cols.append(0)
            else:
                cols.append(_base4_to_int(arrows))

    return np.array(times), np.array(cols, dtype=np.int32)
