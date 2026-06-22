"""Audio/chart file pairing without TensorFlow dependencies."""

import os
import pathlib


def list_audio_chart_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return paired audio and chart file paths found under a data directory.

    Args:
        data_dir: Root directory to search recursively.

    Returns:
        List of ``(audio_path, chart_path)`` tuples with matching filename stems.
    """
    pairs: list[tuple[str, str]] = []
    for root, _, files in os.walk(data_dir):
        audio_files = [f for f in files if f.endswith((".mp3", ".ogg", ".wav"))]
        chart_files = [f for f in files if f.endswith(".txt")]

        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching_charts = [f for f in chart_files if f.startswith(stem)]
            if matching_charts:
                pairs.append(
                    (
                        str(pathlib.Path(root) / audio_file),
                        str(pathlib.Path(root) / matching_charts[0]),
                    )
                )
    return pairs
