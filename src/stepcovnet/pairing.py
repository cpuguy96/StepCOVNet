"""Audio/chart file pairing without TensorFlow dependencies."""

import os
import pathlib

from stepcovnet.dataset_prep import training_loader


def list_audio_chart_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return paired audio and legacy ``.txt`` chart paths under ``data_dir``.

    For ``final_data`` layouts with ``.chart.json``, use
    :func:`list_training_samples` instead.

    Args:
        data_dir: Root directory to search recursively.

    Returns:
        List of ``(audio_path, chart_path)`` tuples with matching filename stems,
        sorted by audio path for stable ordering across platforms.
    """
    pairs: list[tuple[str, str]] = []
    for root, _, files in os.walk(data_dir):
        audio_files = sorted(f for f in files if f.endswith((".mp3", ".ogg", ".wav")))
        chart_files = sorted(f for f in files if f.endswith(".txt"))

        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching_charts = sorted(f for f in chart_files if f.startswith(stem))
            if matching_charts:
                pairs.append(
                    (
                        str(pathlib.Path(root) / audio_file),
                        str(pathlib.Path(root) / matching_charts[0]),
                    )
                )
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def list_training_samples(data_dir: str) -> list[tuple[str, str, int]]:
    """Return training samples as ``(audio_path, chart_path, chart_index)``.

    When ``data_dir`` contains a prepared ``final_data`` layout (``name_map.json``
    or nested ``.chart.json`` files), one row is returned per chart block inside
    each song JSON. Otherwise falls back to legacy ``.txt`` pairs with
    ``chart_index`` 0.

    Args:
        data_dir: Training data root (``data/v2/train``, ``data/final_data``, …).

    Returns:
        Sorted sample refs for dataloaders.
    """
    rows = training_loader.discover_training_rows(data_dir)
    if rows:
        return [(row.audio_path, row.chart_json_path, row.chart_index) for row in rows]
    return [
        (audio_path, chart_path, 0)
        for audio_path, chart_path in list_audio_chart_pairs(data_dir)
    ]
