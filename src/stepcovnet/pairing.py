"""Audio/chart file pairing without TensorFlow dependencies."""

import os
import pathlib
from typing import Literal

from stepcovnet.dataset_prep import training_index, training_loader

SplitName = Literal["train", "val"]


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


def list_training_samples(
    data_dir: str,
    split: SplitName | None = None,
) -> list[tuple[str, str, int]]:
    """Return training samples as ``(audio_path, chart_path, chart_index)``.

    When ``data_dir`` contains a prepared ``final_data`` layout (``name_map.json``
    or nested ``.chart.json`` files), one row is returned per chart block inside
    each song JSON. When ``split`` is ``train`` or ``val`` and
    ``training_index.json`` exists, only manifest rows for that split are returned.
    Otherwise falls back to legacy ``.txt`` pairs with ``chart_index`` 0.

    Args:
        data_dir: Training data root (``data/v2/train``, ``data/final_data``, …).
        split: Optional ``train`` or ``val`` filter when a training index exists.

    Returns:
        Sorted sample refs for dataloaders.

    Raises:
        ValueError: When ``split`` is set but ``training_index.json`` is missing.
    """
    if split is not None:
        index_path = training_index.training_index_path(data_dir)
        if not index_path.is_file():
            raise ValueError(f"split={split!r} requires {index_path}")
        rows = training_index.rows_for_split(data_dir, split)
        return [(row.audio_path, row.chart_json_path, row.chart_index) for row in rows]

    rows = training_loader.discover_training_rows(data_dir)
    if rows:
        return [(row.audio_path, row.chart_json_path, row.chart_index) for row in rows]
    return [
        (audio_path, chart_path, 0)
        for audio_path, chart_path in list_audio_chart_pairs(data_dir)
    ]
