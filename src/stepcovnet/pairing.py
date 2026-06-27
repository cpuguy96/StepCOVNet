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


def list_unique_audio_paths(
    data_ref: str,
    split: SplitName | None = None,
) -> tuple[list[str], str]:
    """Return deduplicated audio paths and data root for MERT extraction.

    Args:
        data_ref: Manifest file, prepared output root, or legacy training directory.
        split: Optional ``train`` or ``val`` filter when loading from a manifest.

    Returns:
        Sorted unique audio paths and the data root for nested ``.mert.npy`` paths.

    Raises:
        ValueError: When no audio paths are found under ``data_ref``.
    """
    samples = list_training_samples(data_ref, split=split)
    if samples:
        index_path, data_root = training_index.locate_training_index(data_ref)
        if index_path is not None:
            index = training_index.load_training_index(index_path)
            root = str(training_index.resolve_output_dir(index, index_path))
        else:
            root = str(data_root)
        unique = sorted({audio_path for audio_path, _, _ in samples})
        if not unique:
            raise ValueError(f"no audio paths found under {data_ref!r}")
        return unique, root

    pairs = list_audio_chart_pairs(data_ref)
    if not pairs:
        raise ValueError(f"no audio-chart pairs found under {data_ref!r}")
    return sorted({audio_path for audio_path, _ in pairs}), data_ref


def list_training_samples(
    data_ref: str,
    split: SplitName | None = None,
) -> list[tuple[str, str, int]]:
    """Return training samples as ``(audio_path, chart_path, chart_index)``.

    ``data_ref`` may be:

    - A path to ``training_index.json`` (or another manifest ``.json``). Entries
      are resolved via the manifest's ``output_dir`` and relative audio/chart paths.
    - A prepared output root (``final_data``). Uses ``training_index.json`` when
      ``split`` is set, otherwise discovers all chart rows under the tree.
    - A legacy layout root with ``.txt`` charts (``chart_index`` 0).

    Args:
        data_ref: Manifest file, prepared output root, or legacy training directory.
        split: Optional ``train`` or ``val`` filter when loading from a manifest.

    Returns:
        Sorted sample refs for dataloaders.

    Raises:
        ValueError: When ``split`` is set but no manifest can be resolved.
    """
    index_path, data_root = training_index.locate_training_index(data_ref)
    if index_path is not None:
        index = training_index.load_training_index(index_path)
        root = training_index.resolve_output_dir(index, index_path)
        rows = training_index.rows_from_index(index, root, split=split)
        return [(row.audio_path, row.chart_json_path, row.chart_index) for row in rows]

    if split is not None:
        raise ValueError(
            f"split={split!r} requires a training index; "
            f"pass a manifest file or a directory containing "
            f"{training_index.TRAINING_INDEX_FILENAME}"
        )

    rows = training_loader.discover_training_rows(data_ref)
    if rows:
        return [(row.audio_path, row.chart_json_path, row.chart_index) for row in rows]
    return [
        (audio_path, chart_path, 0)
        for audio_path, chart_path in list_audio_chart_pairs(data_ref)
    ]
