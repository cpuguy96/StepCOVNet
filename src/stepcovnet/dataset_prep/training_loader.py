"""Discover and load ``final_data`` chart rows for training (P9)."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

import numpy as np

from stepcovnet.dataset_prep import constants, models, normalize, pack_results

_SKIP_DIR_NAMES = frozenset({"_staging", "__pycache__"})


@dataclasses.dataclass(frozen=True)
class TrainingChartRow:
    """One training sample: audio plus a single chart block from ``.chart.json``.

    Attributes:
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.
        chart_index: Index into ``charts[]`` inside the JSON file.
        output_relpath: ``{bundle}/{id}`` relative to the preprocess output root.
        chart_json_path: Absolute path to ``{id}.chart.json``.
        audio_path: Absolute path to copied audio in the song directory.
        difficulty: Lowercase difficulty label for this chart block.
        meter: Raw ``#METER`` for this chart block.
        num_steps: Encoded step count for this chart block.
    """

    normalized_bundle: str
    normalized_id: str
    chart_index: int
    output_relpath: str
    chart_json_path: str
    audio_path: str
    difficulty: str
    meter: int
    num_steps: int


def _load_pack_from_chart_json(
    chart_path: str | os.PathLike[str],
) -> models.ParsedSongPack:
    """Load and validate one ``.chart.json`` file as a parsed song pack.

    Args:
        chart_path: Absolute path to a chart JSON file.

    Returns:
        Parsed song pack for the on-disk JSON object.

    Raises:
        ValueError: When ``schema_version`` is missing or unsupported.
    """
    path = pathlib.Path(chart_path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    version = data.get("schema_version")
    if version is None:
        raise ValueError(f"missing schema_version in {path}")
    if version != constants.SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version} in {path}; "
            f"expected {constants.SCHEMA_VERSION}"
        )
    return models.ParsedSongPack.from_dict(data)


def _row_from_pack(
    pack: models.ParsedSongPack,
    chart_index: int,
    output_dir: pathlib.Path,
) -> TrainingChartRow:
    """Build one training row from a parsed pack and chart index.

    Args:
        pack: Parsed song pack loaded from disk.
        chart_index: Index into ``pack.charts``.
        output_dir: Preprocess output root for resolving paths.

    Returns:
        Training row with absolute audio and chart JSON paths.
    """
    chart = pack.charts[chart_index]
    song_dir = output_dir / pack.normalized_bundle / pack.normalized_id
    chart_json = song_dir / f"{pack.normalized_id}.chart.json"
    audio = song_dir / pack.audio_filename
    return TrainingChartRow(
        normalized_bundle=pack.normalized_bundle,
        normalized_id=pack.normalized_id,
        chart_index=chart_index,
        output_relpath=f"{pack.normalized_bundle}/{pack.normalized_id}",
        chart_json_path=str(chart_json.resolve()),
        audio_path=str(audio.resolve()),
        difficulty=chart.summary.difficulty,
        meter=chart.summary.meter,
        num_steps=chart.summary.num_steps,
    )


def _discover_from_name_map(
    output_dir: pathlib.Path,
    *,
    only_exported: bool,
) -> list[TrainingChartRow]:
    """Discover training rows via ``name_map.json`` exported entries.

    Args:
        output_dir: Preprocess output root containing ``name_map.json``.
        only_exported: When True, skip non-exported name-map rows.

    Returns:
        Sorted rows for every chart block in selected packs.
    """
    name_map = normalize.load_name_map(normalize.name_map_path(output_dir))
    rows: list[TrainingChartRow] = []
    for entry in name_map.entries:
        if only_exported and entry.result != pack_results.PACK_RESULT_EXPORTED:
            continue
        pack = models.load_parsed_song(
            output_dir,
            entry.normalized_bundle,
            entry.normalized_id,
        )
        for chart_index in range(len(pack.charts)):
            rows.append(_row_from_pack(pack, chart_index, output_dir))
    rows.sort(
        key=lambda row: (row.normalized_bundle, row.normalized_id, row.chart_index)
    )
    return rows


def _discover_from_filesystem(output_dir: pathlib.Path) -> list[TrainingChartRow]:
    """Discover training rows by scanning nested ``*.chart.json`` files.

    Args:
        output_dir: Preprocess output root to scan recursively.

    Returns:
        Sorted rows for every chart block found on disk.
    """
    rows: list[TrainingChartRow] = []
    for chart_path in sorted(output_dir.rglob("*.chart.json")):
        if any(part in _SKIP_DIR_NAMES for part in chart_path.parts):
            continue
        try:
            rel = chart_path.relative_to(output_dir)
        except ValueError:
            continue
        if len(rel.parts) != 3:
            continue
        pack = _load_pack_from_chart_json(chart_path)
        for chart_index in range(len(pack.charts)):
            rows.append(_row_from_pack(pack, chart_index, output_dir))
    rows.sort(
        key=lambda row: (row.normalized_bundle, row.normalized_id, row.chart_index)
    )
    return rows


def discover_training_rows(
    output_dir: str | os.PathLike[str],
    *,
    only_exported: bool = True,
) -> list[TrainingChartRow]:
    """List one row per ``(bundle, song, chart_index)`` under a preprocess output root.

    When ``name_map.json`` exists, exported packs are loaded from the manifest.
    Otherwise, nested ``*.chart.json`` files are scanned directly.

    Args:
        output_dir: Preprocess output root (e.g. ``data/final_data``).
        only_exported: When True, skip non-exported name-map rows.

    Returns:
        Sorted training rows. Empty when no prepared layout is found.
    """
    root = pathlib.Path(output_dir)
    if not root.is_dir():
        return []

    name_map_path = normalize.name_map_path(root)
    if name_map_path.is_file():
        return _discover_from_name_map(root, only_exported=only_exported)

    if not any(root.rglob("*.chart.json")):
        return []
    return _discover_from_filesystem(root)


def filter_rows_by_step_cap(
    rows: list[TrainingChartRow],
    *,
    max_steps: int,
) -> list[TrainingChartRow]:
    """Keep rows whose ``num_steps`` is at most ``max_steps``.

    Args:
        rows: Candidate training rows.
        max_steps: Maximum encoded step count (inclusive).

    Returns:
        Filtered rows preserving input order among kept entries.
    """
    return [row for row in rows if row.num_steps <= max_steps]


def load_chart_meter(
    chart_path: str | os.PathLike[str],
    chart_index: int,
) -> int:
    """Return raw ``#METER`` for one chart block inside ``.chart.json``.

    Args:
        chart_path: Path to a ``.chart.json`` file.
        chart_index: Index into ``charts[]`` inside the JSON file.

    Returns:
        Meter integer, or ``0`` when the path is not a chart JSON file.

    Raises:
        IndexError: When ``chart_index`` is out of range for the file.
    """
    path = pathlib.Path(chart_path)
    if path.suffix != ".json" and not str(path).endswith(".chart.json"):
        return 0
    pack = _load_pack_from_chart_json(chart_path)
    if chart_index < 0 or chart_index >= len(pack.charts):
        raise IndexError(
            f"chart_index {chart_index} out of range for {chart_path} "
            f"({len(pack.charts)} charts)"
        )
    return int(pack.charts[chart_index].summary.meter)


def load_chart_times_sec(
    chart_path: str | os.PathLike[str],
    chart_index: int,
) -> np.ndarray:
    """Return sorted onset times for one chart block inside ``.chart.json``.

    Args:
        chart_path: Path to a ``.chart.json`` file.
        chart_index: Index into ``charts[]`` inside the JSON file.

    Returns:
        Sorted onset times in seconds as ``float64``.

    Raises:
        IndexError: When ``chart_index`` is out of range for the file.
    """
    pack = _load_pack_from_chart_json(chart_path)
    if chart_index < 0 or chart_index >= len(pack.charts):
        raise IndexError(
            f"chart_index {chart_index} out of range for {chart_path} "
            f"({len(pack.charts)} charts)"
        )
    times = np.asarray(pack.charts[chart_index].times_sec, dtype=np.float64)
    return np.sort(times)


def load_chart_column_codes(
    chart_path: str | os.PathLike[str],
    chart_index: int,
    *,
    binary_timings: bool = False,
) -> np.ndarray:
    """Return per-step column codes for one chart block inside ``.chart.json``.

    Args:
        chart_path: Path to a ``.chart.json`` file.
        chart_index: Index into ``charts[]`` inside the JSON file.
        binary_timings: When True, return zeros (timing-only labels).

    Returns:
        Column codes as ``int32``; zeros when ``binary_timings`` is True.

    Raises:
        IndexError: When ``chart_index`` is out of range for the file.
    """
    pack = _load_pack_from_chart_json(chart_path)
    if chart_index < 0 or chart_index >= len(pack.charts):
        raise IndexError(
            f"chart_index {chart_index} out of range for {chart_path} "
            f"({len(pack.charts)} charts)"
        )
    chart = pack.charts[chart_index]
    if binary_timings:
        return np.zeros(len(chart.times_sec), dtype=np.int32)
    return np.asarray(chart.column_codes, dtype=np.int32)


def load_chart_bpm(chart_path: str | os.PathLike[str]) -> float:
    """Return ``metadata.initial_bpm`` from a ``.chart.json`` file.

    Args:
        chart_path: Path to a ``.chart.json`` file.

    Returns:
        Initial BPM from the first ``#BPMS`` segment in the simfile metadata.
    """
    pack = _load_pack_from_chart_json(chart_path)
    return float(pack.metadata.initial_bpm)
