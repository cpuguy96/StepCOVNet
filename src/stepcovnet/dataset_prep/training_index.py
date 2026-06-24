"""Build and load ``training_index.json`` train/val manifests (P8)."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import random
from typing import Literal

from stepcovnet.dataset_prep import constants, training_loader

SplitName = Literal["train", "val"]

SPLIT_TRAIN: SplitName = "train"
SPLIT_VAL: SplitName = "val"
SPLIT_POLICY_STRATIFIED_SONG_V1 = "stratified_song_v1"
TRAINING_INDEX_FILENAME = "training_index.json"


@dataclasses.dataclass
class TrainingIndexEntry:
    """One row in ``training_index.json`` — a single chart block with split label.

    Attributes:
        split: ``train`` or ``val``.
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.
        chart_index: Index into ``charts[]`` inside the JSON file.
        output_relpath: ``{bundle}/{id}`` relative to the preprocess output root.
        difficulty: Lowercase difficulty label for this chart block.
        meter: Raw ``#METER`` for this chart block.
        num_steps: Encoded step count for this chart block.
        audio_relpath: Audio path relative to ``output_dir``.
        chart_relpath: ``.chart.json`` path relative to ``output_dir``.
    """

    split: SplitName
    normalized_bundle: str
    normalized_id: str
    chart_index: int
    output_relpath: str
    difficulty: str
    meter: int
    num_steps: int
    audio_relpath: str
    chart_relpath: str


@dataclasses.dataclass
class TrainingIndexCounts:
    """Aggregate song and chart-row counts per split."""

    songs: dict[SplitName, int]
    rows: dict[SplitName, int]


@dataclasses.dataclass
class TrainingIndex:
    """Flat training manifest for ``final_data`` with train/val assignment."""

    schema_version: int
    output_dir: str
    split_policy: str
    split_seed: int
    val_fraction: float
    created_at: str
    counts: TrainingIndexCounts
    entries: list[TrainingIndexEntry]


def training_index_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return the canonical path to ``training_index.json`` under ``output_dir``."""
    return pathlib.Path(output_dir) / TRAINING_INDEX_FILENAME


def song_key(normalized_bundle: str, normalized_id: str) -> str:
    """Stable song identifier for split assignment."""
    return f"{normalized_bundle}/{normalized_id}"


def _bundle_rng(seed: int, normalized_bundle: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{normalized_bundle}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def assign_stratified_song_splits(
    songs_by_bundle: dict[str, list[str]],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, SplitName]:
    """Assign each song to train or val, stratified within bundle.

    Args:
        songs_by_bundle: ``bundle -> sorted unique song ids``.
        val_fraction: Fraction of songs per bundle assigned to validation.
        seed: Global reproducibility seed (combined per bundle for shuffling).

    Returns:
        Map ``song_key`` → ``train`` | ``val``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")

    assignments: dict[str, SplitName] = {}
    for bundle, song_ids in sorted(songs_by_bundle.items()):
        ordered = sorted(song_ids)
        n_val = int(len(ordered) * val_fraction)
        shuffled = ordered.copy()
        _bundle_rng(seed, bundle).shuffle(shuffled)
        val_songs = set(shuffled[:n_val])
        for song_id in ordered:
            key = song_key(bundle, song_id)
            assignments[key] = SPLIT_VAL if song_id in val_songs else SPLIT_TRAIN
    return assignments


def _relpath_under(output_dir: pathlib.Path, absolute_path: str) -> str:
    """Return ``absolute_path`` relative to resolved ``output_dir``."""
    root = output_dir.resolve()
    resolved = pathlib.Path(absolute_path).resolve()
    return resolved.relative_to(root).as_posix()


def _row_to_entry(
    row: training_loader.TrainingChartRow,
    split: SplitName,
    output_dir: pathlib.Path,
) -> TrainingIndexEntry:
    audio_rel = _relpath_under(output_dir, row.audio_path)
    chart_rel = _relpath_under(output_dir, row.chart_json_path)
    return TrainingIndexEntry(
        split=split,
        normalized_bundle=row.normalized_bundle,
        normalized_id=row.normalized_id,
        chart_index=row.chart_index,
        output_relpath=row.output_relpath,
        difficulty=row.difficulty,
        meter=row.meter,
        num_steps=row.num_steps,
        audio_relpath=audio_rel,
        chart_relpath=chart_rel,
    )


def _counts_from_entries(
    entries: list[TrainingIndexEntry],
) -> TrainingIndexCounts:
    songs: dict[SplitName, set[str]] = {SPLIT_TRAIN: set(), SPLIT_VAL: set()}
    rows: dict[SplitName, int] = {SPLIT_TRAIN: 0, SPLIT_VAL: 0}
    for entry in entries:
        rows[entry.split] += 1
        songs[entry.split].add(song_key(entry.normalized_bundle, entry.normalized_id))
    return TrainingIndexCounts(
        songs={SPLIT_TRAIN: len(songs[SPLIT_TRAIN]), SPLIT_VAL: len(songs[SPLIT_VAL])},
        rows=rows,
    )


def build_training_index(
    output_dir: str | os.PathLike[str],
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    only_exported: bool = True,
) -> TrainingIndex:
    """Build a stratified song-level train/val manifest from prepared output.

    Args:
        output_dir: Preprocess output root (e.g. ``data/final_data``).
        val_fraction: Fraction of songs per bundle assigned to validation.
        seed: Reproducibility seed for per-bundle shuffles.
        only_exported: When True, only include exported packs from ``name_map.json``.

    Returns:
        In-memory manifest ready to validate and save.

    Raises:
        ValueError: When no training rows are discovered.
    """
    root = pathlib.Path(output_dir)
    rows = training_loader.discover_training_rows(
        str(root.resolve()), only_exported=only_exported
    )
    if not rows:
        raise ValueError(f"No training rows found under {root}")

    songs_by_bundle: dict[str, set[str]] = {}
    for row in rows:
        songs_by_bundle.setdefault(row.normalized_bundle, set()).add(row.normalized_id)
    song_splits = assign_stratified_song_splits(
        {bundle: sorted(ids) for bundle, ids in songs_by_bundle.items()},
        val_fraction=val_fraction,
        seed=seed,
    )

    entries = [
        _row_to_entry(
            row,
            song_splits[song_key(row.normalized_bundle, row.normalized_id)],
            root,
        )
        for row in rows
    ]
    entries.sort(
        key=lambda entry: (
            entry.split,
            entry.normalized_bundle,
            entry.normalized_id,
            entry.chart_index,
        )
    )
    index = TrainingIndex(
        schema_version=constants.SCHEMA_VERSION,
        output_dir=str(root.resolve()),
        split_policy=SPLIT_POLICY_STRATIFIED_SONG_V1,
        split_seed=seed,
        val_fraction=val_fraction,
        created_at=datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        counts=_counts_from_entries(entries),
        entries=entries,
    )
    errors = validate_training_index(index)
    if errors:
        raise ValueError("training index validation failed: " + "; ".join(errors))
    return index


def validate_training_index(index: TrainingIndex) -> list[str]:
    """Return human-readable validation errors; empty when valid."""
    errors: list[str] = []
    if index.schema_version != constants.SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {index.schema_version}; "
            f"expected {constants.SCHEMA_VERSION}"
        )
    if index.split_policy != SPLIT_POLICY_STRATIFIED_SONG_V1:
        errors.append(f"unsupported split_policy {index.split_policy!r}")

    song_splits: dict[str, set[SplitName]] = {}
    train_audio: set[str] = set()
    val_audio: set[str] = set()

    for entry in index.entries:
        key = song_key(entry.normalized_bundle, entry.normalized_id)
        song_splits.setdefault(key, set()).add(entry.split)
        if entry.split == SPLIT_TRAIN:
            train_audio.add(entry.audio_relpath)
        else:
            val_audio.add(entry.audio_relpath)

    for key, splits in sorted(song_splits.items()):
        if len(splits) != 1:
            errors.append(f"song {key} has mixed splits: {sorted(splits)}")

    overlap = train_audio & val_audio
    if overlap:
        errors.append(f"audio paths in both splits: {sorted(overlap)[:3]}")

    if not index.entries:
        errors.append("entries is empty")

    return errors


def _index_to_dict(index: TrainingIndex) -> dict:
    return {
        "schema_version": index.schema_version,
        "output_dir": index.output_dir,
        "split_policy": index.split_policy,
        "split_seed": index.split_seed,
        "val_fraction": index.val_fraction,
        "created_at": index.created_at,
        "counts": {
            "songs": dict(index.counts.songs),
            "rows": dict(index.counts.rows),
        },
        "entries": [dataclasses.asdict(entry) for entry in index.entries],
    }


def _index_from_dict(data: dict) -> TrainingIndex:
    version = data.get("schema_version")
    if version is None:
        raise ValueError("missing schema_version in training_index.json")
    if version != constants.SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version}; expected {constants.SCHEMA_VERSION}"
        )
    counts_raw = data.get("counts") or {}
    songs_raw = counts_raw.get("songs") or {}
    rows_raw = counts_raw.get("rows") or {}
    counts = TrainingIndexCounts(
        songs={
            SPLIT_TRAIN: int(songs_raw.get(SPLIT_TRAIN, 0)),
            SPLIT_VAL: int(songs_raw.get(SPLIT_VAL, 0)),
        },
        rows={
            SPLIT_TRAIN: int(rows_raw.get(SPLIT_TRAIN, 0)),
            SPLIT_VAL: int(rows_raw.get(SPLIT_VAL, 0)),
        },
    )
    entries = [TrainingIndexEntry(**item) for item in data.get("entries") or []]
    return TrainingIndex(
        schema_version=version,
        output_dir=str(data.get("output_dir", "")),
        split_policy=str(data.get("split_policy", "")),
        split_seed=int(data.get("split_seed", 0)),
        val_fraction=float(data.get("val_fraction", 0.0)),
        created_at=str(data.get("created_at", "")),
        counts=counts,
        entries=entries,
    )


def save_training_index(
    index: TrainingIndex,
    path: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Write manifest JSON to ``path`` or ``{output_dir}/training_index.json``."""
    target = (
        pathlib.Path(path)
        if path is not None
        else training_index_path(index.output_dir)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_index_to_dict(index), handle, indent=2)
        handle.write("\n")
    return target


def load_training_index(path: str | os.PathLike[str]) -> TrainingIndex:
    """Load and parse ``training_index.json``."""
    target = pathlib.Path(path)
    with target.open(encoding="utf-8") as handle:
        data = json.load(handle)
    index = _index_from_dict(data)
    errors = validate_training_index(index)
    if errors:
        raise ValueError(f"invalid training index {target}: " + "; ".join(errors))
    return index


def manifest_split_enabled(
    train_dir: str | os.PathLike[str],
    val_dir: str | os.PathLike[str],
) -> bool:
    """True when train and val share one root that has ``training_index.json``."""
    train_root = pathlib.Path(train_dir).resolve()
    val_root = pathlib.Path(val_dir).resolve()
    if train_root != val_root:
        return False
    return training_index_path(train_root).is_file()


def rows_for_split(
    output_dir: str | os.PathLike[str],
    split: SplitName,
) -> list[training_loader.TrainingChartRow]:
    """Load chart rows for one split from ``training_index.json``."""
    root = pathlib.Path(output_dir)
    index = load_training_index(training_index_path(root))
    rows: list[training_loader.TrainingChartRow] = []
    for entry in index.entries:
        if entry.split != split:
            continue
        audio_path = (root / entry.audio_relpath).resolve()
        chart_path = (root / entry.chart_relpath).resolve()
        rows.append(
            training_loader.TrainingChartRow(
                normalized_bundle=entry.normalized_bundle,
                normalized_id=entry.normalized_id,
                chart_index=entry.chart_index,
                output_relpath=entry.output_relpath,
                chart_json_path=str(chart_path),
                audio_path=str(audio_path),
                difficulty=entry.difficulty,
                meter=entry.meter,
                num_steps=entry.num_steps,
            )
        )
    rows.sort(
        key=lambda row: (row.normalized_bundle, row.normalized_id, row.chart_index)
    )
    return rows
