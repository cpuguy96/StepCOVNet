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
SUBSET_POLICY_LADDER_V1 = "ladder_v1"
STANDARD_POLICY_TAG = "standard_v1"
TRAINING_INDEX_FILENAME = "training_index.json"
STANDARD_INDEX_FILENAME = "training_index_standard.json"


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
    """Aggregate song and chart-row counts per split.

    Attributes:
        songs: Song counts keyed by ``train`` or ``val``.
        rows: Chart-row counts keyed by ``train`` or ``val``.
    """

    songs: dict[SplitName, int]
    rows: dict[SplitName, int]


@dataclasses.dataclass
class TrainingIndex:
    """Flat training manifest for ``final_data`` with train/val assignment.

    Attributes:
        schema_version: Manifest layout version.
        output_dir: Preprocess output root used to build the manifest.
        split_policy: Split algorithm identifier (``stratified_song_v1``).
        split_seed: Reproducibility seed for per-bundle shuffles.
        val_fraction: Fraction of songs per bundle assigned to validation.
        created_at: ISO-8601 UTC timestamp when the manifest was built.
        counts: Aggregate song and row counts per split.
        entries: One row per chart block with split label and relative paths.
        source_sha256: Hex digest of the manifest a subset was sampled from;
            empty for manifests built directly from a preprocess output root.
    """

    schema_version: int
    output_dir: str
    split_policy: str
    split_seed: int
    val_fraction: float
    created_at: str
    counts: TrainingIndexCounts
    entries: list[TrainingIndexEntry]
    source_sha256: str = ""


def training_index_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return ``training_index.json`` path under ``output_dir``.

    Args:
        output_dir: Preprocess output root.

    Returns:
        Path to ``{output_dir}/training_index.json``.
    """
    return pathlib.Path(output_dir) / TRAINING_INDEX_FILENAME


def is_training_index_file(path: str | os.PathLike[str]) -> bool:
    """Return True when ``path`` points at a training manifest JSON file.

    Args:
        path: Candidate manifest file or directory reference.

    Returns:
        True for an existing ``training_index.json`` or ``.json`` manifest file.
    """
    target = pathlib.Path(path)
    return target.is_file() and (
        target.name == TRAINING_INDEX_FILENAME or target.suffix.lower() == ".json"
    )


def resolve_output_dir(
    index: TrainingIndex,
    index_path: str | os.PathLike[str],
) -> pathlib.Path:
    """Resolve the prepared data root for a loaded manifest.

    Prefers ``index.output_dir`` when that directory exists. Otherwise falls back
    to the directory containing ``index_path`` when it looks like a prep output root.

    Args:
        index: Loaded training manifest.
        index_path: Path used to load ``index`` (for sibling-root fallback).

    Returns:
        Absolute preprocess output root for chart and audio paths.

    Raises:
        ValueError: When neither stored nor sibling output roots exist.
    """
    stored = pathlib.Path(index.output_dir)
    if stored.is_dir():
        return stored.resolve()

    sibling_root = pathlib.Path(index_path).resolve().parent
    if (sibling_root / "name_map.json").is_file() or any(
        sibling_root.rglob("*.chart.json")
    ):
        return sibling_root

    raise ValueError(
        f"cannot resolve data root for training index {index_path}: "
        f"output_dir {index.output_dir!r} is missing"
    )


def locate_training_index(
    data_ref: str | os.PathLike[str],
) -> tuple[pathlib.Path | None, pathlib.Path]:
    """Interpret ``data_ref`` as a manifest file path or prepared output root.

    Args:
        data_ref: ``training_index.json`` path or preprocess output directory.

    Returns:
        ``(index_path, data_root)`` when a manifest is found; otherwise
        ``(None, data_ref)`` resolved as a directory path.
    """
    ref = pathlib.Path(data_ref)
    if is_training_index_file(ref):
        index = load_training_index(ref)
        return ref.resolve(), resolve_output_dir(index, ref)

    if ref.is_dir():
        candidate = training_index_path(ref)
        if candidate.is_file():
            index = load_training_index(candidate)
            return candidate.resolve(), resolve_output_dir(index, candidate)

    return None, ref.resolve()


def rows_from_index(
    index: TrainingIndex,
    output_dir: pathlib.Path,
    split: SplitName | None = None,
) -> list[training_loader.TrainingChartRow]:
    """Materialize chart rows from a loaded manifest.

    Args:
        index: Loaded training manifest.
        output_dir: Preprocess output root for resolving relative paths.
        split: When set, keep only rows assigned to this split.

    Returns:
        Sorted chart rows with absolute audio and chart JSON paths.
    """
    root = output_dir.resolve()
    rows: list[training_loader.TrainingChartRow] = []
    for entry in index.entries:
        if split is not None and entry.split != split:
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


def song_key(normalized_bundle: str, normalized_id: str) -> str:
    """Return stable song identifier for split assignment.

    Args:
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.

    Returns:
        ``{normalized_bundle}/{normalized_id}`` string.
    """
    return f"{normalized_bundle}/{normalized_id}"


def _bundle_rng(seed: int, normalized_bundle: str) -> random.Random:
    """Return a deterministic RNG for one bundle's validation shuffle.

    Args:
        seed: Global split seed from the manifest.
        normalized_bundle: Bundle slug combined with ``seed`` for hashing.

    Returns:
        Seeded ``random.Random`` instance unique to the bundle.
    """
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

    Raises:
        ValueError: When ``val_fraction`` is outside ``[0, 1)``.
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
    """Return ``absolute_path`` relative to resolved ``output_dir``.

    Args:
        output_dir: Preprocess output root.
        absolute_path: Absolute audio or chart path to relativize.

    Returns:
        POSIX-style path relative to ``output_dir``.
    """
    root = output_dir.resolve()
    resolved = pathlib.Path(absolute_path).resolve()
    return resolved.relative_to(root).as_posix()


def _row_to_entry(
    row: training_loader.TrainingChartRow,
    split: SplitName,
    output_dir: pathlib.Path,
) -> TrainingIndexEntry:
    """Convert a discovered chart row into one manifest entry.

    Args:
        row: Training row from discovery.
        split: Assigned ``train`` or ``val`` label.
        output_dir: Preprocess output root for relative path fields.

    Returns:
        Manifest entry with audio and chart paths relative to ``output_dir``.
    """
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
    """Aggregate per-split song and row counts from manifest entries.

    Args:
        entries: Flat manifest rows with split labels.

    Returns:
        Song and row totals keyed by ``train`` and ``val``.
    """
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
    """Return human-readable validation errors; empty when valid.

    Args:
        index: Manifest to validate in memory.

    Returns:
        Validation error messages; empty list when ``index`` is valid.
    """
    errors: list[str] = []
    if index.schema_version != constants.SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {index.schema_version}; "
            f"expected {constants.SCHEMA_VERSION}"
        )
    if index.split_policy != SPLIT_POLICY_STRATIFIED_SONG_V1 and not (
        index.split_policy.startswith(f"{SPLIT_POLICY_STRATIFIED_SONG_V1}+")
    ):
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
    """Serialize a training manifest to a JSON-compatible mapping.

    Args:
        index: In-memory manifest.

    Returns:
        Dictionary suitable for ``json.dump``.
    """
    payload = {
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
    if index.source_sha256:
        payload["source_sha256"] = index.source_sha256
    return payload


def _index_from_dict(data: dict) -> TrainingIndex:
    """Parse a training manifest from a JSON-compatible mapping.

    Args:
        data: Root object loaded from ``training_index.json``.

    Returns:
        In-memory manifest without post-load validation.

    Raises:
        ValueError: When ``schema_version`` is missing or unsupported.
    """
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
        source_sha256=str(data.get("source_sha256", "")),
    )


def save_training_index(
    index: TrainingIndex,
    path: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Write manifest JSON to ``path`` or ``{output_dir}/training_index.json``.

    Args:
        index: Manifest to serialize.
        path: Optional explicit output path; defaults to ``index.output_dir``.

    Returns:
        Path where the manifest was written.
    """
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
    """Load and parse ``training_index.json``.

    Args:
        path: Path to a training manifest JSON file.

    Returns:
        Validated in-memory manifest.

    Raises:
        ValueError: When schema version is unsupported or validation fails.
    """
    target = pathlib.Path(path)
    with target.open(encoding="utf-8") as handle:
        data = json.load(handle)
    index = _index_from_dict(data)
    errors = validate_training_index(index)
    if errors:
        raise ValueError(f"invalid training index {target}: " + "; ".join(errors))
    return index


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the hex SHA-256 digest of a file.

    Args:
        path: File to digest.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_sort_key(entry: TrainingIndexEntry) -> tuple[str, str, str, int]:
    """Return the canonical ordering key for a manifest row.

    Args:
        entry: Row to order.

    Returns:
        Split, bundle, song id, and chart index.
    """
    return (
        entry.split,
        entry.normalized_bundle,
        entry.normalized_id,
        entry.chart_index,
    )


def _nested_sample(
    pool: list[TrainingIndexEntry],
    count: int,
    *,
    seed: int,
    label: str,
) -> list[TrainingIndexEntry]:
    """Return ``count`` rows drawn in a size-independent, nesting order.

    The pool is sorted, then shuffled once by a generator seeded only from
    ``seed`` and ``label``. Taking a prefix means a larger ``count`` is a
    superset of a smaller one, and a draw for one split cannot be perturbed by
    the size of another split's draw.

    Args:
        pool: Candidate rows for one split.
        count: Number of rows to take.
        seed: Sampling seed shared by the whole manifest.
        label: Split name mixed into the generator seed.

    Returns:
        Selected rows, in shuffled order.
    """
    shuffled = sorted(pool, key=_entry_sort_key)
    random.Random(f"{seed}:{label}").shuffle(shuffled)
    return shuffled[:count]


def build_training_index_subset(
    source_path: str | os.PathLike[str],
    *,
    train_rows: int,
    val_rows: int,
    seed: int = 42,
    policy_tag: str = SUBSET_POLICY_LADDER_V1,
) -> TrainingIndex:
    """Sample fixed train/val row counts from an existing P8 manifest.

    Draws are independent per split and nest as counts grow, so a subset built
    with a larger ``train_rows`` keeps the same val rows and contains the
    smaller subset's train rows. Protocol: ``docs/research/AR_SCALING_LADDER.md``.

    Args:
        source_path: Path to a full ``training_index.json``.
        train_rows: Number of train chart rows to keep.
        val_rows: Number of val chart rows to keep.
        seed: Shuffle seed for reproducible sampling.
        policy_tag: Suffix recorded in ``split_policy`` for traceability.

    Returns:
        New manifest sharing ``output_dir`` with the source index.

    Raises:
        ValueError: When requested counts exceed available rows or validation fails.
    """
    if train_rows < 1 or val_rows < 1:
        raise ValueError("train_rows and val_rows must be at least 1")
    source = load_training_index(source_path)
    train_pool = [entry for entry in source.entries if entry.split == SPLIT_TRAIN]
    val_pool = [entry for entry in source.entries if entry.split == SPLIT_VAL]
    if train_rows > len(train_pool):
        raise ValueError(
            f"train_rows={train_rows} exceeds available train rows ({len(train_pool)})",
        )
    if val_rows > len(val_pool):
        raise ValueError(
            f"val_rows={val_rows} exceeds available val rows ({len(val_pool)})",
        )
    sampled = _nested_sample(
        train_pool,
        train_rows,
        seed=seed,
        label=SPLIT_TRAIN,
    ) + _nested_sample(val_pool, val_rows, seed=seed, label=SPLIT_VAL)
    sampled.sort(key=_entry_sort_key)
    subset = TrainingIndex(
        schema_version=source.schema_version,
        output_dir=source.output_dir,
        split_policy=f"{source.split_policy}+{policy_tag}",
        split_seed=seed,
        val_fraction=val_rows / (train_rows + val_rows),
        created_at=datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        counts=_counts_from_entries(sampled),
        entries=sampled,
        source_sha256=file_sha256(source_path),
    )
    errors = validate_training_index(subset)
    if errors:
        raise ValueError(
            f"invalid training index subset from {source_path}: " + "; ".join(errors),
        )
    return subset


def standard_index_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return ``{output_dir}/training_index_standard.json``.

    Args:
        output_dir: Preprocess output root that holds the full manifest.

    Returns:
        Path to the standard-difficulty (no ``edit``) training manifest.
    """
    return pathlib.Path(output_dir) / STANDARD_INDEX_FILENAME


def filter_training_index(
    source_path: str | os.PathLike[str],
    *,
    keep_difficulties: frozenset[str] | set[str] | None = None,
    policy_tag: str = STANDARD_POLICY_TAG,
) -> TrainingIndex:
    """Keep chart rows whose difficulty is in ``keep_difficulties``.

    Song-level train/val assignments are unchanged. Default keeps the five
    StepMania standard labels (drops ``edit`` and other custom charts) so Dataset
    A matches the DDC 450-chart table (`donahue2017ddc`).

    Args:
        source_path: Path to an existing ``training_index.json``.
        keep_difficulties: Lowercase labels to keep. Defaults to
            ``dataset_prep.constants.STANDARD_DIFFICULTIES``.
        policy_tag: Suffix appended to ``split_policy`` for traceability.

    Returns:
        New manifest sharing ``output_dir`` with the source index.

    Raises:
        ValueError: When ``policy_tag`` is empty, no rows remain, or validation
            fails.
    """
    tag = str(policy_tag).strip()
    if not tag:
        raise ValueError("policy_tag must be a non-empty string")
    allowed = {
        item.lower()
        for item in (
            keep_difficulties
            if keep_difficulties is not None
            else constants.STANDARD_DIFFICULTIES
        )
    }
    if not allowed:
        raise ValueError("keep_difficulties must contain at least one label")
    source = load_training_index(source_path)
    kept = [entry for entry in source.entries if entry.difficulty.lower() in allowed]
    if not kept:
        raise ValueError(
            f"no chart rows left after filtering difficulties {sorted(allowed)}"
        )
    kept.sort(key=_entry_sort_key)
    filtered = TrainingIndex(
        schema_version=source.schema_version,
        output_dir=source.output_dir,
        split_policy=f"{source.split_policy}+{tag}",
        split_seed=source.split_seed,
        val_fraction=source.val_fraction,
        created_at=datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        counts=_counts_from_entries(kept),
        entries=kept,
        source_sha256=file_sha256(source_path),
    )
    errors = validate_training_index(filtered)
    if errors:
        raise ValueError(
            f"invalid filtered training index from {source_path}: " + "; ".join(errors),
        )
    return filtered


def unique_audio_relpaths(entries: list[TrainingIndexEntry]) -> set[str]:
    """Return the set of ``audio_relpath`` values referenced by manifest rows.

    Args:
        entries: Manifest rows.

    Returns:
        Unique audio paths relative to the preprocess output root.
    """
    return {entry.audio_relpath for entry in entries}


def manifest_split_enabled(
    train_dir: str | os.PathLike[str],
    val_dir: str | os.PathLike[str],
) -> bool:
    """Return True when train and val share one root that has ``training_index.json``.

    Args:
        train_dir: Training data directory or manifest path.
        val_dir: Validation data directory or manifest path.

    Returns:
        True when both references resolve to the same indexed output root.
    """
    train_ref = pathlib.Path(train_dir)
    val_ref = pathlib.Path(val_dir)
    if is_training_index_file(train_ref) and is_training_index_file(val_ref):
        return train_ref.resolve() == val_ref.resolve()
    train_root = train_ref.resolve()
    val_root = val_ref.resolve()
    if train_root != val_root:
        return False
    return training_index_path(train_root).is_file()


def rows_for_split(
    data_ref: str | os.PathLike[str],
    split: SplitName,
) -> list[training_loader.TrainingChartRow]:
    """Load chart rows for one split from a manifest file or output root.

    Args:
        data_ref: ``training_index.json`` path or preprocess output directory.
        split: ``train`` or ``val`` split to materialize.

    Returns:
        Sorted chart rows for the requested split.
    """
    index_path, data_root = locate_training_index(data_ref)
    if index_path is None:
        index_path = training_index_path(data_root)
    index = load_training_index(index_path)
    root = resolve_output_dir(index, index_path)
    return rows_from_index(index, root, split=split)
