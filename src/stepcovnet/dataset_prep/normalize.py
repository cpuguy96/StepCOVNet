"""Bundle and song slug normalization."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import pathlib
import re
import unicodedata

from stepcovnet.dataset_prep import (
    config,
    constants,
    discovery,
    manifests,
    pack_results,
    simfile_adapter,
)


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for manifest dataclasses."""

    def as_dict(self) -> dict:
        """Convert manifest object to dictionary for JSON serialization.

        Returns:
            Serializable mapping of dataclass fields.
        """
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create manifest object from dictionary.

        Args:
            data: Serialized field values for the dataclass.

        Returns:
            Instance with fields taken from ``data``.
        """
        return cls(**data)


@dataclasses.dataclass
class NameMapEntry(_DictSerializableMixin):
    """One row in ``name_map.json`` after normalization.

    Attributes:
        normalized_bundle: Output bundle slug.
        normalized_id: Output song slug within the bundle.
        output_relpath: ``{bundle}/{id}`` relative to ``output_dir``.
        source_bundle: Raw bundle folder name.
        source_pack: Pack path relative to preprocess input root.
        source_simfile: Simfile basename from discovery.
        title: Resolved ``#TITLE`` (may be empty).
        artist: Resolved ``#ARTIST`` (may be empty).
        audio_source: Filled after pack processing completes.
        result: Coarse outcome — ``pack_pending``, ``pack_exported``, ``pack_skipped``, or ``pack_error``.
        reason: Detail code when ``result`` is ``skipped`` or ``error``.
        warnings: Normalization and processing warnings.
        charts_exported: Charts written when ``result`` is ``pack_exported``.
        charts_skipped: Per-chart skips recorded during export.
        message: Optional error detail when ``result`` is ``pack_error``.
    """

    normalized_bundle: str
    normalized_id: str
    output_relpath: str
    source_bundle: str
    source_pack: str
    source_simfile: str
    title: str
    artist: str
    audio_source: str
    result: str
    reason: str | None
    warnings: list[str]
    charts_exported: int = 0
    charts_skipped: int = 0
    message: str = ""

    @classmethod
    def from_dict(cls, data: dict):
        """Parse name map entry with defaults for newer optional fields.

        Args:
            data: Serialized name-map row.

        Returns:
            Parsed entry with defaults for optional chart counters.
        """
        payload = dict(data)
        payload.setdefault("charts_exported", 0)
        payload.setdefault("charts_skipped", 0)
        payload.setdefault("message", "")
        return cls(**payload)


@dataclasses.dataclass
class NameMap(_DictSerializableMixin):
    """Output mapping raw packs to normalized slugs.

    Attributes:
        schema_version: Manifest layout version.
        raw_input_root: Root used to resolve ``source_pack`` paths.
        output_dir: Preprocess output root.
        entries: One row per pack to process.
    """

    schema_version: int
    raw_input_root: str
    output_dir: str
    entries: list[NameMapEntry]

    @classmethod
    def from_dict(cls, data: dict):
        """Parse name map including nested entries.

        Args:
            data: Serialized ``name_map.json`` root object.

        Returns:
            Parsed name map with nested entry rows.
        """
        payload = dict(data)
        payload["raw_input_root"] = data.get("raw_input_root") or data.get(
            "input_dir", ""
        )
        payload["entries"] = [
            NameMapEntry.from_dict(item) for item in data.get("entries", [])
        ]
        payload.pop("input_dir", None)
        return cls(**payload)


def slugify(raw: str) -> str:
    """Fold text into a lowercase ASCII slug.

    Args:
        raw: Arbitrary bundle, title, or folder name.

    Returns:
        Slug up to ``MAX_SLUG_LENGTH`` characters, or empty when invalid.
    """
    if not raw or not str(raw).strip():
        return ""
    normalized = unicodedata.normalize("NFKD", str(raw))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug in {"", ".", ".."}:
        return ""
    if len(slug) > constants.MAX_SLUG_LENGTH:
        slug = slug[: constants.MAX_SLUG_LENGTH].rstrip("_")
    return slug


def finalize_slug(slug: str) -> tuple[str, list[str]]:
    """Apply Windows reserved-name rewrite when needed.

    Args:
        slug: Candidate slug after ``slugify``.

    Returns:
        Final slug and warning codes.
    """
    warnings: list[str] = []
    if not slug:
        return slug, warnings
    if slug.lower() in constants.WINDOWS_RESERVED_SLUGS:
        slug = f"{slug}_dir"
        warnings.append("reserved_slug_rewritten")
    return slug, warnings


def assign_unique_slug(base: str, used: set[str]) -> str:
    """Assign ``base`` or ``base_N`` unique within ``used``.

    Args:
        base: Desired slug (non-empty).
        used: Slugs already assigned in the current scope.

    Returns:
        Unique slug; ``used`` is updated in place.
    """
    if base not in used:
        used.add(base)
        return base
    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        suffix += 1


@dataclasses.dataclass
class PackIdentity:
    """Lightweight simfile tags for slug assignment.

    Attributes:
        title: ``#TITLE`` value.
        artist: ``#ARTIST`` value.
        title_source: ``title``, ``translit``, or ``folder``.
        warnings: Read warnings (e.g. encoding fallback).
    """

    title: str
    artist: str
    title_source: str
    warnings: list[str]


def _pack_folder_name(pack_relpath: str) -> str:
    return pathlib.PurePosixPath(pack_relpath).name


def read_pack_identity(
    input_root: pathlib.Path,
    pack_relpath: str,
    simfile_name: str,
    *,
    folder_fallback_only: bool = False,
) -> PackIdentity:
    """Read title and artist from a pack for slug assignment.

    Args:
        input_root: Preprocess input directory.
        pack_relpath: Pack path relative to ``input_root``.
        simfile_name: Simfile basename from discovery.
        folder_fallback_only: When True, skip simfile read (dry-run preview).

    Returns:
        Identity with ``title_source`` indicating which field drove the slug.
    """
    folder_name = _pack_folder_name(pack_relpath)
    if folder_fallback_only:
        return PackIdentity(
            title=folder_name,
            artist="",
            title_source="folder",
            warnings=["dry_run_folder_slug"],
        )

    pack_dir = input_root / pack_relpath
    sim_path = pack_dir / simfile_name
    warnings: list[str] = []
    try:
        sim = simfile_adapter.open_simfile(sim_path)
    except UnicodeDecodeError:
        warnings.append("encoding_error_slug_fallback")
        return PackIdentity(
            title=folder_name,
            artist="",
            title_source="folder",
            warnings=warnings,
        )
    except (OSError, ValueError):
        warnings.append("parse_error_slug_fallback")
        return PackIdentity(
            title=folder_name,
            artist="",
            title_source="folder",
            warnings=warnings,
        )

    title = (sim.title or "").strip()
    translit = (getattr(sim, "titletranslit", None) or "").strip()
    artist = (sim.artist or "").strip()

    if slugify(title):
        return PackIdentity(
            title=title,
            artist=artist,
            title_source="title",
            warnings=warnings,
        )
    if slugify(translit):
        return PackIdentity(
            title=translit,
            artist=artist,
            title_source="translit",
            warnings=warnings,
        )
    warnings.append("empty_title_slug_fallback")
    return PackIdentity(
        title=folder_name,
        artist=artist,
        title_source="folder",
        warnings=warnings,
    )


def song_slug_from_identity(
    identity: PackIdentity, pack_relpath: str
) -> tuple[str, list[str]]:
    """Derive ``normalized_id`` base slug from pack identity.

    Args:
        identity: Pack identity from ``read_pack_identity``.
        pack_relpath: Pack path relative to input root.

    Returns:
        Base song slug (before collision suffix) and warnings.
    """
    warnings = list(identity.warnings)
    if identity.title_source == "folder":
        raw = _pack_folder_name(pack_relpath)
    else:
        raw = identity.title
    base = slugify(raw)
    if not base:
        base = slugify(_pack_folder_name(pack_relpath))
    if not base:
        base = "song"
    final, reserved_warnings = finalize_slug(base)
    warnings.extend(reserved_warnings)
    return final, warnings


def build_name_map(
    manifest: discovery.PacksManifest,
    prep_config: config.PrepConfig,
    *,
    folder_fallback_only: bool = False,
    limit: int | None = None,
) -> NameMap:
    """Build a name map from a discovery manifest.

    Args:
        manifest: Packs manifest from discovery.
        prep_config: Prep settings including ``output_dir``.
        folder_fallback_only: Use pack folder names only (dry-run preview).
        limit: When set, keep only the first N packs sorted by ``source_pack``.

    Returns:
        Name map with ``result=pending`` on every entry.
    """
    input_root = manifests.manifest_raw_input_root(manifest)
    packs = sorted(manifest.packs, key=lambda item: item.pack_relpath.lower())
    if limit is not None:
        packs = packs[:limit]

    bundle_slug_by_source: dict[str, str] = {}
    used_bundle_slugs: set[str] = set()

    for bundle in manifest.bundles:
        base = slugify(bundle.source_bundle)
        if not base:
            base = slugify(pathlib.Path(bundle.bundle_relpath).name) or "bundle"
        base, _warnings = finalize_slug(base)
        normalized_bundle = assign_unique_slug(base, used_bundle_slugs)
        bundle_slug_by_source[bundle.source_bundle] = normalized_bundle

    entries: list[NameMapEntry] = []
    used_song_slugs_by_bundle: dict[str, set[str]] = {}

    for pack in packs:
        normalized_bundle = bundle_slug_by_source[pack.source_bundle]
        used_song = used_song_slugs_by_bundle.setdefault(normalized_bundle, set())

        identity = read_pack_identity(
            input_root,
            pack.pack_relpath,
            pack.simfile,
            folder_fallback_only=folder_fallback_only,
        )
        song_base, warnings = song_slug_from_identity(identity, pack.pack_relpath)
        normalized_id = assign_unique_slug(song_base, used_song)
        output_relpath = f"{normalized_bundle}/{normalized_id}"

        entries.append(
            NameMapEntry(
                normalized_bundle=normalized_bundle,
                normalized_id=normalized_id,
                output_relpath=output_relpath,
                source_bundle=pack.source_bundle,
                source_pack=pack.pack_relpath,
                source_simfile=pack.simfile,
                title=identity.title if identity.title_source != "folder" else "",
                artist=identity.artist,
                audio_source="",
                result=pack_results.PACK_RESULT_PENDING,
                reason=None,
                warnings=list(dict.fromkeys(warnings)),
            )
        )

    return NameMap(
        schema_version=constants.SCHEMA_VERSION,
        raw_input_root=str(input_root.resolve()),
        output_dir=str(pathlib.Path(prep_config.output_dir).resolve()),
        entries=entries,
    )


def name_map_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return ``name_map.json`` path under ``output_dir``.

    Args:
        output_dir: Preprocess output root.

    Returns:
        Path to ``{output_dir}/name_map.json``.
    """
    return pathlib.Path(output_dir) / "name_map.json"


def name_map_csv_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return ``name_map.csv`` path under ``output_dir``.

    Args:
        output_dir: Preprocess output root.

    Returns:
        Path to ``{output_dir}/name_map.csv``.
    """
    return pathlib.Path(output_dir) / "name_map.csv"


def save_name_map(
    name_map: NameMap,
    output_dir: str | os.PathLike[str],
    *,
    merge: bool = True,
) -> NameMap:
    """Write ``name_map.json`` and ``name_map.csv``.

    Args:
        name_map: Name map for the current run.
        output_dir: Preprocess output root.
        merge: When True, merge with an existing name map instead of replacing it.

    Returns:
        The name map that was written (merged when ``merge`` is True).
    """
    path = name_map_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if merge and path.is_file():
        name_map = manifests.merge_name_maps(load_name_map(path), name_map)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(name_map.as_dict(), handle, indent=2)
        handle.write("\n")
    _save_name_map_csv(name_map, output_dir)
    return name_map


def _save_name_map_csv(name_map: NameMap, output_dir: str | os.PathLike[str]) -> None:
    csv_path = name_map_csv_path(output_dir)
    fieldnames = [
        "normalized_bundle",
        "normalized_id",
        "output_relpath",
        "source_bundle",
        "source_pack",
        "source_simfile",
        "title",
        "artist",
        "result",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in name_map.entries:
            writer.writerow(
                {
                    key: ("" if getattr(entry, key) is None else getattr(entry, key))
                    for key in fieldnames
                }
            )


def load_name_map(path: str | os.PathLike[str]) -> NameMap:
    """Load ``name_map.json`` from disk.

    Args:
        path: Path to a name map JSON file.

    Returns:
        Parsed name map with legacy ``source_pack`` paths canonicalized.

    Raises:
        ValueError: If ``schema_version`` is missing or unsupported.
    """
    path_obj = pathlib.Path(path)
    with path_obj.open(encoding="utf-8") as handle:
        data = json.load(handle)
    version = data.get("schema_version")
    if version is None:
        raise ValueError(f"missing schema_version in {path_obj}")
    if version != constants.SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version} in {path_obj}; "
            f"expected {constants.SCHEMA_VERSION}"
        )
    name_map = NameMap.from_dict(data)

    return dataclasses.replace(
        name_map,
        entries=[
            manifests.canonicalize_source_pack(entry) for entry in name_map.entries
        ],
    )


def run_normalization(
    manifest: discovery.PacksManifest,
    prep_config: config.PrepConfig,
    *,
    limit: int | None = None,
) -> NameMap:
    """Build and write ``name_map.json`` for a discovery manifest.

    Args:
        manifest: Packs manifest from discovery.
        prep_config: Prep settings including ``output_dir`` and ``dry_run``.
        limit: When set, keep only the first N packs sorted by ``source_pack``.

    Returns:
        Merged name map written to ``{output_dir}/name_map.json``.
    """
    name_map = build_name_map(
        manifest,
        prep_config,
        folder_fallback_only=prep_config.dry_run,
        limit=limit,
    )
    return save_name_map(name_map, prep_config.output_dir)
