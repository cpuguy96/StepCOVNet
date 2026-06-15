"""Discover raw song packs under bundle directories (P1)."""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import pathlib

from stepcovnet.dataset_prep import config, constants


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for manifest dataclasses."""

    def as_dict(self) -> dict:
        """Convert manifest object to dictionary for JSON serialization."""
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create manifest object from dictionary.

        Returns:
            Instance of the manifest class with fields taken from data.
        """
        return cls(**data)


class DiscoveryMode(enum.StrEnum):
    """How ``--input-dir`` is interpreted for bundle grouping."""

    MULTI_BUNDLE = "multi_bundle"
    SINGLE_BUNDLE = "single_bundle"


@dataclasses.dataclass
class PackManifestEntry(_DictSerializableMixin):
    """One discovered pack row in ``packs_manifest.json``.

    Attributes:
        pack_relpath: Pack directory relative to ``input_dir``.
        simfile: Basename of chosen simfile (``.ssc`` preferred over ``.sm``).
        source_bundle: Raw bundle folder name for provenance.
        bundle_relpath: Bundle path relative to ``input_dir``.
    """

    pack_relpath: str
    simfile: str
    source_bundle: str
    bundle_relpath: str


@dataclasses.dataclass
class BundleManifestEntry(_DictSerializableMixin):
    """Summary of one bundle in ``packs_manifest.json``.

    Attributes:
        source_bundle: Raw bundle folder name.
        bundle_relpath: Bundle path relative to ``input_dir``.
        pack_count: Number of packs discovered under this bundle.
    """

    source_bundle: str
    bundle_relpath: str
    pack_count: int


@dataclasses.dataclass
class PacksManifest(_DictSerializableMixin):
    """Phase O1 output listing every pack to process.

    Attributes:
        schema_version: Manifest layout version.
        input_dir: Absolute or normalized input root used for discovery.
        discovery_mode: ``multi_bundle`` or ``single_bundle``.
        bundles: Per-bundle summaries.
        packs: Flat list of pack rows for workers and normalization.
        warnings: Non-fatal discovery issues (e.g. empty bundle folders).
    """

    schema_version: int
    input_dir: str
    discovery_mode: DiscoveryMode
    bundles: list[BundleManifestEntry]
    packs: list[PackManifestEntry]
    warnings: list[str]

    @classmethod
    def from_dict(cls, data: dict):
        """Parse manifest including nested bundle and pack rows."""
        payload = dict(data)
        payload["discovery_mode"] = DiscoveryMode(data["discovery_mode"])
        payload["bundles"] = [
            BundleManifestEntry.from_dict(item) for item in data.get("bundles", [])
        ]
        payload["packs"] = [
            PackManifestEntry.from_dict(item) for item in data.get("packs", [])
        ]
        payload["warnings"] = list(data.get("warnings", []))
        return cls(**payload)


def is_simfile_name(name: str) -> bool:
    """Return True if ``name`` ends with a supported simfile extension.

    Args:
        name: File basename or path tail.

    Returns:
        True for ``.ssc`` or ``.sm`` (case-insensitive).
    """
    lowered = name.lower()
    return any(lowered.endswith(ext) for ext in constants.SIMFILE_EXTENSIONS)


def is_pack_dir(pack_dir: pathlib.Path) -> bool:
    """Return True if ``pack_dir`` directly contains a simfile.

    Args:
        pack_dir: Candidate pack directory.

    Returns:
        True when a ``*.ssc`` or ``*.sm`` file exists in the directory (non-recursive).
    """
    if not pack_dir.is_dir():
        return False
    for path in pack_dir.iterdir():
        if path.is_file() and is_simfile_name(path.name):
            return True
    return False


def choose_simfile(pack_dir: pathlib.Path) -> str:
    """Pick the simfile basename for a pack directory.

    Prefers ``*.ssc`` over ``*.sm``; within one extension, lowest sorted name wins.

    Args:
        pack_dir: Pack directory containing simfiles.

    Returns:
        Chosen simfile basename.

    Raises:
        FileNotFoundError: If no simfile is present in ``pack_dir``.
    """
    ssc_files = sorted(path.name for path in pack_dir.glob("*.ssc") if path.is_file())
    if ssc_files:
        return ssc_files[0]
    sm_files = sorted(path.name for path in pack_dir.glob("*.sm") if path.is_file())
    if sm_files:
        return sm_files[0]
    raise FileNotFoundError(f"no simfile in pack directory: {pack_dir}")


def list_pack_dirs(bundle_dir: pathlib.Path) -> list[pathlib.Path]:
    """List immediate child directories of ``bundle_dir`` that are packs.

    Args:
        bundle_dir: Bundle root containing pack subdirectories.

    Returns:
        Sorted pack directory paths (one level only).
    """
    packs: list[pathlib.Path] = []
    if not bundle_dir.is_dir():
        return packs
    for child in sorted(bundle_dir.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and is_pack_dir(child):
            packs.append(child)
    return packs


def _relpath_posix(root: pathlib.Path, path: pathlib.Path) -> str:
    """Return ``path`` relative to ``root`` using forward slashes."""
    return path.relative_to(root).as_posix()


def build_packs_manifest(input_dir: str | os.PathLike[str]) -> PacksManifest:
    """Run phase O1 discovery on ``input_dir``.

    Args:
        input_dir: Multi-bundle root (e.g. ``data/raw_data``) or single bundle path.

    Returns:
        Manifest with every pack row and bundle summaries.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
    """
    root = pathlib.Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {root}")

    direct_packs = list_pack_dirs(root)
    warnings: list[str] = []
    packs: list[PackManifestEntry] = []
    bundles: list[BundleManifestEntry] = []

    if direct_packs:
        discovery_mode = DiscoveryMode.SINGLE_BUNDLE
        source_bundle = root.name
        bundle_relpath = source_bundle
        for pack_dir in direct_packs:
            packs.append(
                PackManifestEntry(
                    pack_relpath=_relpath_posix(root, pack_dir),
                    simfile=choose_simfile(pack_dir),
                    source_bundle=source_bundle,
                    bundle_relpath=bundle_relpath,
                )
            )
        bundles.append(
            BundleManifestEntry(
                source_bundle=source_bundle,
                bundle_relpath=bundle_relpath,
                pack_count=len(direct_packs),
            )
        )
    else:
        discovery_mode = DiscoveryMode.MULTI_BUNDLE
        for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
            if not child.is_dir():
                continue
            child_packs = list_pack_dirs(child)
            if not child_packs:
                warnings.append(f"empty_bundle:{_relpath_posix(root, child)}")
                continue
            source_bundle = child.name
            bundle_relpath = _relpath_posix(root, child)
            bundles.append(
                BundleManifestEntry(
                    source_bundle=source_bundle,
                    bundle_relpath=bundle_relpath,
                    pack_count=len(child_packs),
                )
            )
            for pack_dir in child_packs:
                packs.append(
                    PackManifestEntry(
                        pack_relpath=_relpath_posix(root, pack_dir),
                        simfile=choose_simfile(pack_dir),
                        source_bundle=source_bundle,
                        bundle_relpath=bundle_relpath,
                    )
                )

    return PacksManifest(
        schema_version=constants.SCHEMA_VERSION,
        input_dir=str(root),
        discovery_mode=discovery_mode,
        bundles=bundles,
        packs=packs,
        warnings=warnings,
    )


def packs_manifest_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return the canonical ``packs_manifest.json`` path under ``output_dir``.

    Args:
        output_dir: Preprocess output root.

    Returns:
        Path to ``{output_dir}/packs_manifest.json``.
    """
    return pathlib.Path(output_dir) / "packs_manifest.json"


def save_packs_manifest(
    manifest: PacksManifest,
    output_dir: str | os.PathLike[str],
) -> pathlib.Path:
    """Write ``packs_manifest.json`` under ``output_dir``.

    Args:
        manifest: Discovery result to serialize.
        output_dir: Directory that receives the manifest file.

    Returns:
        Path to the written JSON file.
    """
    path = packs_manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.as_dict()
    payload["discovery_mode"] = str(manifest.discovery_mode)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def load_packs_manifest(path: str | os.PathLike[str]) -> PacksManifest:
    """Load ``packs_manifest.json`` from disk.

    Args:
        path: Path to a packs manifest JSON file.

    Returns:
        Parsed manifest.

    Raises:
        ValueError: If ``schema_version`` is missing or unsupported.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    version = data.get("schema_version")
    if version is None:
        raise ValueError(f"missing schema_version in {path}")
    if version != constants.SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {version} in {path}; "
            f"expected {constants.SCHEMA_VERSION}"
        )
    return PacksManifest.from_dict(data)


def run_discovery(prep_config: config.PrepConfig) -> PacksManifest:
    """Validate config, discover packs, and write ``packs_manifest.json``.

    Args:
        prep_config: Prep settings including ``input_dir`` and ``output_dir``.

    Returns:
        Discovery manifest also written to ``{output_dir}/packs_manifest.json``.

    Raises:
        ValueError: If prep config fails validation.
        FileNotFoundError: If ``input_dir`` does not exist.
    """
    config.validate_prep_config(prep_config)
    manifest = build_packs_manifest(prep_config.input_dir)
    save_packs_manifest(manifest, prep_config.output_dir)
    return manifest
