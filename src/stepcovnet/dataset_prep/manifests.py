"""Merge discovery and normalization manifests across preprocess runs."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib

from stepcovnet.dataset_prep import constants, discovery, normalize, pack_results


def canonicalize_pack_relpath(
    pack: discovery.PackManifestEntry,
) -> discovery.PackManifestEntry:
    """Rewrite legacy single-bundle ``pack_relpath`` values to include the bundle.

    Args:
        pack: Discovery pack row, possibly with a folder-only ``pack_relpath``.

    Returns:
        ``pack`` unchanged when already canonical; otherwise a copy with
        ``{bundle_relpath}/{folder}`` as ``pack_relpath``.
    """
    prefix = f"{pack.bundle_relpath}/"
    if pack.pack_relpath.startswith(prefix):
        return pack
    if "/" not in pack.pack_relpath:
        return dataclasses.replace(
            pack,
            pack_relpath=f"{pack.bundle_relpath}/{pack.pack_relpath}",
        )
    return pack


def canonicalize_source_pack(entry: normalize.NameMapEntry) -> normalize.NameMapEntry:
    """Rewrite legacy ``source_pack`` values to include ``source_bundle``.

    Args:
        entry: Name map row, possibly with a folder-only ``source_pack``.

    Returns:
        ``entry`` unchanged when already canonical; otherwise a copy with
        ``{source_bundle}/{folder}`` as ``source_pack``.
    """
    prefix = f"{entry.source_bundle}/"
    if entry.source_pack.startswith(prefix):
        return entry
    if "/" not in entry.source_pack:
        return dataclasses.replace(
            entry,
            source_pack=f"{entry.source_bundle}/{entry.source_pack}",
        )
    return entry


def manifest_raw_input_root(manifest: discovery.PacksManifest) -> pathlib.Path:
    """Return the directory used to resolve ``pack_relpath`` / ``source_pack`` paths.

    Args:
        manifest: Discovery manifest from ``build_packs_manifest`` or disk.

    Returns:
        Parent of ``manifest.input_dir`` in single-bundle mode; ``manifest.input_dir``
        itself in multi-bundle mode.
    """
    root = pathlib.Path(manifest.input_dir)
    if manifest.discovery_mode == discovery.DiscoveryMode.SINGLE_BUNDLE:
        return root.parent
    return root


def merge_raw_input_roots(existing: str, new: str) -> str:
    """Pick a shared raw input root when merging manifests from separate runs.

    Args:
        existing: Raw input root recorded in an on-disk manifest.
        new: Raw input root from the current run.

    Returns:
        The more general (ancestor) path when one root contains the other;
        otherwise ``new`` resolved as a string.
    """
    existing_path = pathlib.Path(existing).resolve()
    new_path = pathlib.Path(new).resolve()
    if existing_path == new_path:
        return str(existing_path)
    if existing_path in new_path.parents:
        return str(existing_path)
    if new_path in existing_path.parents:
        return str(new_path)
    return str(new_path)


def merge_packs_manifest(
    existing: discovery.PacksManifest,
    new: discovery.PacksManifest,
) -> discovery.PacksManifest:
    """Merge two discovery manifests, with ``new`` rows overriding on key clash.

    Args:
        existing: Manifest already on disk or from a prior run.
        new: Manifest from the current discovery pass.

    Returns:
        Combined manifest with ``discovery_mode=multi_bundle``, deduplicated
        warnings, and pack rows keyed by canonical ``pack_relpath``.
    """
    packs_by_relpath = {
        canonicalize_pack_relpath(pack).pack_relpath: canonicalize_pack_relpath(pack)
        for pack in existing.packs
    }
    for pack in new.packs:
        canonical = canonicalize_pack_relpath(pack)
        packs_by_relpath[canonical.pack_relpath] = canonical

    bundles_by_name = {bundle.source_bundle: bundle for bundle in existing.bundles}
    for bundle in new.bundles:
        bundles_by_name[bundle.source_bundle] = bundle

    warnings = list(dict.fromkeys(existing.warnings + new.warnings))
    raw_input_root = merge_raw_input_roots(
        str(manifest_raw_input_root(existing)),
        str(manifest_raw_input_root(new)),
    )

    return discovery.PacksManifest(
        schema_version=new.schema_version,
        input_dir=raw_input_root,
        discovery_mode=discovery.DiscoveryMode.MULTI_BUNDLE,
        bundles=sorted(
            bundles_by_name.values(),
            key=lambda item: item.bundle_relpath.lower(),
        ),
        packs=sorted(
            packs_by_relpath.values(),
            key=lambda item: item.pack_relpath.lower(),
        ),
        warnings=warnings,
    )


def merge_name_maps(
    existing: normalize.NameMap,
    new: normalize.NameMap,
) -> normalize.NameMap:
    """Merge two name maps, with ``new`` rows overriding on ``source_pack`` clash.

    Terminal rows in ``existing`` (exported, skipped, or error) are preserved
    when ``new`` supplies only a pending row for the same ``source_pack``.

    Args:
        existing: Name map already on disk or from a prior run.
        new: Name map from the current normalization pass.

    Returns:
        Combined name map sorted by ``(normalized_bundle, normalized_id)``.
    """
    by_source_pack = {
        canonicalize_source_pack(entry).source_pack: canonicalize_source_pack(entry)
        for entry in existing.entries
    }
    for entry in new.entries:
        canonical = canonicalize_source_pack(entry)
        previous = by_source_pack.get(canonical.source_pack)
        if (
            previous is not None
            and previous.result != pack_results.PACK_RESULT_PENDING
            and canonical.result == pack_results.PACK_RESULT_PENDING
        ):
            continue
        by_source_pack[canonical.source_pack] = canonical

    raw_input_root = merge_raw_input_roots(existing.raw_input_root, new.raw_input_root)
    return normalize.NameMap(
        schema_version=new.schema_version,
        raw_input_root=raw_input_root,
        output_dir=new.output_dir,
        entries=sorted(
            by_source_pack.values(),
            key=lambda item: (
                item.normalized_bundle.lower(),
                item.normalized_id.lower(),
            ),
        ),
    )


def _entry_report_row(entry: normalize.NameMapEntry) -> dict | None:
    if entry.result == pack_results.PACK_RESULT_SKIPPED:
        return pack_results.pack_entry_row(
            source_pack=entry.source_pack,
            normalized_bundle=entry.normalized_bundle,
            normalized_id=entry.normalized_id,
            reason=entry.reason or pack_results.REASON_PARSE_ERROR,
            warnings=entry.warnings,
        )
    if entry.result == pack_results.PACK_RESULT_ERROR:
        return pack_results.pack_entry_row(
            source_pack=entry.source_pack,
            normalized_bundle=entry.normalized_bundle,
            normalized_id=entry.normalized_id,
            reason=entry.reason or pack_results.REASON_IO_ERROR,
            warnings=entry.warnings,
            message=entry.message,
        )
    return None


def build_preprocess_report(
    name_map: normalize.NameMap,
    *,
    packs_manifest: discovery.PacksManifest | None,
    chart_skips: list[dict],
    started_at: str,
    finished_at: str,
    dry_run: bool,
) -> dict:
    """Build cumulative preprocess report JSON payload from merged name map state.

    Args:
        name_map: Merged name map after normalization and pack processing.
        packs_manifest: Optional merged discovery manifest for ``packs_discovered``.
        chart_skips: Per-chart skip rows to embed in the report.
        started_at: ISO-8601 local timestamp at batch start.
        finished_at: ISO-8601 local timestamp at batch end.
        dry_run: Whether the run skipped pack writes.

    Returns:
        Serializable report dict matching ``preprocess_report.json`` layout.
    """
    exported = 0
    skipped = 0
    errors = 0
    charts_exported = 0
    charts_skipped = 0
    skipped_packs: list[dict] = []
    pack_errors: list[dict] = []

    for entry in name_map.entries:
        if entry.result == pack_results.PACK_RESULT_EXPORTED:
            exported += 1
            charts_exported += entry.charts_exported
            charts_skipped += entry.charts_skipped
            continue
        if entry.result == pack_results.PACK_RESULT_PENDING:
            continue
        row = _entry_report_row(entry)
        if row is None:
            continue
        if entry.result == pack_results.PACK_RESULT_SKIPPED:
            skipped += 1
            skipped_packs.append(row)
        elif entry.result == pack_results.PACK_RESULT_ERROR:
            errors += 1
            pack_errors.append(row)

    packs_discovered = (
        len(packs_manifest.packs)
        if packs_manifest is not None
        else len(name_map.entries)
    )

    return {
        "schema_version": constants.SCHEMA_VERSION,
        "raw_input_root": name_map.raw_input_root,
        "output_dir": name_map.output_dir,
        "started_at": started_at,
        "finished_at": finished_at,
        "dry_run": dry_run,
        "counts": {
            "packs_discovered": packs_discovered,
            "packs_scheduled": len(name_map.entries),
            "packs_exported": exported,
            "packs_skipped": skipped,
            "packs_errors": errors,
            "charts_exported": charts_exported,
            "charts_skipped": charts_skipped,
        },
        "skipped_packs": skipped_packs,
        "pack_errors": pack_errors,
        "chart_skips": chart_skips,
    }


def load_chart_skips(output_dir: str | os.PathLike[str]) -> list[dict]:
    """Load chart skip rows from an existing preprocess report if present.

    Args:
        output_dir: Preprocess output root containing ``preprocess_report.json``.

    Returns:
        ``chart_skips`` list from the report, or an empty list when the report
        is missing.
    """
    path = pathlib.Path(output_dir) / "preprocess_report.json"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("chart_skips", []))


def merge_chart_skips(existing: list[dict], new_items: list[dict]) -> list[dict]:
    """Merge chart skip rows by output path, difficulty, and reason.

    Args:
        existing: Chart skip rows from prior runs or the on-disk report.
        new_items: Chart skip rows from the current batch.

    Returns:
        Deduplicated list; later rows override earlier rows on the same key.
    """
    merged: dict[tuple[str, str, str], dict] = {}
    for item in existing + new_items:
        key = (
            item.get("output_relpath", ""),
            item.get("difficulty", ""),
            item.get("reason", ""),
        )
        merged[key] = item
    return list(merged.values())


def load_worker_results(output_dir: str | os.PathLike[str]) -> list[dict]:
    """Load per-pack worker JSON results from ``_staging/worker_results/``.

    Args:
        output_dir: Preprocess output root.

    Returns:
        Parsed worker result dicts sorted by filename, or an empty list when
        the staging directory is missing.
    """
    staging = pathlib.Path(output_dir) / "_staging" / "worker_results"
    if not staging.is_dir():
        return []
    results: list[dict] = []
    for path in sorted(staging.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def resolve_pack_for_worker_result(
    worker_result: dict,
    packs_manifest: discovery.PacksManifest,
) -> discovery.PackManifestEntry | None:
    """Match a worker result to a discovery pack row.

    Args:
        worker_result: Serialized ``WorkerResult`` dict from staging.
        packs_manifest: Merged discovery manifest for bundle and pack lookup.

    Returns:
        Canonical pack row when a unique match is found; otherwise ``None``.
    """
    source_pack = worker_result.get("source_pack", "")
    bundle_slug = worker_result.get("normalized_bundle", "")
    for pack in packs_manifest.packs:
        if normalize.slugify(pack.source_bundle) != bundle_slug:
            continue
        canonical = canonicalize_pack_relpath(pack)
        tail = canonical.pack_relpath.split("/")[-1]
        if (
            canonical.pack_relpath == source_pack
            or canonical.pack_relpath.endswith("/" + source_pack)
            or tail == source_pack.split("/")[-1]
        ):
            return canonical
    return None


def supplement_name_map_from_worker_results(
    name_map: normalize.NameMap,
    worker_results: list[dict],
    packs_manifest: discovery.PacksManifest | None,
) -> normalize.NameMap:
    """Merge staged worker results into ``name_map`` for report and manifest state.

    Worker rows override or add entries keyed by ``(normalized_bundle,
    normalized_id)``. Title and artist are preserved from existing name-map
    rows when present.

    Args:
        name_map: Current merged name map.
        worker_results: Parsed worker JSON dicts from ``load_worker_results``.
        packs_manifest: Optional discovery manifest used to resolve ``source_pack``
            and ``source_simfile`` for staging rows with legacy paths.

    Returns:
        Updated name map with worker outcomes applied.
    """
    by_key = {
        (entry.normalized_bundle, entry.normalized_id): entry
        for entry in name_map.entries
    }

    for worker_result in worker_results:
        key = (
            worker_result["normalized_bundle"],
            worker_result["normalized_id"],
        )
        existing = by_key.get(key)
        pack = (
            resolve_pack_for_worker_result(worker_result, packs_manifest)
            if packs_manifest is not None
            else None
        )
        source_bundle = pack.source_bundle if pack is not None else ""
        source_simfile = pack.simfile if pack is not None else ""
        source_pack = (
            pack.pack_relpath
            if pack is not None
            else worker_result.get("source_pack", "")
        )
        if source_bundle:
            source_pack = canonicalize_source_pack(
                normalize.NameMapEntry(
                    normalized_bundle=worker_result["normalized_bundle"],
                    normalized_id=worker_result["normalized_id"],
                    output_relpath=worker_result["output_relpath"],
                    source_bundle=source_bundle,
                    source_pack=source_pack,
                    source_simfile=source_simfile,
                    title="",
                    artist="",
                    audio_source="",
                    result=worker_result["result"],
                    reason=worker_result.get("reason"),
                    warnings=worker_result.get("warnings", []),
                )
            ).source_pack

        by_key[key] = normalize.NameMapEntry(
            normalized_bundle=worker_result["normalized_bundle"],
            normalized_id=worker_result["normalized_id"],
            output_relpath=worker_result["output_relpath"],
            source_bundle=source_bundle or (existing.source_bundle if existing else ""),
            source_pack=source_pack,
            source_simfile=source_simfile
            or (existing.source_simfile if existing else ""),
            title=existing.title if existing else "",
            artist=existing.artist if existing else "",
            audio_source=existing.audio_source if existing else "",
            result=worker_result["result"],
            reason=worker_result.get("reason"),
            warnings=list(
                dict.fromkeys(
                    (existing.warnings if existing else [])
                    + worker_result.get("warnings", [])
                )
            ),
            charts_exported=worker_result.get("charts_exported", 0),
            charts_skipped=worker_result.get("charts_skipped", 0),
            message=worker_result.get("message", ""),
        )

    return normalize.NameMap(
        schema_version=name_map.schema_version,
        raw_input_root=name_map.raw_input_root,
        output_dir=name_map.output_dir,
        entries=sorted(
            by_key.values(),
            key=lambda item: (
                item.normalized_bundle.lower(),
                item.normalized_id.lower(),
            ),
        ),
    )
