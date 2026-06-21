"""Batch orchestration for dataset preprocessing."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import pathlib
from datetime import datetime

from stepcovnet.dataset_prep import (
    config,
    discovery,
    export,
    manifests,
    normalize,
    pack_results,
    simfile_adapter,
    validate,
)


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for report dataclasses."""

    def as_dict(self) -> dict:
        """Convert object to dictionary for JSON serialization."""
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create object from dictionary."""
        return cls(**data)


@dataclasses.dataclass
class PreprocessReport(_DictSerializableMixin):
    """Aggregate batch report for a preprocess run.

    Attributes:
        schema_version: Report layout version.
        raw_input_root: Root used to resolve ``source_pack`` paths.
        output_dir: Output root used for the run.
        started_at: ISO-8601 local timestamp at run start.
        finished_at: ISO-8601 local timestamp at run end.
        dry_run: True when discovery and normalization ran without pack writes.
        counts: Aggregate counters for the batch.
        skipped_packs: Expected filter outcomes (no export).
        pack_errors: Unexpected failures during processing.
        chart_skips: Per-chart skip records across the batch.
    """

    schema_version: int
    raw_input_root: str
    output_dir: str
    started_at: str
    finished_at: str
    dry_run: bool
    counts: dict[str, int]
    skipped_packs: list[dict]
    pack_errors: list[dict]
    chart_skips: list[dict]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def preprocess_report_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return ``preprocess_report.json`` path under ``output_dir``.

    Args:
        output_dir: Preprocess output root.

    Returns:
        Path to ``{output_dir}/preprocess_report.json``.
    """
    return pathlib.Path(output_dir) / "preprocess_report.json"


def save_preprocess_report(
    report: PreprocessReport | dict,
    output_dir: str | os.PathLike[str],
) -> pathlib.Path:
    """Write ``preprocess_report.json``.

    Args:
        report: ``PreprocessReport`` instance or already-serialized dict payload.
        output_dir: Preprocess output root.

    Returns:
        Path to the written JSON file.
    """
    path = preprocess_report_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report if isinstance(report, dict) else report.as_dict()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def _report_from_dict(payload: dict) -> PreprocessReport:
    return PreprocessReport(
        schema_version=payload["schema_version"],
        raw_input_root=payload.get("raw_input_root", payload.get("input_dir", "")),
        output_dir=payload["output_dir"],
        started_at=payload["started_at"],
        finished_at=payload["finished_at"],
        dry_run=payload["dry_run"],
        counts=payload["counts"],
        skipped_packs=payload["skipped_packs"],
        pack_errors=payload["pack_errors"],
        chart_skips=payload["chart_skips"],
    )


def _write_cumulative_report(
    name_map: normalize.NameMap,
    *,
    output_dir: str | os.PathLike[str],
    packs_manifest: discovery.PacksManifest,
    chart_skips: list[dict],
    started_at: str,
    finished_at: str,
    dry_run: bool,
) -> PreprocessReport:
    report_dict = manifests.build_preprocess_report(
        name_map,
        packs_manifest=packs_manifest,
        chart_skips=chart_skips,
        started_at=started_at,
        finished_at=finished_at,
        dry_run=dry_run,
    )
    save_preprocess_report(report_dict, output_dir)
    return _report_from_dict(report_dict)


def _chart_skip_dict(skip: simfile_adapter.ChartSkip, output_relpath: str) -> dict:
    return {
        "output_relpath": output_relpath,
        "difficulty": skip.difficulty,
        "meter": skip.meter,
        "reason": skip.reason,
    }


def _worker_result(
    entry: normalize.NameMapEntry,
    *,
    reason: str | None,
    warnings: list[str],
    charts_exported: int,
    charts_skipped: int,
    chart_skips: list[dict],
    message: str = "",
) -> export.WorkerResult:
    return export.WorkerResult(
        normalized_bundle=entry.normalized_bundle,
        normalized_id=entry.normalized_id,
        output_relpath=entry.output_relpath,
        source_pack=entry.source_pack,
        result=pack_results.pack_result(reason),
        reason=reason,
        warnings=warnings,
        charts_exported=charts_exported,
        charts_skipped=charts_skipped,
        chart_skips=chart_skips,
        message=message,
    )


def process_pack_entry(
    entry: normalize.NameMapEntry,
    *,
    input_dir: str,
    prep_config: config.PrepConfig,
) -> export.WorkerResult:
    """Parse, validate, and write one ``name_map`` entry.

    Args:
        entry: Normalized pack row to process.
        input_dir: Raw input root used to resolve ``entry.source_pack``.
        prep_config: Prep settings including output dir, caps, and overwrite flag.

    Returns:
        Worker outcome with result, reason, warnings, and chart skip records.
        Failures are returned as skipped or error results; this function does
        not raise for per-pack parse, validation, or I/O failures.
    """
    input_root = pathlib.Path(input_dir)
    raw_pack_dir = input_root / entry.source_pack
    output_relpath = entry.output_relpath

    final_dir = export.song_output_dir(
        prep_config.output_dir,
        entry.normalized_bundle,
        entry.normalized_id,
    )
    if final_dir.exists() and not prep_config.overwrite:
        return _worker_result(
            entry,
            reason=pack_results.REASON_OUTPUT_EXISTS,
            warnings=["output_exists"],
            charts_exported=0,
            charts_skipped=0,
            chart_skips=[],
            message="output directory exists and --overwrite is off",
        )

    parse_result = simfile_adapter.parse_song_pack(
        raw_pack_dir,
        simfile_name=entry.source_simfile,
        normalized_bundle=entry.normalized_bundle,
        normalized_id=entry.normalized_id,
        source_pack_relpath=entry.source_pack,
        prep_config=prep_config,
    )
    chart_skips = [
        _chart_skip_dict(skip, output_relpath) for skip in parse_result.chart_skips
    ]

    if parse_result.reason is not None or parse_result.pack is None:
        return _worker_result(
            entry,
            reason=parse_result.reason or pack_results.REASON_PARSE_ERROR,
            warnings=parse_result.warnings,
            charts_exported=0,
            charts_skipped=len(chart_skips),
            chart_skips=chart_skips,
        )

    validation_errors = validate.validate_parsed_pack(parse_result.pack)
    if validation_errors:
        return _worker_result(
            entry,
            reason=pack_results.REASON_VALIDATION_FAILED,
            warnings=parse_result.warnings + validation_errors,
            charts_exported=0,
            charts_skipped=len(chart_skips),
            chart_skips=chart_skips,
            message="validation failed",
        )

    try:
        export.write_song_pack(
            parse_result.pack,
            raw_pack_dir=raw_pack_dir,
            output_dir=prep_config.output_dir,
            prep_config=prep_config,
        )
    except (OSError, FileNotFoundError) as exc:
        return _worker_result(
            entry,
            reason=pack_results.REASON_IO_ERROR,
            warnings=parse_result.warnings,
            charts_exported=0,
            charts_skipped=len(chart_skips),
            chart_skips=chart_skips,
            message=str(exc),
        )

    pack = parse_result.pack
    return _worker_result(
        entry,
        reason=None,
        warnings=pack.warnings,
        charts_exported=len(pack.charts),
        charts_skipped=len(chart_skips),
        chart_skips=chart_skips,
    )


def _process_pack_entry_job(payload: dict) -> dict:
    """Picklable worker entry point for the process pool."""
    entry = normalize.NameMapEntry.from_dict(payload["entry"])
    prep_config = config.PrepConfig.from_dict(payload["prep_config"])
    if "export_mode" in payload["prep_config"]:
        prep_config.export_mode = config.ExportMode(
            payload["prep_config"]["export_mode"]
        )
    result = process_pack_entry(
        entry,
        input_dir=payload["input_dir"],
        prep_config=prep_config,
    )
    return result.as_dict()


def entry_needs_processing(
    entry: normalize.NameMapEntry,
    *,
    overwrite: bool,
) -> bool:
    """Return True when a name-map row should enter the worker pool.

    Args:
        entry: Normalized pack row from the merged name map.
        overwrite: When True, re-export rows that were already exported.

    Returns:
        True for pending rows, and for exported rows when ``overwrite`` is set.
    """
    if entry.result == pack_results.PACK_RESULT_PENDING:
        return True
    return overwrite and entry.result == pack_results.PACK_RESULT_EXPORTED


def _merge_results(
    name_map: normalize.NameMap,
    worker_results: list[export.WorkerResult],
) -> normalize.NameMap:
    result_by_key = {
        (item.normalized_bundle, item.normalized_id): item for item in worker_results
    }

    for entry in name_map.entries:
        key = (entry.normalized_bundle, entry.normalized_id)
        result = result_by_key.get(key)
        if result is None:
            continue

        entry.result = result.result
        entry.reason = result.reason
        entry.warnings = list(dict.fromkeys(entry.warnings + result.warnings))
        entry.charts_exported = result.charts_exported
        entry.charts_skipped = result.charts_skipped
        entry.message = result.message

    return name_map


def run_preprocess(prep_config: config.PrepConfig) -> PreprocessReport:
    """Run the full preprocess pipeline for a configuration.

    Args:
        prep_config: Validated prep settings.

    Returns:
        Final batch report written to ``preprocess_report.json``.
    """
    config.validate_prep_config(prep_config)
    started_at = _timestamp()
    limit = prep_config.limit_packs

    manifest = discovery.run_discovery(prep_config)
    name_map = normalize.run_normalization(manifest, prep_config, limit=limit)
    merged_manifest = discovery.load_packs_manifest(
        discovery.packs_manifest_path(prep_config.output_dir)
    )
    chart_skips = manifests.load_chart_skips(prep_config.output_dir)

    if prep_config.dry_run:
        finished_at = _timestamp()
        return _write_cumulative_report(
            name_map,
            output_dir=prep_config.output_dir,
            packs_manifest=merged_manifest,
            chart_skips=chart_skips,
            started_at=started_at,
            finished_at=finished_at,
            dry_run=True,
        )

    worker_results: list[export.WorkerResult] = []
    prep_payload = prep_config.as_dict()
    prep_payload["export_mode"] = str(prep_config.export_mode)

    jobs = [
        {
            "entry": entry.as_dict(),
            "input_dir": name_map.raw_input_root,
            "prep_config": prep_payload,
        }
        for entry in name_map.entries
        if entry_needs_processing(entry, overwrite=prep_config.overwrite)
    ]

    if prep_config.workers == 1:
        for job in jobs:
            result_dict = _process_pack_entry_job(job)
            result = export.WorkerResult.from_dict(result_dict)
            worker_results.append(result)
            export.save_worker_result(result, prep_config.output_dir)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=prep_config.workers
        ) as pool:
            futures = [pool.submit(_process_pack_entry_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = export.WorkerResult.from_dict(future.result())
                worker_results.append(result)
                export.save_worker_result(result, prep_config.output_dir)

    finished_at = _timestamp()
    run_chart_skips: list[dict] = []
    for result in worker_results:
        run_chart_skips.extend(result.chart_skips)
    chart_skips = manifests.merge_chart_skips(chart_skips, run_chart_skips)

    run_name_map = normalize.build_name_map(manifest, prep_config, limit=limit)
    merged_name_map = _merge_results(run_name_map, worker_results)
    name_map = normalize.save_name_map(merged_name_map, prep_config.output_dir)
    name_map = manifests.supplement_name_map_from_worker_results(
        name_map,
        manifests.load_worker_results(prep_config.output_dir),
        merged_manifest,
    )
    normalize.save_name_map(name_map, prep_config.output_dir, merge=False)
    return _write_cumulative_report(
        name_map,
        output_dir=prep_config.output_dir,
        packs_manifest=merged_manifest,
        chart_skips=chart_skips,
        started_at=started_at,
        finished_at=finished_at,
        dry_run=False,
    )
