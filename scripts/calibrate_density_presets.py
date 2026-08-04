#!/usr/bin/env python
"""Write customer difficulty presets (fixed onsets/sec targets by default)."""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import librosa

from stepcovnet.dataset_prep import training_index, training_loader
from stepcovnet.onset_ar import density_presets
from stepcovnet.onset_events import charts
from stepcovnet.onset_events import targets as event_targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write configs/ar/density_presets.json. Default: fixed design targets "
            "(2/4/6/8/10 onsets/s). Optionally scan a training index for corpus "
            "coverage counts only — tier values are not derived from labels or buckets."
        ),
    )
    parser.add_argument(
        "--method",
        choices=(
            density_presets.CALIBRATION_METHOD_FIXED,
            density_presets.CALIBRATION_METHOD_EQUAL_COUNT,
        ),
        default=density_presets.CALIBRATION_METHOD_FIXED,
        help="Preset source (default: fixed_onset_hz_targets).",
    )
    parser.add_argument(
        "--training-index-path",
        default="data/final_data/training_index.json",
        help="Manifest for optional coverage report (default: full index).",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Do not scan the training index; tier n_rows stay 0.",
    )
    parser.add_argument(
        "--output",
        default=density_presets.DEFAULT_PRESETS_PATH,
        help="Output preset JSON path.",
    )
    parser.add_argument(
        "--onset-hz-norm",
        type=float,
        default=15.0,
        help="Normalization used by onset_density conditioning (default 15).",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=300.0,
        help="Clip charts/audio to this duration (matches AR training cap).",
    )
    parser.add_argument(
        "--max-steps-per-chart",
        type=int,
        default=2048,
        help="Skip charts exceeding this encoded step cap.",
    )
    return parser


def _duration_sec(audio_path: str, cache: dict[str, float]) -> float:
    if audio_path not in cache:
        cache[audio_path] = float(librosa.get_duration(path=audio_path))
    return cache[audio_path]


def _onsets_per_sec_for_row(
    row: training_loader.TrainingChartRow,
    *,
    duration_cache: dict[str, float],
    max_audio_seconds: float,
    max_steps_per_chart: int,
) -> float | None:
    if charts.chart_exceeds_step_cap(
        row.chart_json_path,
        max_steps=max_steps_per_chart,
        chart_index=row.chart_index,
    ):
        return None
    raw_times = charts.load_onset_times(
        row.chart_json_path,
        max_steps=max_steps_per_chart,
        chart_index=row.chart_index,
    )
    if raw_times is None:
        return None
    duration = min(
        _duration_sec(row.audio_path, duration_cache),
        float(max_audio_seconds),
    )
    if duration <= 0.0:
        return None
    times = event_targets.clip_times_to_duration(raw_times, duration)
    if times.size == 0:
        return None
    return float(times.size) / duration


def collect_onsets_per_sec(
    training_index_path: str | pathlib.Path,
    *,
    max_audio_seconds: float = 300.0,
    max_steps_per_chart: int = 2048,
) -> tuple[list[float], pathlib.Path]:
    """Measure onsets/sec for every chart row in a manifest."""
    index_path = pathlib.Path(training_index_path)
    index = training_index.load_training_index(index_path)
    root = training_index.resolve_output_dir(index, index_path)
    rows = training_index.rows_from_index(index, root, split=None)

    onsets_per_sec: list[float] = []
    duration_cache: dict[str, float] = {}
    for row in rows:
        hz = _onsets_per_sec_for_row(
            row,
            duration_cache=duration_cache,
            max_audio_seconds=max_audio_seconds,
            max_steps_per_chart=max_steps_per_chart,
        )
        if hz is not None:
            onsets_per_sec.append(hz)
    return onsets_per_sec, index_path


def calibrate_presets(
    training_index_path: str | pathlib.Path | None,
    *,
    method: str = density_presets.CALIBRATION_METHOD_FIXED,
    onset_hz_norm: float = 15.0,
    max_audio_seconds: float = 300.0,
    max_steps_per_chart: int = 2048,
    skip_coverage: bool = False,
) -> density_presets.DensityPresets:
    """Build presets; optional index scan is coverage-only for fixed targets."""
    onsets_per_sec: list[float] = []
    index_path = pathlib.Path(training_index_path or "")
    if not skip_coverage and training_index_path:
        onsets_per_sec, index_path = collect_onsets_per_sec(
            training_index_path,
            max_audio_seconds=max_audio_seconds,
            max_steps_per_chart=max_steps_per_chart,
        )

    created = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    created_at = created.isoformat().replace("+00:00", "Z")
    index_str = str(index_path.as_posix()) if training_index_path else ""

    if method == density_presets.CALIBRATION_METHOD_FIXED:
        coverage = (
            density_presets.coverage_counts_for_onsets_per_sec(onsets_per_sec)
            if onsets_per_sec
            else None
        )
        return density_presets.build_fixed_density_presets(
            onset_hz_norm=onset_hz_norm,
            coverage_counts=coverage,
            source_training_index_path=index_str,
            n_rows_total=len(onsets_per_sec),
            created_at=created_at,
        )

    if not onsets_per_sec:
        raise ValueError(
            "equal_count_quintiles requires a training index scan; omit --skip-coverage",
        )
    buckets = density_presets.bucket_onsets_per_sec_equal_count(onsets_per_sec)
    tier_stats = density_presets.build_tier_presets_from_buckets(
        buckets,
        tier_order=density_presets.CUSTOMER_TIER_ORDER,
        onset_hz_norm=onset_hz_norm,
    )
    return density_presets.DensityPresets(
        schema_version=1,
        onset_hz_norm=float(onset_hz_norm),
        source_training_index_path=index_str,
        n_rows_total=len(onsets_per_sec),
        created_at=created_at,
        tiers=tier_stats,
        tier_order=density_presets.CUSTOMER_TIER_ORDER,
        calibration_method=density_presets.CALIBRATION_METHOD_EQUAL_COUNT,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    index_path = None if args.skip_coverage else args.training_index_path
    presets = calibrate_presets(
        index_path,
        method=args.method,
        onset_hz_norm=float(args.onset_hz_norm),
        max_audio_seconds=float(args.max_audio_seconds),
        max_steps_per_chart=int(args.max_steps_per_chart),
        skip_coverage=bool(args.skip_coverage),
    )
    density_presets.save_density_presets(presets, args.output)
    print(
        f"Wrote {args.output} (method={presets.calibration_method})",
        file=sys.stderr,
    )
    if presets.calibration_method == density_presets.CALIBRATION_METHOD_FIXED:
        cuts = density_presets.DEFAULT_COVERAGE_THRESHOLDS
        print(
            "  Fixed targets (onsets/s): "
            + ", ".join(
                f"{tier}={density_presets.DEFAULT_FIXED_ONSET_HZ_TARGETS[tier]:g}"
                for tier in presets.tier_order
            ),
            file=sys.stderr,
        )
        print(
            f"  Coverage bands (hz): "
            f"[0,{cuts[0]}), [{cuts[0]},{cuts[1]}), [{cuts[1]},{cuts[2]}), "
            f"[{cuts[2]},{cuts[3]}), [{cuts[3]},inf)",
            file=sys.stderr,
        )
    if presets.n_rows_total:
        print(f"  Corpus charts scanned: {presets.n_rows_total}", file=sys.stderr)
    for tier in presets.tier_order:
        item = presets.tiers[tier]
        label = (
            "target"
            if presets.calibration_method == density_presets.CALIBRATION_METHOD_FIXED
            else "median"
        )
        suffix = f"  coverage_n={item.n_rows}" if item.n_rows else ""
        print(
            f"  {tier:10s}  {label}={item.onsets_per_sec_median:6.3f} onsets/s  "
            f"density={item.density_scalar:.3f}{suffix}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
