#!/usr/bin/env python
"""CLI for raw simfile pack preprocessing."""

from __future__ import annotations

import argparse
import json
import sys

from stepcovnet.dataset_prep import config, constants, pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess raw StepMania packs into nested final_data layout.",
    )
    parser.add_argument(
        "--input-dir",
        default=constants.DEFAULT_INPUT_DIR,
        help="Raw pack root or single bundle directory",
    )
    parser.add_argument(
        "--output-dir",
        default=constants.DEFAULT_OUTPUT_DIR,
        help="Processed output root",
    )
    parser.add_argument(
        "--export-mode",
        default=constants.EXPORT_MODE_ALL_SINGLES,
        choices=[constants.EXPORT_MODE_ALL_SINGLES],
    )
    parser.add_argument(
        "--max-steps-per-chart",
        type=int,
        default=constants.MAX_STEPS_PER_CHART,
    )
    parser.add_argument(
        "--export-legacy-txt",
        action="store_true",
        help="Write multi-block v2 .txt beside chart JSON",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery and normalization only (no pack writes)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing bundle/id output directories",
    )
    parser.add_argument(
        "--allow-over-cap",
        action="store_true",
        help="Export charts above max-steps-per-chart",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        dest="limit_packs",
        metavar="N",
        help="Process only the first N packs (sorted by source path)",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Optional JSON config; CLI flags override file values",
    )
    return parser


def _prep_config_from_args(args: argparse.Namespace) -> config.PrepConfig:
    if args.config_path:
        cfg = config.load_prep_config_json(args.config_path)
    else:
        cfg = config.default_prep_config()

    cfg.input_dir = args.input_dir
    cfg.output_dir = args.output_dir
    cfg.export_mode = config.ExportMode(args.export_mode)
    cfg.max_steps_per_chart = args.max_steps_per_chart
    cfg.export_legacy_txt = args.export_legacy_txt
    cfg.workers = args.workers
    cfg.dry_run = args.dry_run
    cfg.overwrite = args.overwrite
    cfg.allow_over_cap = args.allow_over_cap
    if args.limit_packs is not None:
        cfg.limit_packs = args.limit_packs
    config.validate_prep_config(cfg)
    return cfg


def main(argv: list[str] | None = None) -> int:
    """Run preprocess_dataset CLI."""
    args = _build_parser().parse_args(argv)
    try:
        prep_config = _prep_config_from_args(args)
        report = pipeline.run_preprocess(prep_config)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
