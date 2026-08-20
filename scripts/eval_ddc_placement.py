#!/usr/bin/env python
"""Evaluate DDC placement F-score_c / F-score_m plus parallel ``timing_match``."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/eval_ddc_placement.py")

import tensorflow as tf

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddc import config, datasets, evaluation, models, slot48


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DDC C-LSTM placement: Hamming ±20 ms F1 and timing_match."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--split",
        default=training_index.SPLIT_VAL,
        help="Manifest split to score (default: val)",
    )
    parser.add_argument(
        "--slot48",
        action="store_true",
        help="Also snap Hamming peaks onto M-slot48 (POST conversion).",
    )
    return parser


def _resolve_model_path(
    experiment: config.PlacementExperimentConfig, model_path: str
) -> str:
    """Return an explicit checkpoint path or the saved experiment artifact.

    Args:
        experiment: Placement experiment config.
        model_path: Optional override path.

    Returns:
        Path to a ``.keras`` file.

    Raises:
        FileNotFoundError: If no unique checkpoint exists.
    """
    if model_path:
        return model_path
    model_dir = pathlib.Path(experiment.run.model_output_dir)
    keras_files = sorted(path.name for path in model_dir.glob("*.keras"))
    if len(keras_files) != 1:
        raise FileNotFoundError(
            f"expected one .keras in {model_dir}, found {keras_files!r}"
        )
    return str(model_dir / keras_files[0])


def main(argv: list[str] | None = None) -> int:
    """Run DDC placement evaluation.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    wsl_gpu.guard_tensorflow_gpu_job(__file__)
    experiment = config.PlacementExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment, args.model_path)
    output_path = args.output or str(
        pathlib.Path(experiment.run.model_output_dir) / "eval_val_ddc.json"
    )
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"DdcPerFrameCNN": models.DdcPerFrameCNN},
    )
    charts = datasets.load_split_charts(experiment.dataset, args.split)
    report = evaluation.evaluate_placement(model, charts, seed=experiment.run.seed)
    pathlib.Path(output_path).write_text(
        json.dumps(report.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), indent=2))
    if args.slot48:
        slot_report = slot48.evaluate_peak_times_as_slot48(
            charts,
            [row.pred_times for row in report.charts],
            seed=experiment.run.seed,
        )
        slot_payload = slot48.report_as_dict(slot_report)
        slot_path = str(
            pathlib.Path(experiment.run.model_output_dir) / "eval_val_slot48.json"
        )
        pathlib.Path(slot_path).write_text(
            json.dumps(slot_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(slot_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
