#!/usr/bin/env python
"""Evaluate DDCL placement ``M-slot48`` F1 at 0.5 and max-F1."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/eval_ddcl_placement.py")

import tensorflow as tf

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, datasets, evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DDCL ConvLSTM placement: M-slot48 F1.",
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
        "--diagnose_timing",
        action="store_true",
        help=(
            "Score counterfactual slot masks and bipartite event F1 alongside "
            "timing_match."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Slot occupancy cutoff for --diagnose_timing (default: 0.5).",
    )
    return parser


def _resolve_model_path(
    experiment: config.DdclExperimentConfig, model_path: str
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
    named = model_dir / f"{experiment.run.model_name}.keras"
    preferred = (
        model_dir / "best.keras",
        named,
        model_dir / "last.keras",
    )
    for candidate in preferred:
        if candidate.is_file():
            return str(candidate)
    keras_files = sorted(path.name for path in model_dir.glob("*.keras"))
    if len(keras_files) != 1:
        raise FileNotFoundError(
            f"expected best.keras, {named.name}, last.keras, or one .keras in "
            f"{model_dir}, found {keras_files!r}"
        )
    return str(model_dir / keras_files[0])


def main(argv: list[str] | None = None) -> int:
    """Run DDCL placement evaluation.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    wsl_gpu.guard_tensorflow_gpu_job(__file__)
    experiment = config.DdclExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment, args.model_path)
    default_filename = (
        "eval_val_timing_diag.json" if args.diagnose_timing else "eval_val_slot48.json"
    )
    output_path = args.output or str(
        pathlib.Path(experiment.run.model_output_dir) / default_filename
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    charts = datasets.load_split_charts(experiment.dataset, args.split)
    if args.diagnose_timing:
        diag_report = evaluation.diagnose_decoded_timing(
            model,
            charts,
            threshold=args.threshold,
            seed=experiment.run.seed,
        )
        payload = diag_report.as_dict()
        payload["model_path"] = model_path
        payload["config_path"] = args.config
    else:
        report = evaluation.evaluate_slot48(model, charts, seed=experiment.run.seed)
        payload = report.as_dict()
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
