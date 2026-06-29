"""Canonical peak-pick event-F1 evaluation for dense frame onset models."""

import argparse
import json
import pathlib
import sys

import tensorflow as tf

from stepcovnet import config, dense_overfit_eval


def _find_saved_model_path(model_output_dir: str) -> str:
    model_dir = pathlib.Path(model_output_dir)
    keras_files = sorted(
        path.name for path in model_dir.iterdir() if path.name.endswith(".keras")
    )
    if len(keras_files) != 1:
        raise FileNotFoundError(
            f"expected one .keras in {model_output_dir}, found {keras_files!r}",
        )
    return str(model_dir / keras_files[0])


def _resolve_model_path(
    experiment: config.OnsetExperimentConfig, model_path: str
) -> str:
    if model_path:
        return model_path
    return _find_saved_model_path(experiment.run.model_output_dir)


def _default_output_path(experiment: config.OnsetExperimentConfig) -> str:
    return str(pathlib.Path(experiment.run.model_output_dir) / "eval_val_event_f1.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate dense onset model event F1 on the val split.",
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Checkpoint path; default is the sole .keras under run.model_output_dir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Peak-pick confidence threshold; default is run.confidence_threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="JSON report path; default is <model_output_dir>/eval_val_event_f1.json.",
    )
    args = parser.parse_args(argv)

    experiment = config.OnsetExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment, args.model_path)
    output_path = args.output or _default_output_path(experiment)
    threshold = (
        args.threshold
        if args.threshold is not None
        else experiment.run.confidence_threshold
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    report = dense_overfit_eval.eval_dense_val_event_f1(
        model,
        experiment.dataset,
        experiment.model,
        confidence_threshold=threshold,
        min_onset_distance_ms=experiment.run.min_onset_distance_ms,
        tolerance_sec=experiment.run.tolerance_sec,
    )
    report["model_path"] = model_path
    report["config_path"] = args.config

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out_file:
        json.dump(report, out_file, indent=2)

    print(f"wrote {output_path}")
    print(
        f"  songs={report['num_songs']} "
        f"mean_event_f1={report['mean_event_f1']:.4f} "
        f"micro_event_f1={report['micro_event_f1']:.4f} "
        f"micro_timing_match={report['micro_timing_match']:.4f} "
        f"({int(report['timing_match_n_matched'])}/{int(report['timing_match_n_denom'])}) "
        f"@ threshold={threshold}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
