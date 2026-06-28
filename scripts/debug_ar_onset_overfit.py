"""Diagnose AR onset overfit checkpoints (per-onset timing errors).

Usage:
    python scripts/debug_ar_onset_overfit.py --config configs/onset_ar_tide.json
    python scripts/debug_ar_onset_overfit.py \\
        --config configs/onset_ar_tide.json \\
        --model_path models_wsl/ar_tide_overfit_gate_v5/ar_onset_model.keras
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from stepcovnet import wsl_gpu

SCRIPT_REL = "scripts/debug_ar_onset_overfit.py"


def _bootstrap_wsl_gpu() -> None:
    script_path = str(pathlib.Path(__file__).resolve())
    argv = [script_path, *sys.argv[1:]]
    wsl_gpu.maybe_dispatch_for_training(SCRIPT_REL, argv)
    wsl_gpu.reexec_with_tensorflow_gpu_env_if_needed(argv)


_bootstrap_wsl_gpu()

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, datasets, inference, trainers

PARSER = argparse.ArgumentParser(description="Debug AR onset overfit checkpoint.")
PARSER.add_argument(
    "--config",
    type=str,
    default="configs/onset_ar_tide.json",
    help="AR experiment config JSON.",
)
PARSER.add_argument(
    "--model_path",
    type=str,
    default="",
    help="Checkpoint path (default: run.model_output_dir/ar_onset_model.keras).",
)
PARSER.add_argument(
    "--worst_k",
    type=int,
    default=20,
    help="Number of largest-error onsets to include in the report.",
)


def _resolve_model_path(
    experiment_config: config.ArExperimentConfig,
    model_path: str,
) -> pathlib.Path:
    if model_path:
        return pathlib.Path(model_path)
    output_dir = experiment_config.run.model_output_dir
    if not output_dir:
        raise ValueError(
            "run.model_output_dir is required when --model_path is omitted"
        )
    return pathlib.Path(output_dir) / "ar_onset_model.keras"


def _model_inputs(batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    return {
        "mert_patches": batch["mert_patches"],
        "patch_mask": batch["patch_mask"],
        "decoder_input_ids": batch["decoder_input_ids"],
        "decoder_mask": batch["decoder_mask"],
    }


def _diagnose_batch(
    outputs: dict[str, tf.Tensor],
    batch: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
) -> dict[str, object]:
    run_config = experiment_config.run
    model_config = experiment_config.model
    hop_sec = experiment_config.dataset.hop_sec
    patch_frames = model_config.patch_frames
    tolerance_sec = run_config.tolerance_sec

    pointer_logits = outputs["pointer_logits"].numpy()[0]
    residual_sec = outputs["residual_sec"].numpy()[0]
    onset_mask = batch["onset_step_mask"][0] > 0.5
    target_patches = batch["target_patch_indices"][0]
    target_residual = batch["target_residual_sec"][0]
    target_times = batch["target_times"][0]
    gt_times = batch["gt_times"][0][batch["gt_mask"][0] > 0.5]

    pred_times = inference.decode_teacher_fed_times_numpy(
        pointer_logits,
        residual_sec,
        batch["onset_step_mask"][0],
        patch_frames=patch_frames,
        hop_sec=hop_sec,
    )
    pred_patch = np.argmax(pointer_logits, axis=-1)

    step_indices = np.flatnonzero(onset_mask)
    abs_err_sec = np.abs(pred_times - target_times[step_indices])
    abs_err_ms = abs_err_sec * 1000.0
    patch_ok = pred_patch[step_indices] == target_patches[step_indices]
    residual_err_ms = (
        np.abs(residual_sec[step_indices] - target_residual[step_indices]) * 1000.0
    )

    tp, fp, fn = trainers._ar_event_onset_counts_numpy(  # noqa: SLF001
        pred_times,
        np.ones(pred_times.shape, dtype=np.float32),
        gt_times,
        np.ones(gt_times.shape, dtype=np.float32),
        tolerance_sec=tolerance_sec,
    )
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    event_f1 = 2.0 * precision * recall / (precision + recall + 1e-9)

    within_tol = int(np.sum(abs_err_sec <= tolerance_sec))
    patch_wrong = int(np.sum(~patch_ok))
    patch_ok_residual_bad = int(
        np.sum(patch_ok & (abs_err_sec > tolerance_sec)),
    )

    return {
        "n_onsets": int(pred_times.size),
        "event_f1": float(event_f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "abs_error_ms": {
            "p50": float(np.percentile(abs_err_ms, 50)),
            "p90": float(np.percentile(abs_err_ms, 90)),
            "p99": float(np.percentile(abs_err_ms, 99)),
            "max": float(np.max(abs_err_ms)),
            "mean": float(np.mean(abs_err_ms)),
        },
        "n_within_tolerance": within_tol,
        "n_patch_wrong": patch_wrong,
        "n_patch_ok_timing_wrong": patch_ok_residual_bad,
        "residual_error_ms": {
            "p50": float(np.percentile(residual_err_ms, 50)),
            "p90": float(np.percentile(residual_err_ms, 90)),
            "max": float(np.max(residual_err_ms)),
            "mean": float(np.mean(residual_err_ms)),
        },
        "_step_indices": step_indices,
        "_abs_err_ms": abs_err_ms,
        "_patch_ok": patch_ok,
        "_residual_err_ms": residual_err_ms,
        "_pred_times": pred_times,
        "_target_times": target_times[step_indices],
        "_pred_patches": pred_patch[step_indices],
        "_target_patches": target_patches[step_indices],
        "_pred_residual_ms": residual_sec[step_indices] * 1000.0,
        "_target_residual_ms": target_residual[step_indices] * 1000.0,
    }


def _worst_onsets(report: dict[str, object], worst_k: int) -> list[dict[str, float]]:
    order = np.argsort(-np.asarray(report["_abs_err_ms"], dtype=np.float64))
    worst: list[dict[str, float]] = []
    for rank in order[:worst_k]:
        idx = int(rank)
        worst.append(
            {
                "step": int(report["_step_indices"][idx]),
                "gt_sec": float(report["_target_times"][idx]),
                "pred_sec": float(report["_pred_times"][idx]),
                "abs_err_ms": float(report["_abs_err_ms"][idx]),
                "patch_ok": bool(report["_patch_ok"][idx]),
                "pred_patch": int(report["_pred_patches"][idx]),
                "target_patch": int(report["_target_patches"][idx]),
                "pred_residual_ms": float(report["_pred_residual_ms"][idx]),
                "target_residual_ms": float(report["_target_residual_ms"][idx]),
                "residual_err_ms": float(report["_residual_err_ms"][idx]),
            },
        )
    return worst


def main() -> int:
    args = PARSER.parse_args()
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment_config, args.model_path)
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    batch_np = datasets.sample_to_training_batch(
        datasets.load_overfit_sample(experiment_config),
        experiment_config,
    )
    batch_tf = {key: tf.constant(value) for key, value in batch_np.items()}

    model = tf.keras.models.load_model(str(model_path), compile=False)
    outputs = model(_model_inputs(batch_tf), training=False)
    report = _diagnose_batch(outputs, batch_np, experiment_config=experiment_config)
    report["model_path"] = str(model_path)
    report["worst_onsets"] = _worst_onsets(report, args.worst_k)
    for key in list(report):
        if key.startswith("_"):
            del report[key]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
