"""Diagnose AR onset overfit checkpoints (per-onset timing errors).

Usage:
    python scripts/debug_ar_onset_overfit.py --config configs/ar/tide.json
    python scripts/debug_ar_onset_overfit.py \\
        --config configs/ar/tide.json \\
        --model_path models_wsl/ar/gate_tide_overfit/ar_onset_model.keras
    python scripts/debug_ar_onset_overfit.py \\
        --config configs/ar/overfit_perfect/run3.json \\
        --model_path models_wsl/ar/perfect_overfit/run3/ar_onset_model.keras \\
        --ar_decode
    python scripts/debug_ar_onset_overfit.py \\
        --config configs/ar/tide.json \\
        --model_path models_wsl/ar/gate_tide_overfit/ar_onset_model.keras \\
        --ar_decode
    python scripts/debug_ar_onset_overfit.py \\
        --config configs/ar/tide.json \\
        --model_path models_wsl/ar/gate_tide_overfit/ar_onset_model.keras \\
        --ar_decode --full-diagnostics

With ``--ar_decode``, top-level ``ar_decode`` metrics use **two-pass** timing
(``decode_autoregressive_gate_with_stats_numpy``). Pass ``--full-diagnostics``
for token trace, gt_timing parity, and other slow ``ar_decode.diagnostics``.

Human-readable progress and a completion summary go to **stderr**; full JSON stays
on **stdout** (pipe-friendly).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

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

from stepcovnet import timing_match
from stepcovnet.onset_ar import config, datasets, inference, targets, trainers

PARSER = argparse.ArgumentParser(description="Debug AR onset overfit checkpoint.")
PARSER.add_argument(
    "--config",
    type=str,
    default="configs/ar/tide.json",
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
PARSER.add_argument(
    "--ar_decode",
    action="store_true",
    help=("Offline free-run decode gate (two-pass timing_match + Hungarian F1)."),
)
PARSER.add_argument(
    "--full-diagnostics",
    action="store_true",
    help=(
        "With --ar_decode, also run slow diagnostics (token trace, gt_timing, "
        "incremental vs parallel pointer, token detokenize)."
    ),
)
PARSER.add_argument(
    "--ar_decode_time_source",
    type=str,
    choices=("pointer_residual", "tokens"),
    default="pointer_residual",
    help="How to convert free-run tokens into onset times (with --ar_decode).",
)
PARSER.add_argument(
    "--json-only",
    action="store_true",
    help="Skip human-readable stderr progress/summary; emit JSON only.",
)
PARSER.add_argument(
    "--token_trace_steps",
    type=int,
    default=20,
    help="With --ar_decode --full-diagnostics, log first N decode steps.",
)


def _log(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def _fmt_f1(value: float) -> str:
    return f"{value:.4f}"


def _fmt_ms(stats: dict[str, float]) -> str:
    return (
        f"mean {stats['mean']:.1f} ms, p50 {stats['p50']:.1f} ms, "
        f"p90 {stats['p90']:.1f} ms, max {stats['max']:.1f} ms"
    )


def _event_f1_line(block: dict[str, object]) -> str:
    tp = int(block["true_positives"])
    fp = int(block["false_positives"])
    fn = int(block["false_negatives"])
    n_gt = int(block.get("n_gt_onsets", tp + fn))
    return (
        f"F1 {_fmt_f1(float(block['event_f1']))} "
        f"({tp} TP, {fp} FP, {fn} FN; {n_gt} GT onsets)"
    )


def _ordered_onset_report(
    pred_times: np.ndarray,
    target_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> dict[str, float | int]:
    report = timing_match.timing_match_report(
        pred_times,
        target_times,
        tolerance_sec=tolerance_sec,
    )
    return {
        "n_matched": int(report["n_matched"]),
        "n_pred": int(report["n_pred"]),
        "n_gt": int(report["n_ref"]),
        "n_ref": int(report["n_ref"]),
        "n_denom": int(report["n_denom"]),
        "rate": float(report["rate"]),
    }


def _ordered_gate_line(block: dict[str, object], *, tol_ms: float) -> str:
    n_matched = int(block["n_matched"])
    n_denom = int(block.get("n_denom", block["n_gt"]))
    rate = float(block["rate"])
    status = "PASS" if rate >= 1.0 - 1e-9 and n_denom > 0 else "FAIL"
    return f"Ordered @ {tol_ms:.0f} ms: {n_matched}/{n_denom} ({rate:.4f}) — {status}"


def _print_teacher_summary(
    report: dict[str, object],
    *,
    tolerance_sec: float,
    quiet: bool,
) -> None:
    tol_ms = tolerance_sec * 1000.0
    ordered = report["ordered_onset_match"]
    assert isinstance(ordered, dict)
    _log("", quiet=quiet)
    _log("=== Teacher-fed gate ===", quiet=quiet)
    _log(f"  {_ordered_gate_line(ordered, tol_ms=tol_ms)}", quiet=quiet)
    _log("", quiet=quiet)
    _log("--- Aux (Hungarian / timing detail) ---", quiet=quiet)
    _log(f"  {_event_f1_line(report)}", quiet=quiet)
    _log(
        f"  Per-step within {tol_ms:.0f} ms: "
        f"{report['n_within_tolerance']}/{report['n_onsets']}",
        quiet=quiet,
    )
    _log(
        f"  Timing: {_fmt_ms(report['abs_error_ms'])}",
        quiet=quiet,
    )
    _log(
        "  Patch errors: "
        f"{report['n_patch_wrong']} | "
        f"patch OK, timing wrong: {report['n_patch_ok_timing_wrong']}",
        quiet=quiet,
    )


def _print_ar_decode_summary(
    ar_decode: dict[str, object],
    *,
    tolerance_sec: float,
    quiet: bool,
) -> None:
    tol_ms = tolerance_sec * 1000.0
    ordered = ar_decode["ordered_onset_match"]
    assert isinstance(ordered, dict)
    diagnostics = ar_decode.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    _log("", quiet=quiet)
    _log("=== Free-run AR gate (two-pass) ===", quiet=quiet)
    _log(
        f"  Decode length: {ar_decode['ar_decode_length']} | "
        f"EOS: {'yes' if ar_decode['stopped_on_eos'] else 'no'}",
        quiet=quiet,
    )
    _log(f"  {_ordered_gate_line(ordered, tol_ms=tol_ms)}", quiet=quiet)
    _log("", quiet=quiet)
    _log("--- Aux ---", quiet=quiet)
    _log(f"  Hungarian F1: {_event_f1_line(ar_decode)}", quiet=quiet)

    inc = diagnostics.get("incremental_pointer_residual")
    if isinstance(inc, dict):
        _log(
            f"  Incremental ptr+residual: {_event_f1_line(inc)}",
            quiet=quiet,
        )

    gt_timing = diagnostics.get("gt_timing")
    if isinstance(gt_timing, dict):
        gt_par = gt_timing.get("gt_parallel")
        gt_inc = gt_timing.get("gt_incremental")
        if isinstance(gt_par, dict):
            _log(
                f"  GT tokens + parallel pointer: {_event_f1_line(gt_par)}",
                quiet=quiet,
            )
        if isinstance(gt_inc, dict):
            _log(
                f"  GT tokens + incremental pointer: {_event_f1_line(gt_inc)}",
                quiet=quiet,
            )

    token_detok = diagnostics.get("token_detokenize")
    if isinstance(token_detok, dict):
        _log(
            f"  Token detokenize only: {_event_f1_line(token_detok)}",
            quiet=quiet,
        )

    first_mismatch = diagnostics.get("first_mismatch_step")
    eos_at = diagnostics.get("eos_at_step")
    trace = diagnostics.get("token_trace")
    if isinstance(trace, list) and trace:
        n_match = sum(1 for row in trace if row.get("match"))
        mismatch_note = (
            f"first mismatch at step {first_mismatch}"
            if first_mismatch is not None
            else "all traced steps matched"
        )
        eos_note = (
            f"early EOS at step {eos_at}"
            if eos_at is not None
            else "no early EOS in trace"
        )
        _log(
            f"  Token trace ({len(trace)} steps): "
            f"{n_match}/{len(trace)} matched | {mismatch_note} | {eos_note}",
            quiet=quiet,
        )


def _print_debug_summary(
    report: dict[str, object],
    *,
    model_path: pathlib.Path,
    tolerance_sec: float,
    ar_decode: bool,
    quiet: bool,
) -> None:
    _log("=== AR overfit debug ===", quiet=quiet)
    _log(f"Model: {model_path}", quiet=quiet)
    ordered = report["ordered_onset_match"]
    assert isinstance(ordered, dict)
    _log(
        f"Primary metric: ordered onset match @ {tolerance_sec * 1000.0:.0f} ms "
        f"({int(ordered['n_matched'])}/{int(ordered['n_denom'])})",
        quiet=quiet,
    )
    _print_teacher_summary(report, tolerance_sec=tolerance_sec, quiet=quiet)
    if ar_decode:
        ar_block = report.get("ar_decode")
        if isinstance(ar_block, dict):
            _print_ar_decode_summary(
                ar_block,
                tolerance_sec=tolerance_sec,
                quiet=quiet,
            )
    _log("", quiet=quiet)
    _log("Full JSON report on stdout.", quiet=quiet)


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
    ordered = _ordered_onset_report(
        pred_times,
        target_times[step_indices],
        tolerance_sec=tolerance_sec,
    )

    return {
        "n_onsets": int(pred_times.size),
        "timing_match": ordered,
        "ordered_onset_match": ordered,
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


def _event_f1_report(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> dict[str, float | int]:
    pred_mask = np.ones(pred_times.shape, dtype=np.float32)
    gt_mask = np.ones(gt_times.shape, dtype=np.float32)
    tp, fp, fn = trainers._ar_event_onset_counts_numpy(  # noqa: SLF001
        pred_times,
        pred_mask,
        gt_times,
        gt_mask,
        tolerance_sec=tolerance_sec,
    )
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    event_f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return {
        "n_pred_onsets": int(pred_times.size),
        "n_gt_onsets": int(gt_times.size),
        "event_f1": float(event_f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def _gt_timing_diagnostic(
    model: tf.keras.Model,
    batch_np: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
    gt_times: np.ndarray,
    tolerance_sec: float,
) -> dict[str, object]:
    """Compare pointer+residual times under GT parallel vs GT incremental decode."""
    mert = batch_np["mert_patches"]
    patch_mask = batch_np["patch_mask"]
    onset_mask = batch_np["onset_step_mask"][0] > 0.5
    gt_tokens = batch_np["decoder_target_ids"][0][onset_mask]

    parallel_times = inference.decode_parallel_pointer_times_numpy(
        model,
        mert,
        patch_mask,
        gt_tokens,
        experiment_config=experiment_config,
    )
    incremental_times = inference.decode_gt_incremental_pointer_times_numpy(
        model,
        mert,
        patch_mask,
        batch_np["decoder_input_ids"][0],
        batch_np["onset_step_mask"][0],
        experiment_config=experiment_config,
    )
    return {
        "gt_parallel": _event_f1_report(
            parallel_times,
            gt_times,
            tolerance_sec=tolerance_sec,
        ),
        "gt_incremental": _event_f1_report(
            incremental_times,
            gt_times,
            tolerance_sec=tolerance_sec,
        ),
    }


def _ar_decode_report(
    model: tf.keras.Model,
    batch_np: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
    token_trace_steps: int,
    time_source: inference.ArTimeSource = "pointer_residual",
    compare_time_sources: bool = False,
) -> dict[str, object]:
    """Two-pass gate metrics; incremental timing under ``diagnostics``."""
    run_config = experiment_config.run
    model_config = experiment_config.model
    max_decoder_len = experiment_config.max_decoder_len()
    mert = batch_np["mert_patches"]
    patch_mask = batch_np["patch_mask"]
    gt_times = batch_np["gt_times"][0][batch_np["gt_mask"][0] > 0.5]
    gt_target = batch_np["decoder_target_ids"][0]
    gt_mask = batch_np["decoder_mask"][0]

    decode_kwargs = {
        "max_decoder_len": max_decoder_len,
        "patch_frames": model_config.patch_frames,
        "hop_sec": experiment_config.dataset.hop_sec,
        "experiment_config": experiment_config,
    }

    incremental_stats = inference.decode_autoregressive_with_stats_numpy(
        model,
        mert,
        patch_mask,
        time_source=time_source,
        **decode_kwargs,
    )
    gate_stats = inference.decode_autoregressive_two_pass_with_stats_numpy(
        model,
        mert,
        patch_mask,
        token_pass=incremental_stats,
        **decode_kwargs,
    )

    diagnostics: dict[str, object] = {
        "incremental_pointer_residual": {
            "time_source": time_source,
            "ar_decode_length": incremental_stats.n_forward_steps,
            "stopped_on_eos": incremental_stats.stopped_on_eos,
            **_event_f1_report(
                incremental_stats.times,
                gt_times,
                tolerance_sec=run_config.tolerance_sec,
            ),
        },
        "gt_timing": _gt_timing_diagnostic(
            model,
            batch_np,
            experiment_config=experiment_config,
            gt_times=gt_times,
            tolerance_sec=run_config.tolerance_sec,
        ),
    }

    if compare_time_sources and incremental_stats.onset_token_ids is not None:
        token_times = inference.decode_onset_tokens_to_times(
            incremental_stats.onset_token_ids,
            experiment_config=experiment_config,
            patch_mask=patch_mask,
        )
        diagnostics["token_detokenize"] = _event_f1_report(
            token_times,
            gt_times,
            tolerance_sec=run_config.tolerance_sec,
        )

    ordered = _ordered_onset_report(
        gate_stats.times,
        gt_times,
        tolerance_sec=run_config.tolerance_sec,
    )
    report: dict[str, object] = {
        "timing_mode": "two_pass",
        "timing_match": ordered,
        "ordered_onset_match": ordered,
        "ar_decode_length": gate_stats.n_forward_steps,
        "stopped_on_eos": gate_stats.stopped_on_eos,
        **_event_f1_report(
            gate_stats.times,
            gt_times,
            tolerance_sec=run_config.tolerance_sec,
        ),
        "diagnostics": diagnostics,
    }

    if token_trace_steps <= 0:
        return report

    memory, pm_b = inference.get_encoder_memory_numpy(
        model,
        mert,
        patch_mask,
        experiment_config,
    )
    _, decoder = inference.get_inference_encoder_decoder(model, experiment_config)
    dec_in = np.zeros((1, max_decoder_len), dtype=np.int32)
    dec_mask = np.zeros((1, max_decoder_len), dtype=np.float32)
    dec_in[0, 0] = targets.BOS_ID
    dec_mask[0, 0] = 1.0

    trace: list[dict[str, int | bool]] = []
    eos_at: int | None = None
    n_valid = int((gt_mask > 0.5).sum())
    for cur_len in range(1, min(n_valid + 2, token_trace_steps + 1)):
        outputs = decoder(
            {
                "encoder_memory": memory,
                "patch_mask": pm_b,
                "decoder_input_ids": dec_in,
                "decoder_mask": dec_mask,
            },
            training=False,
        )
        pos = cur_len - 1
        pred = int(np.argmax(outputs["token_logits"].numpy()[0, pos]))
        tgt = int(gt_target[pos])
        trace.append(
            {
                "step": pos,
                "input_id": int(dec_in[0, pos]),
                "pred_id": pred,
                "target_id": tgt,
                "match": pred == tgt,
            },
        )
        if pred == targets.EOS_ID:
            eos_at = pos
            break
        dec_in[0, cur_len] = pred
        dec_mask[0, cur_len] = 1.0

    report["diagnostics"]["token_trace"] = trace
    report["diagnostics"]["eos_at_step"] = eos_at
    report["diagnostics"]["first_mismatch_step"] = next(
        (row["step"] for row in trace if not row["match"]),
        None,
    )
    return report


def _ar_decode_gate_only_report(
    model: tf.keras.Model,
    batch_np: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
) -> dict[str, object]:
    """Two-pass gate metrics only (no diagnostics overhead)."""
    run_config = experiment_config.run
    model_config = experiment_config.model
    mert = batch_np["mert_patches"]
    patch_mask = batch_np["patch_mask"]
    gt_times = batch_np["gt_times"][0][batch_np["gt_mask"][0] > 0.5]

    gate_stats = inference.decode_autoregressive_gate_with_stats_numpy(
        model,
        mert,
        patch_mask,
        max_decoder_len=experiment_config.max_decoder_len(),
        patch_frames=model_config.patch_frames,
        hop_sec=experiment_config.dataset.hop_sec,
        experiment_config=experiment_config,
    )
    ordered = _ordered_onset_report(
        gate_stats.times,
        gt_times,
        tolerance_sec=run_config.tolerance_sec,
    )
    return {
        "timing_mode": "two_pass",
        "timing_match": ordered,
        "ordered_onset_match": ordered,
        "ar_decode_length": gate_stats.n_forward_steps,
        "stopped_on_eos": gate_stats.stopped_on_eos,
        **_event_f1_report(
            gate_stats.times,
            gt_times,
            tolerance_sec=run_config.tolerance_sec,
        ),
    }


def main() -> int:
    args = PARSER.parse_args()
    quiet = args.json_only
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment_config, args.model_path)
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    _log(f"Loading checkpoint: {model_path}", quiet=quiet)
    batch_np = datasets.sample_to_training_batch(
        datasets.load_overfit_sample(experiment_config),
        experiment_config,
    )
    batch_tf = {key: tf.constant(value) for key, value in batch_np.items()}

    model = tf.keras.models.load_model(str(model_path), compile=False)
    t0 = time.perf_counter()
    _log("Running teacher-fed eval...", quiet=quiet)
    outputs = model(_model_inputs(batch_tf), training=False)
    report = _diagnose_batch(outputs, batch_np, experiment_config=experiment_config)
    report["model_path"] = str(model_path)
    report["worst_onsets"] = _worst_onsets(report, args.worst_k)
    if args.ar_decode:
        if args.full_diagnostics:
            _log(
                "Running free-run AR decode (two-pass, full diagnostics)...",
                quiet=quiet,
            )
            report["ar_decode"] = _ar_decode_report(
                model,
                batch_np,
                experiment_config=experiment_config,
                token_trace_steps=args.token_trace_steps,
                time_source=args.ar_decode_time_source,
                compare_time_sources=True,
            )
        else:
            _log("Running free-run AR gate (two-pass)...", quiet=quiet)
            report["ar_decode"] = _ar_decode_gate_only_report(
                model,
                batch_np,
                experiment_config=experiment_config,
            )
    elapsed_sec = time.perf_counter() - t0
    report["eval_elapsed_sec"] = round(elapsed_sec, 3)
    for key in list(report):
        if key.startswith("_"):
            del report[key]
    _print_debug_summary(
        report,
        model_path=model_path,
        tolerance_sec=experiment_config.run.tolerance_sec,
        ar_decode=args.ar_decode,
        quiet=quiet,
    )
    if not quiet:
        _log(f"Eval wall time: {elapsed_sec:.2f} s", quiet=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
