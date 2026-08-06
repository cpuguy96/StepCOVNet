"""Offline evaluation for AR onset checkpoints (teacher-fed and optional free-run).

Scores saved models on a single overfit song or a train/val manifest split:
ordered timing match (primary), Hungarian F1 (aux), null-baseline skill, and
optional two-pass free-run decode with EOS trace.

Usage:
    python scripts/eval_ar_onset_offline.py --config configs/ar/tide_overfit.json
    python scripts/eval_ar_onset_offline.py \\
        --config configs/ar/tide_overfit.json --ar_decode
    python scripts/eval_ar_onset_offline.py \\
        --config configs/ar/ladder_50t_50v.json --split val --ar_decode
    python scripts/eval_ar_onset_offline.py \\
        --config configs/ar/versions/tide_overfit/v3.json \\
        --model_path models_wsl/ar/perfect_overfit/run2/ar_onset_model.keras \\
        --ar_decode
    python scripts/eval_ar_onset_offline.py \\
        --config configs/ar/tide_overfit.json --ar_decode --full-diagnostics

With ``--ar_decode``, top-level ``ar_decode.ordered_onset_match`` uses **two-pass**
timing vs training ``target_times`` (primary). Raw chart ``gt_times`` is logged as
``chart_ordered_onset_match`` and Hungarian F1 (aux). Pass ``--full-diagnostics``
for token trace, gt_timing parity, and other slow ``ar_decode.diagnostics``.

Free-run length is reported as ``ar_decode.eos_trace`` (per-step ``<EOS>``
probability summary). ``--ar_decode_eos_logit_bias`` and
``--ar_decode_min_onset_tokens`` constrain when the decoder may stop, so an
under-generating checkpoint can be probed without retraining:

    python scripts/eval_ar_onset_offline.py \\
        --config configs/ar/scale_200t_50v.json --split val --ar_decode \\
        --ar_decode_min_onset_tokens 400

``--split train|val`` micro-averages metrics over the manifest split.
``--split overfit`` (default) evaluates the config's single overfit song.

With ``--ar_decode`` on multi-song splits, teacher-fed metrics are computed
first; free-run is **skipped** when the teacher gate fails (near-zero timing
match and no positive null skill). Use ``--force-ar-decode`` to run free-run
anyway. Overfit runs require a perfect teacher gate (same bar as tide iter).

Human-readable progress and a completion summary go to **stderr**; full JSON stays
on **stdout** (pipe-friendly).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

from stepcovnet import wsl_gpu

SCRIPT_REL = "scripts/eval_ar_onset_offline.py"

wsl_gpu.bootstrap_gpu_script(SCRIPT_REL)

import numpy as np
import tensorflow as tf

from stepcovnet import onset_metric_names as mn
from stepcovnet import onset_null_baseline, timing_match
from stepcovnet.onset_ar import (
    config,
    datasets,
    density_presets,
    inference,
    targets,
    trainers,
)

TEACHER_PERFECT_EPS = 1e-9
DEFAULT_MIN_TEACHER_RATE = 0.01

PARSER = argparse.ArgumentParser(
    description="Offline AR onset checkpoint eval (teacher-fed and optional free-run)."
)
PARSER.add_argument(
    "--config",
    type=str,
    default="configs/ar/tide_overfit.json",
    help="AR experiment config JSON.",
)
PARSER.add_argument(
    "--model_path",
    type=str,
    default="",
    help="Checkpoint path (default: run.model_output_dir/ar_onset_model.keras).",
)
PARSER.add_argument(
    "--split",
    type=str,
    choices=("overfit", "train", "val"),
    default="overfit",
    help="overfit: single-song path; train/val: micro-avg over manifest split.",
)
PARSER.add_argument(
    "--limit",
    type=int,
    default=-1,
    help="With --split train|val, evaluate at most N songs (-1 = all).",
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
        "incremental vs parallel pointer, token detokenize). "
        "Single-song overfit only."
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
    "--ar_decode_eos_logit_bias",
    type=float,
    default=0.0,
    help=(
        "Additive bias on the <EOS> logit during free-run decode; negative "
        "values discourage stopping (with --ar_decode)."
    ),
)
PARSER.add_argument(
    "--ar_decode_min_onset_tokens",
    type=int,
    default=0,
    help=(
        "Suppress <EOS> during free-run decode until this many onset tokens "
        "have been emitted (with --ar_decode)."
    ),
)
PARSER.add_argument(
    "--force-ar-decode",
    action="store_true",
    help=(
        "With --ar_decode, run free-run even when the teacher-fed preflight gate "
        "fails (overfit: not perfect; val/train: low timing_match / negative null skill)."
    ),
)
PARSER.add_argument(
    "--min-teacher-rate",
    type=float,
    default=0.01,
    help=(
        "With --split val|train and --ar_decode, minimum micro-averaged teacher "
        "timing_match rate before free-run (default 0.01). Ignored when null skill "
        "is positive."
    ),
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
PARSER.add_argument(
    "--difficulty-tier",
    type=str,
    default="",
    help=(
        "Customer difficulty for free-run decode (beginner, easy, medium, …). "
        "Overrides oracle chart density with configs/ar/density_presets.json."
    ),
)
PARSER.add_argument(
    "--density-presets-path",
    type=str,
    default="",
    help="Optional density_presets.json (default: configs/ar/density_presets.json).",
)


def _apply_customer_density_tier(
    batch_np: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
    difficulty_tier: str,
    density_presets_path: str,
) -> float | None:
    """Replace oracle ``density_scalar`` with a customer tier preset when requested."""
    tier = str(difficulty_tier).strip()
    if not tier or not config.density_conditioning_active(experiment_config.model):
        return None
    preset_path = density_presets_path or density_presets.DEFAULT_PRESETS_PATH
    presets = density_presets.load_density_presets(preset_path)
    duration = float(np.asarray(batch_np["duration"]).reshape(-1)[0])
    scalar = density_presets.customer_density_scalar(
        tier,
        model_config=experiment_config.model,
        presets=presets,
        duration_sec=duration,
    )
    batch_np["density_scalar"] = np.asarray([scalar], dtype=np.float32)
    return scalar


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


def _eos_trace_line(block: dict[str, object]) -> str:
    """Return a one-line ``<EOS>`` probability summary for a song or a split.

    Args:
        block: Per-song ``eos_trace`` summary or its split-level aggregate.
    """
    if "first_mean" in block:
        return (
            f"EOS prob (mean over {int(block['n_songs'])} songs): "
            f"first {float(block['first_mean']):.4f} | "
            f"final {float(block['final_mean']):.4f} | "
            f"max {float(block['max_mean']):.4f} | "
            f"songs reaching 0.5: {int(block['n_songs_ge_half'])}"
        )
    ge_half = block.get("first_step_ge_half")
    ge_half_text = "never" if ge_half is None else f"step {int(ge_half)}"
    return (
        f"EOS prob over {int(block['n_steps'])} steps: "
        f"first {float(block['first']):.4f} | "
        f"final {float(block['final']):.4f} | "
        f"max {float(block['max']):.4f} | "
        f"reaches 0.5: {ge_half_text}"
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


def _teacher_ordered_block(report: dict[str, object]) -> dict[str, object] | None:
    block = report.get(mn.TIMING_MATCH_TEACHER, report.get("ordered_onset_match"))
    return block if isinstance(block, dict) else None


def _teacher_gate_passes(
    report: dict[str, object],
    *,
    split: str,
    min_teacher_rate: float = DEFAULT_MIN_TEACHER_RATE,
) -> tuple[bool, str]:
    """Return whether teacher-fed metrics justify an expensive free-run decode."""
    ordered = _teacher_ordered_block(report)
    if ordered is None:
        return False, "missing teacher ordered_onset_match"

    n_matched = int(ordered.get("n_matched", 0))
    n_denom = int(ordered.get("n_denom", 0))
    rate = float(ordered.get("rate", 0.0))
    summary = f"{n_matched}/{n_denom} ({rate:.4f})"

    if split == "overfit":
        if n_denom > 0 and n_matched == n_denom and rate >= 1.0 - TEACHER_PERFECT_EPS:
            return True, ""
        return False, f"teacher ordered gate not perfect ({summary})"

    null_block = report.get("null_baseline")
    if isinstance(null_block, dict):
        for key in ("skill_timing_match", "skill_event_f1"):
            skill = null_block.get(key)
            if skill is not None and float(skill) > 0.0:
                return True, ""

    if n_denom > 0 and rate >= min_teacher_rate:
        return True, ""

    skill_tm = (
        null_block.get("skill_timing_match") if isinstance(null_block, dict) else None
    )
    skill_f1 = (
        null_block.get("skill_event_f1") if isinstance(null_block, dict) else None
    )
    return False, (
        f"teacher timing_match {summary} below min {min_teacher_rate:.4f} "
        f"and null skill not positive "
        f"(skill_timing_match={skill_tm}, skill_event_f1={skill_f1})"
    )


def _mark_ar_decode_skipped(
    report: dict[str, object],
    *,
    reason: str,
) -> None:
    report["teacher_gate_failed"] = True
    report["ar_decode_skipped"] = True
    report["teacher_gate_reason"] = reason


def _onset_reference_times(
    batch_np: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Training labels and raw chart times in decoder onset order."""
    onset_mask = batch_np["onset_step_mask"][0] > 0.5
    step_indices = np.flatnonzero(onset_mask)
    target = batch_np["target_times"][0][step_indices]
    chart = batch_np["gt_times"][0][batch_np["gt_mask"][0] > 0.5]
    return target, chart


def _null_baseline_block(
    batch_np: dict[str, np.ndarray],
    *,
    n_pred: int,
    tolerance_sec: float,
    hop_sec: float,
) -> list[dict[str, object]]:
    """Score audio-blind baselines for one song at the model's prediction count.

    Args:
        batch_np: Single-song batch arrays (supplies GT times and duration).
        n_pred: Onsets the model emitted for this song.
        tolerance_sec: Match tolerance in seconds.
        hop_sec: Prediction grid the baselines snap to.

    Returns:
        JSON-friendly per-kind match counts.
    """
    _, chart_times = _onset_reference_times(batch_np)
    duration = float(np.asarray(batch_np["duration"]).reshape(-1)[0])
    counts = onset_null_baseline.null_counts_for_song(
        chart_times,
        duration_sec=duration,
        n_pred=int(n_pred),
        tolerance_sec=tolerance_sec,
        hop_sec=hop_sec,
    )
    return [dataclasses.asdict(row) for row in counts]


def _aggregate_null_baselines(
    rows: list[list[dict[str, object]]],
    *,
    model_event_f1: float,
    model_timing_match: float,
) -> dict[str, object]:
    """Micro-average baseline counts and score the model against the hardest one.

    Args:
        rows: Per-song lists produced by :func:`_null_baseline_block`.
        model_event_f1: Split-level Hungarian F1 the model achieved.
        model_timing_match: Split-level ordered timing-match rate.
    """
    parsed = [
        [onset_null_baseline.NullCounts(**entry) for entry in row]  # type: ignore[arg-type]
        for row in rows
    ]
    aggregated = onset_null_baseline.aggregate_null_counts(parsed)
    kind, floor = onset_null_baseline.strongest_null(aggregated, metric="event_f1")
    tm_kind, tm_floor = onset_null_baseline.strongest_null(
        aggregated,
        metric="timing_match",
    )
    return {
        "by_kind": aggregated,
        "strongest": {"event_f1": kind, "timing_match": tm_kind},
        "event_f1_floor": floor,
        "timing_match_floor": tm_floor,
        "skill_event_f1": onset_null_baseline.skill_over_null(model_event_f1, floor),
        "skill_timing_match": onset_null_baseline.skill_over_null(
            model_timing_match,
            tm_floor,
        ),
    }


def _eos_trace_stats(eos_prob_trace: np.ndarray | None) -> dict[str, object] | None:
    """Summarize the per-step ``<EOS>`` probability trace of one free-run decode.

    Args:
        eos_prob_trace: Per-step ``<EOS>`` probability, or ``None`` if unavailable.

    Returns:
        Summary of the trace, or ``None`` when no steps were recorded.
    """
    if eos_prob_trace is None:
        return None
    trace = np.asarray(eos_prob_trace, dtype=np.float64).reshape(-1)
    if trace.size == 0:
        return None
    above_half = np.flatnonzero(trace >= 0.5)
    return {
        "n_steps": int(trace.size),
        "first": float(trace[0]),
        "final": float(trace[-1]),
        "mean": float(trace.mean()),
        "max": float(trace.max()),
        "first_step_ge_half": int(above_half[0]) if above_half.size else None,
    }


def _aggregate_eos_traces(eos_traces: list[dict[str, object]]) -> dict[str, object]:
    """Average per-song ``<EOS>`` trace summaries across a split.

    Args:
        eos_traces: Per-song summaries produced by ``_eos_trace_stats``.
    """

    def _mean(key: str) -> float:
        values = [float(t[key]) for t in eos_traces if isinstance(t.get(key), float)]
        return float(np.mean(values)) if values else 0.0

    return {
        "n_songs": len(eos_traces),
        "first_mean": _mean("first"),
        "final_mean": _mean("final"),
        "max_mean": _mean("max"),
        "n_songs_ge_half": sum(
            1 for t in eos_traces if t.get("first_step_ge_half") is not None
        ),
    }


def _length_control_report(
    length_control: inference.ArLengthControl | None,
) -> dict[str, object] | None:
    """Return a JSON-friendly record of active decode length control, if any.

    Args:
        length_control: Length control applied during free-run decode.
    """
    if length_control is None or not length_control.is_active():
        return None
    return {
        "eos_logit_bias": length_control.eos_logit_bias,
        "min_onset_tokens": length_control.min_onset_tokens,
    }


def _ar_decode_gate_metrics(
    pred_times: np.ndarray,
    target_times: np.ndarray,
    chart_times: np.ndarray,
    *,
    tolerance_sec: float,
    ar_decode_length: int,
    stopped_on_eos: bool,
    timing_mode: str = "two_pass",
    eos_prob_trace: np.ndarray | None = None,
    length_control: inference.ArLengthControl | None = None,
) -> dict[str, object]:
    """Primary ordered match vs target_times; aux vs raw chart."""
    ordered = _ordered_onset_report(
        pred_times,
        target_times,
        tolerance_sec=tolerance_sec,
    )
    chart_ordered = _ordered_onset_report(
        pred_times,
        chart_times,
        tolerance_sec=tolerance_sec,
    )
    event_block = _event_f1_report(
        pred_times,
        chart_times,
        tolerance_sec=tolerance_sec,
    )
    metrics: dict[str, object] = {
        "timing_mode": timing_mode,
        mn.TIMING_MATCH_AR_DECODE: ordered,
        "timing_match": ordered,
        "ordered_onset_match": ordered,
        mn.AUX_TIMING_MATCH_CHART: chart_ordered,
        "chart_ordered_onset_match": chart_ordered,
        "ar_decode_length": ar_decode_length,
        "stopped_on_eos": stopped_on_eos,
        **event_block,
        mn.AUX_F1_HUNGARIAN: event_block["event_f1"],
    }
    eos_trace = _eos_trace_stats(eos_prob_trace)
    if eos_trace is not None:
        metrics["eos_trace"] = eos_trace
    control = _length_control_report(length_control)
    if control is not None:
        metrics["length_control"] = control
    return metrics


def _primary_timing_block(report: dict[str, object]) -> dict[str, object]:
    block = report.get(mn.TIMING_MATCH_TEACHER, report.get("ordered_onset_match"))
    assert isinstance(block, dict)
    return block


def _attach_metrics_by_tier(report: dict[str, object]) -> None:
    """Add ``metrics_by_tier`` for JSON consumers (legacy flat keys unchanged)."""
    primary: dict[str, object] = {}
    aux: dict[str, object] = {}
    diag: dict[str, object] = {}

    teacher = report.get(mn.TIMING_MATCH_TEACHER, report.get("ordered_onset_match"))
    if isinstance(teacher, dict):
        primary[mn.TIMING_MATCH_TEACHER] = teacher

    chart = report.get(
        mn.AUX_TIMING_MATCH_CHART, report.get("chart_ordered_onset_match")
    )
    if isinstance(chart, dict):
        aux[mn.AUX_TIMING_MATCH_CHART] = chart

    if "event_f1" in report:
        aux[mn.AUX_F1_HUNGARIAN] = report["event_f1"]

    ar_decode = report.get("ar_decode")
    if isinstance(ar_decode, dict):
        decode_primary = ar_decode.get(
            mn.TIMING_MATCH_AR_DECODE,
            ar_decode.get("ordered_onset_match"),
        )
        if isinstance(decode_primary, dict):
            primary[mn.TIMING_MATCH_AR_DECODE] = decode_primary
        decode_chart = ar_decode.get(
            mn.AUX_TIMING_MATCH_CHART,
            ar_decode.get("chart_ordered_onset_match"),
        )
        if isinstance(decode_chart, dict):
            aux[f"{mn.AUX_TIMING_MATCH_CHART}_ar_decode"] = decode_chart
        if "event_f1" in ar_decode:
            aux[mn.AUX_F1_HUNGARIAN_AR_DECODE] = ar_decode["event_f1"]

    for key in (
        "abs_error_ms",
        "residual_error_ms",
        "n_within_tolerance",
        "n_patch_wrong",
        "n_patch_ok_timing_wrong",
        "worst_onsets",
        "eval_elapsed_sec",
    ):
        if key in report:
            diag[key] = report[key]

    report["metrics_by_tier"] = {"primary": primary, "aux": aux, "diag": diag}


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
    ordered = _primary_timing_block(report)
    assert isinstance(ordered, dict)
    _log("", quiet=quiet)
    _log("=== Teacher-fed gate ===", quiet=quiet)
    _log(
        f"  {_ordered_gate_line(ordered, tol_ms=tol_ms)} (vs target_times)",
        quiet=quiet,
    )
    chart = report.get("chart_ordered_onset_match")
    if isinstance(chart, dict):
        _log(
            f"  {_ordered_gate_line(chart, tol_ms=tol_ms)} (aux: raw chart)",
            quiet=quiet,
        )
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


def _log_null_baseline(block: object, *, quiet: bool) -> None:
    """Log the audio-blind chance floor and the model's skill over it.

    Args:
        block: ``null_baseline`` aggregate, or ``None`` when not computed.
        quiet: Suppress stderr logging.
    """
    if not isinstance(block, dict):
        return
    by_kind = block.get("by_kind")
    if isinstance(by_kind, dict):
        parts = " | ".join(
            f"{kind} {float(vals['event_f1']):.4f}" for kind, vals in by_kind.items()
        )
        _log(f"  Null F1 @ matched count: {parts}", quiet=quiet)
    _log(
        f"  Skill over strongest null ({block.get('strongest', {}).get('event_f1', '')}"
        f" {float(block.get('event_f1_floor', 0.0)):.4f}): "
        f"F1 {float(block.get('skill_event_f1', 0.0)):+.4f} | "
        f"timing_match {float(block.get('skill_timing_match', 0.0)):+.4f}",
        quiet=quiet,
    )


def _print_ar_decode_summary(
    ar_decode: dict[str, object],
    *,
    tolerance_sec: float,
    quiet: bool,
) -> None:
    tol_ms = tolerance_sec * 1000.0
    ordered = ar_decode.get(mn.TIMING_MATCH_AR_DECODE, ar_decode["ordered_onset_match"])
    assert isinstance(ordered, dict)
    diagnostics = ar_decode.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    _log("", quiet=quiet)
    _log("=== Free-run AR gate (two-pass) ===", quiet=quiet)
    if "ar_decode_length" in ar_decode:
        _log(
            f"  Decode length: {ar_decode['ar_decode_length']} | "
            f"EOS: {'yes' if ar_decode['stopped_on_eos'] else 'no'}",
            quiet=quiet,
        )
    elif "ar_decode_length_sum" in ar_decode:
        n_eos = int(ar_decode.get("n_songs_stopped_on_eos", 0))
        _log(
            f"  Decode length (sum): {ar_decode['ar_decode_length_sum']} | "
            f"songs stopped on EOS: {n_eos}",
            quiet=quiet,
        )
    _log(
        f"  {_ordered_gate_line(ordered, tol_ms=tol_ms)} (vs target_times)",
        quiet=quiet,
    )
    chart = ar_decode.get("chart_ordered_onset_match")
    if isinstance(chart, dict):
        _log(
            f"  {_ordered_gate_line(chart, tol_ms=tol_ms)} (aux: raw chart)",
            quiet=quiet,
        )
    control = ar_decode.get("length_control")
    if isinstance(control, dict):
        _log(
            f"  Length control: eos_logit_bias={control['eos_logit_bias']} "
            f"min_onset_tokens={control['min_onset_tokens']}",
            quiet=quiet,
        )
    eos_trace = ar_decode.get("eos_trace")
    if isinstance(eos_trace, dict):
        _log(f"  {_eos_trace_line(eos_trace)}", quiet=quiet)
    _log("", quiet=quiet)
    _log("--- Aux ---", quiet=quiet)
    _log(f"  Hungarian F1: {_event_f1_line(ar_decode)}", quiet=quiet)
    _log_null_baseline(ar_decode.get("null_baseline"), quiet=quiet)

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
    split = report.get("split", "overfit")
    n_songs = report.get("n_songs")
    title = "=== AR offline eval ==="
    if n_songs is not None:
        title = f"=== AR offline eval ({split}, {n_songs} songs) ==="
    elif split != "overfit":
        title = f"=== AR offline eval ({split}) ==="
    _log(title, quiet=quiet)
    _log(f"Model: {model_path}", quiet=quiet)
    ordered = report["ordered_onset_match"]
    assert isinstance(ordered, dict)
    _log(
        f"Primary metric: ordered onset match @ {tolerance_sec * 1000.0:.0f} ms "
        f"({int(ordered['n_matched'])}/{int(ordered['n_denom'])})",
        quiet=quiet,
    )
    _print_teacher_summary(report, tolerance_sec=tolerance_sec, quiet=quiet)
    _log_null_baseline(report.get("null_baseline"), quiet=quiet)
    if report.get("ar_decode_skipped"):
        reason = report.get("teacher_gate_reason", "teacher gate failed")
        _log(f"Free-run skipped: {reason}", quiet=quiet)
    elif ar_decode:
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


def _model_inputs(
    batch: dict[str, tf.Tensor],
    experiment_config: config.ArExperimentConfig,
) -> dict[str, tf.Tensor]:
    inputs = {
        "mert_patches": batch["mert_patches"],
        "patch_mask": batch["patch_mask"],
        "decoder_input_ids": batch["decoder_input_ids"],
        "decoder_mask": batch["decoder_mask"],
    }
    if config.density_conditioning_active(experiment_config.model):
        inputs["density_scalar"] = batch["density_scalar"]
    return inputs


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
    target_kept = target_times[step_indices]
    ordered = _ordered_onset_report(
        pred_times,
        target_kept,
        tolerance_sec=tolerance_sec,
    )
    chart_ordered = _ordered_onset_report(
        pred_times,
        gt_times,
        tolerance_sec=tolerance_sec,
    )

    return {
        "n_onsets": int(pred_times.size),
        mn.TIMING_MATCH_TEACHER: ordered,
        "timing_match": ordered,
        "ordered_onset_match": ordered,
        mn.AUX_TIMING_MATCH_CHART: chart_ordered,
        "chart_ordered_onset_match": chart_ordered,
        "event_f1": float(event_f1),
        mn.AUX_F1_HUNGARIAN: float(event_f1),
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
    length_control: inference.ArLengthControl | None = None,
) -> dict[str, object]:
    """Two-pass gate metrics; incremental timing under ``diagnostics``."""
    run_config = experiment_config.run
    model_config = experiment_config.model
    max_decoder_len = experiment_config.max_decoder_len()
    mert = batch_np["mert_patches"]
    patch_mask = batch_np["patch_mask"]
    target_times, chart_times = _onset_reference_times(batch_np)
    gt_target = batch_np["decoder_target_ids"][0]
    gt_mask = batch_np["decoder_mask"][0]

    decode_kwargs = {
        "max_decoder_len": max_decoder_len,
        "patch_frames": model_config.patch_frames,
        "hop_sec": experiment_config.dataset.hop_sec,
        "experiment_config": experiment_config,
    }
    if config.density_conditioning_active(model_config):
        decode_kwargs["density_scalar"] = batch_np["density_scalar"]

    incremental_stats = inference.decode_autoregressive_with_stats_numpy(
        model,
        mert,
        patch_mask,
        time_source=time_source,
        length_control=length_control,
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
                chart_times,
                tolerance_sec=run_config.tolerance_sec,
            ),
        },
        "gt_timing": _gt_timing_diagnostic(
            model,
            batch_np,
            experiment_config=experiment_config,
            gt_times=chart_times,
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
            chart_times,
            tolerance_sec=run_config.tolerance_sec,
        )

    report: dict[str, object] = {
        **_ar_decode_gate_metrics(
            gate_stats.times,
            target_times,
            chart_times,
            tolerance_sec=run_config.tolerance_sec,
            ar_decode_length=gate_stats.n_forward_steps,
            stopped_on_eos=gate_stats.stopped_on_eos,
            eos_prob_trace=gate_stats.eos_prob_trace,
            length_control=length_control,
        ),
        "diagnostics": diagnostics,
    }

    if token_trace_steps <= 0:
        return report

    memory, key_input, pm_b = inference.get_encoder_memory_numpy(
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
        decoder_inputs = {
            "encoder_memory": memory,
            "patch_mask": pm_b,
            "decoder_input_ids": dec_in,
            "decoder_mask": dec_mask,
        }
        if config.content_pointer_active(experiment_config.model):
            decoder_inputs["pointer_key_input"] = key_input
        if config.density_conditioning_active(experiment_config.model):
            decoder_inputs["density_scalar"] = batch_np["density_scalar"]
        outputs = decoder(
            decoder_inputs,
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
    length_control: inference.ArLengthControl | None = None,
) -> dict[str, object]:
    """Two-pass gate metrics only (no diagnostics overhead)."""
    run_config = experiment_config.run
    model_config = experiment_config.model
    mert = batch_np["mert_patches"]
    patch_mask = batch_np["patch_mask"]
    target_times, chart_times = _onset_reference_times(batch_np)

    gate_kwargs = {
        "max_decoder_len": experiment_config.max_decoder_len(),
        "patch_frames": model_config.patch_frames,
        "hop_sec": experiment_config.dataset.hop_sec,
        "experiment_config": experiment_config,
        "length_control": length_control,
    }
    if config.density_conditioning_active(model_config):
        gate_kwargs["density_scalar"] = batch_np["density_scalar"]

    gate_stats = inference.decode_autoregressive_gate_with_stats_numpy(
        model,
        mert,
        patch_mask,
        **gate_kwargs,
    )
    return _ar_decode_gate_metrics(
        gate_stats.times,
        target_times,
        chart_times,
        tolerance_sec=run_config.tolerance_sec,
        ar_decode_length=gate_stats.n_forward_steps,
        stopped_on_eos=gate_stats.stopped_on_eos,
        eos_prob_trace=gate_stats.eos_prob_trace,
        length_control=length_control,
    )


def _sum_ordered(blocks: list[dict[str, object]]) -> dict[str, float | int]:
    n_matched = sum(int(b["n_matched"]) for b in blocks)
    n_pred = sum(int(b["n_pred"]) for b in blocks)
    n_ref = sum(int(b["n_ref"]) for b in blocks)
    n_denom = sum(int(b["n_denom"]) for b in blocks)
    return {
        "n_matched": n_matched,
        "n_pred": n_pred,
        "n_gt": n_ref,
        "n_ref": n_ref,
        "n_denom": n_denom,
        "rate": float(n_matched / n_denom) if n_denom else 0.0,
    }


def _sum_event_f1(rows: list[dict[str, object]]) -> dict[str, float | int]:
    tp = sum(int(r["true_positives"]) for r in rows)
    fp = sum(int(r["false_positives"]) for r in rows)
    fn = sum(int(r["false_negatives"]) for r in rows)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    event_f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return {
        "n_pred_onsets": sum(
            int(r.get("n_pred_onsets", r.get("n_onsets", 0))) for r in rows
        ),
        "n_gt_onsets": tp + fn,
        "event_f1": float(event_f1),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _percentile_stats(values_ms: np.ndarray) -> dict[str, float]:
    if values_ms.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "p50": float(np.percentile(values_ms, 50)),
        "p90": float(np.percentile(values_ms, 90)),
        "p99": float(np.percentile(values_ms, 99)),
        "max": float(np.max(values_ms)),
        "mean": float(np.mean(values_ms)),
    }


def _list_split_samples(
    experiment_config: config.ArExperimentConfig,
    *,
    split: str,
    limit: int,
) -> list[tuple[str, str, int]]:
    samples = datasets._filter_valid_ar_samples(  # noqa: SLF001
        datasets.list_ar_training_samples(experiment_config, split=split),
        experiment_config.dataset,
    )
    if limit > 0:
        samples = samples[:limit]
    return samples


def _load_split_batch(
    experiment_config: config.ArExperimentConfig,
    audio_path: str,
    chart_path: str,
    chart_index: int,
) -> dict[str, np.ndarray]:
    sample = datasets.load_ar_sample(
        audio_path,
        chart_path,
        dataset_config=experiment_config.dataset,
        model_config=experiment_config.model,
        vocab=experiment_config.build_vocab(),
        chart_index=chart_index,
    )
    return datasets.sample_to_training_batch(sample, experiment_config)


def _eval_one_batch(
    model: tf.keras.Model,
    batch_np: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
    ar_decode: bool,
    full_diagnostics: bool,
    token_trace_steps: int,
    ar_decode_time_source: str,
    worst_k: int,
    keep_error_arrays: bool = False,
    length_control: inference.ArLengthControl | None = None,
) -> dict[str, object]:
    batch_tf = {key: tf.constant(value) for key, value in batch_np.items()}
    outputs = model(_model_inputs(batch_tf, experiment_config), training=False)
    report = _diagnose_batch(outputs, batch_np, experiment_config=experiment_config)
    report["worst_onsets"] = _worst_onsets(report, worst_k)
    if ar_decode:
        _attach_ar_decode_to_report(
            report,
            model,
            batch_np,
            experiment_config=experiment_config,
            full_diagnostics=full_diagnostics,
            token_trace_steps=token_trace_steps,
            ar_decode_time_source=ar_decode_time_source,
            length_control=length_control,
        )
    hop_sec = experiment_config.dataset.hop_sec
    tolerance_sec = experiment_config.run.tolerance_sec
    teacher_ordered = report.get("ordered_onset_match")
    if isinstance(teacher_ordered, dict):
        report["null_baseline_counts"] = _null_baseline_block(
            batch_np,
            n_pred=int(teacher_ordered["n_pred"]),
            tolerance_sec=tolerance_sec,
            hop_sec=hop_sec,
        )
    ar_block = report.get("ar_decode")
    if isinstance(ar_block, dict):
        ar_block["null_baseline_counts"] = _null_baseline_block(
            batch_np,
            n_pred=int(ar_block.get("n_pred_onsets", 0)),
            tolerance_sec=tolerance_sec,
            hop_sec=hop_sec,
        )
    abs_err = report.get("_abs_err_ms")
    residual_err = report.get("_residual_err_ms")
    for key in list(report):
        if key.startswith("_"):
            del report[key]
    if keep_error_arrays:
        if abs_err is not None:
            report["_abs_err_ms_all"] = np.asarray(abs_err, dtype=np.float64).tolist()
        if residual_err is not None:
            report["_residual_err_ms_all"] = np.asarray(
                residual_err,
                dtype=np.float64,
            ).tolist()
    return report


def _attach_ar_decode_to_report(
    report: dict[str, object],
    model: tf.keras.Model,
    batch_np: dict[str, np.ndarray],
    *,
    experiment_config: config.ArExperimentConfig,
    full_diagnostics: bool,
    token_trace_steps: int,
    ar_decode_time_source: str,
    length_control: inference.ArLengthControl | None = None,
) -> None:
    if full_diagnostics:
        report["ar_decode"] = _ar_decode_report(
            model,
            batch_np,
            experiment_config=experiment_config,
            token_trace_steps=token_trace_steps,
            time_source=ar_decode_time_source,
            compare_time_sources=True,
            length_control=length_control,
        )
    else:
        report["ar_decode"] = _ar_decode_gate_only_report(
            model,
            batch_np,
            experiment_config=experiment_config,
            length_control=length_control,
        )
    hop_sec = experiment_config.dataset.hop_sec
    tolerance_sec = experiment_config.run.tolerance_sec
    ar_block = report.get("ar_decode")
    if isinstance(ar_block, dict):
        ar_block["null_baseline_counts"] = _null_baseline_block(
            batch_np,
            n_pred=int(ar_block.get("n_pred_onsets", 0)),
            tolerance_sec=tolerance_sec,
            hop_sec=hop_sec,
        )


def _aggregate_split_reports(
    song_reports: list[dict[str, object]],
    *,
    ar_decode: bool,
) -> dict[str, object]:
    teacher_ordered = _sum_ordered(
        [r["ordered_onset_match"] for r in song_reports],  # type: ignore[arg-type]
    )
    teacher_chart = _sum_ordered(
        [r["chart_ordered_onset_match"] for r in song_reports],  # type: ignore[arg-type]
    )
    teacher_f1 = _sum_event_f1(song_reports)
    abs_chunks = []
    residual_chunks = []
    public_songs: list[dict[str, object]] = []
    for row in song_reports:
        errs = row.pop("_abs_err_ms_all", None)
        if isinstance(errs, list):
            abs_chunks.append(np.asarray(errs, dtype=np.float64))
        res = row.pop("_residual_err_ms_all", None)
        if isinstance(res, list):
            residual_chunks.append(np.asarray(res, dtype=np.float64))
        public_songs.append(row)

    abs_all = (
        np.concatenate(abs_chunks) if abs_chunks else np.asarray([], dtype=np.float64)
    )
    residual_all = (
        np.concatenate(residual_chunks)
        if residual_chunks
        else np.asarray([], dtype=np.float64)
    )

    report: dict[str, object] = {
        "n_songs": len(public_songs),
        "n_onsets": int(teacher_ordered["n_pred"]),
        mn.TIMING_MATCH_TEACHER: teacher_ordered,
        "timing_match": teacher_ordered,
        "ordered_onset_match": teacher_ordered,
        mn.AUX_TIMING_MATCH_CHART: teacher_chart,
        "chart_ordered_onset_match": teacher_chart,
        "event_f1": float(teacher_f1["event_f1"]),
        mn.AUX_F1_HUNGARIAN: float(teacher_f1["event_f1"]),
        "true_positives": teacher_f1["true_positives"],
        "false_positives": teacher_f1["false_positives"],
        "false_negatives": teacher_f1["false_negatives"],
        "abs_error_ms": _percentile_stats(abs_all),
        "n_within_tolerance": sum(
            int(r.get("n_within_tolerance", 0)) for r in public_songs
        ),
        "n_patch_wrong": sum(int(r.get("n_patch_wrong", 0)) for r in public_songs),
        "n_patch_ok_timing_wrong": sum(
            int(r.get("n_patch_ok_timing_wrong", 0)) for r in public_songs
        ),
        "residual_error_ms": _percentile_stats(residual_all),
        "songs": public_songs,
    }
    teacher_nulls = [
        r["null_baseline_counts"]
        for r in public_songs
        if isinstance(r.get("null_baseline_counts"), list)
    ]
    if teacher_nulls:
        report["null_baseline"] = _aggregate_null_baselines(
            teacher_nulls,  # type: ignore[arg-type]
            model_event_f1=float(teacher_f1["event_f1"]),
            model_timing_match=float(teacher_ordered["rate"]),
        )
    if ar_decode:
        ar_rows = [
            r["ar_decode"] for r in public_songs if isinstance(r.get("ar_decode"), dict)
        ]
        ar_ordered = _sum_ordered(
            [r["ordered_onset_match"] for r in ar_rows],  # type: ignore[arg-type]
        )
        ar_chart = _sum_ordered(
            [r["chart_ordered_onset_match"] for r in ar_rows],  # type: ignore[arg-type]
        )
        ar_f1 = _sum_event_f1(ar_rows)
        ar_block: dict[str, object] = {
            "timing_mode": "two_pass",
            mn.TIMING_MATCH_AR_DECODE: ar_ordered,
            "timing_match": ar_ordered,
            "ordered_onset_match": ar_ordered,
            mn.AUX_TIMING_MATCH_CHART: ar_chart,
            "chart_ordered_onset_match": ar_chart,
            **ar_f1,
            mn.AUX_F1_HUNGARIAN: ar_f1["event_f1"],
            "n_songs_stopped_on_eos": sum(
                1 for r in ar_rows if bool(r.get("stopped_on_eos"))
            ),
            "ar_decode_length_sum": sum(
                int(r.get("ar_decode_length", 0)) for r in ar_rows
            ),
        }
        eos_traces = [
            r["eos_trace"] for r in ar_rows if isinstance(r.get("eos_trace"), dict)
        ]
        if eos_traces:
            ar_block["eos_trace"] = _aggregate_eos_traces(eos_traces)
        control = next(
            (r["length_control"] for r in ar_rows if r.get("length_control")),
            None,
        )
        if control is not None:
            ar_block["length_control"] = control
        ar_nulls = [
            r["null_baseline_counts"]
            for r in ar_rows
            if isinstance(r.get("null_baseline_counts"), list)
        ]
        if ar_nulls:
            ar_block["null_baseline"] = _aggregate_null_baselines(
                ar_nulls,  # type: ignore[arg-type]
                model_event_f1=float(ar_f1["event_f1"]),
                model_timing_match=float(ar_ordered["rate"]),
            )
        report["ar_decode"] = ar_block
    return report


def _eval_split(
    model: tf.keras.Model,
    *,
    experiment_config: config.ArExperimentConfig,
    split: str,
    limit: int,
    ar_decode: bool,
    force_ar_decode: bool,
    min_teacher_rate: float,
    quiet: bool,
    worst_k: int,
    length_control: inference.ArLengthControl | None = None,
    difficulty_tier: str = "",
    density_presets_path: str = "",
) -> dict[str, object]:
    samples = _list_split_samples(experiment_config, split=split, limit=limit)
    if not samples:
        raise ValueError(f"No valid AR samples for split={split!r}")
    _log(
        f"Evaluating {len(samples)} {split} songs (teacher-fed)...",
        quiet=quiet,
    )
    song_reports: list[dict[str, object]] = []
    for i, (audio_path, chart_path, chart_index) in enumerate(samples, start=1):
        label = f"{pathlib.Path(chart_path).name}#{chart_index}"
        _log(f"  [{i}/{len(samples)}] {label}", quiet=quiet)
        batch_np = _load_split_batch(
            experiment_config,
            audio_path,
            chart_path,
            chart_index,
        )
        song = _eval_one_batch(
            model,
            batch_np,
            experiment_config=experiment_config,
            ar_decode=False,
            full_diagnostics=False,
            token_trace_steps=0,
            ar_decode_time_source="pointer_residual",
            worst_k=worst_k,
            keep_error_arrays=True,
            length_control=length_control,
        )
        teacher = song.get("ordered_onset_match", {})
        assert isinstance(teacher, dict)
        _log(
            f"       teacher ordered "
            f"{int(teacher.get('n_matched', 0))}/{int(teacher.get('n_denom', 0))} "
            f"rate={float(teacher.get('rate', 0.0)):.4f}",
            quiet=quiet,
        )
        song["audio_path"] = audio_path
        song["chart_path"] = chart_path
        song["chart_index"] = chart_index
        song["label"] = label
        song_reports.append(song)

    report = _aggregate_split_reports(song_reports, ar_decode=False)
    if not ar_decode:
        return report

    passed, reason = _teacher_gate_passes(
        report,
        split=split,
        min_teacher_rate=min_teacher_rate,
    )
    if not passed and not force_ar_decode:
        _mark_ar_decode_skipped(report, reason=reason)
        _log(f"Skipping free-run: {reason}", quiet=quiet)
        return report

    if passed:
        _log("Teacher gate passed; running free-run...", quiet=quiet)
    else:
        _log(
            f"Teacher gate failed ({reason}); running free-run anyway "
            f"(--force-ar-decode)",
            quiet=quiet,
        )
    report["teacher_gate_passed"] = passed

    for i, ((audio_path, chart_path, chart_index), song) in enumerate(
        zip(samples, song_reports, strict=True),
        start=1,
    ):
        label = f"{pathlib.Path(chart_path).name}#{chart_index}"
        _log(f"  [{i}/{len(samples)}] {label} (free-run)", quiet=quiet)
        batch_np = _load_split_batch(
            experiment_config,
            audio_path,
            chart_path,
            chart_index,
        )
        if difficulty_tier:
            _apply_customer_density_tier(
                batch_np,
                experiment_config=experiment_config,
                difficulty_tier=difficulty_tier,
                density_presets_path=density_presets_path,
            )
        _attach_ar_decode_to_report(
            song,
            model,
            batch_np,
            experiment_config=experiment_config,
            full_diagnostics=False,
            token_trace_steps=0,
            ar_decode_time_source="pointer_residual",
            length_control=length_control,
        )
        ar_block = song.get("ar_decode")
        if isinstance(ar_block, dict):
            ordered = ar_block.get("ordered_onset_match", {})
            assert isinstance(ordered, dict)
            _log(
                f"       free-run ordered "
                f"{int(ordered.get('n_matched', 0))}/"
                f"{int(ordered.get('n_denom', 0))} "
                f"rate={float(ordered.get('rate', 0.0)):.4f}",
                quiet=quiet,
            )

    return _aggregate_split_reports(song_reports, ar_decode=True)


def main() -> int:
    args = PARSER.parse_args()
    quiet = args.json_only
    wsl_gpu.guard_tensorflow_gpu_job(__file__)
    return _run_main(args, quiet=quiet)


def _run_main(args: argparse.Namespace, *, quiet: bool) -> int:
    if args.full_diagnostics and args.split != "overfit":
        print(
            "--full-diagnostics is only supported with --split overfit",
            file=sys.stderr,
        )
        return 1
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment_config, args.model_path)
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    _log(f"Loading checkpoint: {model_path}", quiet=quiet)
    model = tf.keras.models.load_model(str(model_path), compile=False)
    t0 = time.perf_counter()

    length_control = inference.ArLengthControl(
        eos_logit_bias=args.ar_decode_eos_logit_bias,
        min_onset_tokens=args.ar_decode_min_onset_tokens,
    )
    if args.ar_decode and length_control.is_active():
        _log(
            f"Free-run length control: eos_logit_bias="
            f"{length_control.eos_logit_bias} "
            f"min_onset_tokens={length_control.min_onset_tokens}",
            quiet=quiet,
        )
    if args.difficulty_tier and not args.ar_decode:
        print("--difficulty-tier requires --ar_decode", file=sys.stderr)
        return 1
    if args.ar_decode and args.difficulty_tier:
        if not config.density_conditioning_active(experiment_config.model):
            print(
                "--difficulty-tier requires model.density_conditioning in config",
                file=sys.stderr,
            )
            return 1
        _log(
            f"Customer density tier: {args.difficulty_tier!r} "
            f"(presets={args.density_presets_path or density_presets.DEFAULT_PRESETS_PATH})",
            quiet=quiet,
        )

    if args.split == "overfit":
        _log("Running teacher-fed eval...", quiet=quiet)
        batch_np = datasets.sample_to_training_batch(
            datasets.load_overfit_sample(experiment_config),
            experiment_config,
        )
        if args.difficulty_tier and args.ar_decode:
            _apply_customer_density_tier(
                batch_np,
                experiment_config=experiment_config,
                difficulty_tier=args.difficulty_tier,
                density_presets_path=args.density_presets_path,
            )
        report = _eval_one_batch(
            model,
            batch_np,
            experiment_config=experiment_config,
            ar_decode=False,
            full_diagnostics=False,
            token_trace_steps=args.token_trace_steps,
            ar_decode_time_source=args.ar_decode_time_source,
            worst_k=args.worst_k,
            length_control=length_control,
        )
        if args.ar_decode:
            passed, reason = _teacher_gate_passes(report, split="overfit")
            if not passed and not args.force_ar_decode:
                _mark_ar_decode_skipped(report, reason=reason)
                _log(f"Skipping free-run: {reason}", quiet=quiet)
            else:
                if passed:
                    _log("Teacher gate passed; running free-run...", quiet=quiet)
                else:
                    _log(
                        f"Teacher gate failed ({reason}); running free-run anyway "
                        f"(--force-ar-decode)",
                        quiet=quiet,
                    )
                if args.full_diagnostics:
                    _log(
                        "Running free-run AR decode (two-pass, full diagnostics)...",
                        quiet=quiet,
                    )
                else:
                    _log("Running free-run AR gate (two-pass)...", quiet=quiet)
                _attach_ar_decode_to_report(
                    report,
                    model,
                    batch_np,
                    experiment_config=experiment_config,
                    full_diagnostics=args.full_diagnostics,
                    token_trace_steps=args.token_trace_steps,
                    ar_decode_time_source=args.ar_decode_time_source,
                    length_control=length_control,
                )
                report["teacher_gate_passed"] = passed
    else:
        report = _eval_split(
            model,
            experiment_config=experiment_config,
            split=args.split,
            limit=args.limit,
            ar_decode=args.ar_decode,
            force_ar_decode=args.force_ar_decode,
            min_teacher_rate=args.min_teacher_rate,
            quiet=quiet,
            worst_k=args.worst_k,
            length_control=length_control,
            difficulty_tier=args.difficulty_tier,
            density_presets_path=args.density_presets_path,
        )

    elapsed_sec = time.perf_counter() - t0
    report["model_path"] = str(model_path)
    report["config"] = args.config
    report["split"] = args.split
    if args.difficulty_tier:
        report["difficulty_tier"] = args.difficulty_tier
    report["eval_elapsed_sec"] = round(elapsed_sec, 3)
    _attach_metrics_by_tier(report)
    _print_debug_summary(
        report,
        model_path=model_path,
        tolerance_sec=experiment_config.run.tolerance_sec,
        ar_decode=args.ar_decode,
        quiet=quiet,
    )
    if not quiet:
        n_songs = report.get("n_songs")
        if n_songs is not None:
            _log(f"Songs evaluated: {n_songs}", quiet=False)
        _log(f"Eval wall time: {elapsed_sec:.2f} s", quiet=False)
    print(json.dumps(report, indent=2))
    if args.ar_decode and report.get("teacher_gate_failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
