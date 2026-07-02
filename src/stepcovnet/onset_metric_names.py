"""Canonical onset metric names and legacy aliases.

Tier prefixes make training logs and eval JSON scannable:

- ``gate_*``       — composite pass/fail signals (checkpoint / early stop)
- ``timing_match_*`` — primary ordered timing @ tolerance
- ``token_accuracy`` — AR token correctness (support for gates)
- ``aux_*``        — Hungarian F1, raw-chart timing, legacy scoreboards
- ``loss_*``       — optimization terms (not pass/fail)
- ``diag_*``       — offline-only debugging fields

Keras validation logs prefix metric names with ``val_`` (e.g. ``val_gate_teacher``).
"""

from __future__ import annotations

# --- Primary: ordered timing match @ tolerance_sec (default 20 ms) ---

TIMING_MATCH_TEACHER = "timing_match_teacher"
TIMING_MATCH_AR_DECODE = "timing_match_ar_decode"
TIMING_MATCH_MICRO = "timing_match_micro"  # dense multi-song eval

# --- Gates (composite decision metrics) ---

GATE_TEACHER = "gate_teacher"  # min(token_accuracy, timing_match_teacher)
GATE_AR_DECODE = "gate_ar_decode"  # reserved: min(...) on free-run path

# --- Support ---

TOKEN_ACCURACY = "token_accuracy"

# --- Auxiliary ---

AUX_F1_HUNGARIAN = "aux_f1_hungarian"
AUX_F1_HUNGARIAN_MINGAP = "aux_f1_hungarian_mingap"
AUX_F1_HUNGARIAN_AR_DECODE = "aux_f1_hungarian_ar_decode"
AUX_TIMING_MATCH_CHART = "aux_timing_match_chart"

# --- Losses (AR trainer) ---

LOSS_TOTAL = "loss"
LOSS_TOKEN = "token_loss"
LOSS_POINTER = "pointer_loss"
LOSS_TIME = "time_loss"
LOSS_RESIDUAL = "residual_loss"
LOSS_INCREMENTAL_CONSISTENCY = "incremental_consistency_loss"

# Keras / TensorBoard metric name -> canonical name (no val_ prefix).
LEGACY_METRIC_ALIASES: dict[str, str] = {
    "overfit_gate": GATE_TEACHER,
    "ordered_onset_match": TIMING_MATCH_TEACHER,
    "event_onset_f1": AUX_F1_HUNGARIAN,
    "event_onset_f1_mingap": AUX_F1_HUNGARIAN_MINGAP,
    "ar_decode_ordered_onset_match": TIMING_MATCH_AR_DECODE,
    "ar_decode_event_f1": AUX_F1_HUNGARIAN_AR_DECODE,
    "onset_f1_score": "frame_onset_f1",  # dense frame metric (legacy)
}

# Canonical -> legacy Keras metric name (for dual-publish in logs).
CANONICAL_TO_LEGACY_METRIC: dict[str, str] = {
    GATE_TEACHER: "overfit_gate",
    TIMING_MATCH_TEACHER: "ordered_onset_match",
    AUX_F1_HUNGARIAN: "event_onset_f1",
    AUX_F1_HUNGARIAN_MINGAP: "event_onset_f1_mingap",
    TIMING_MATCH_AR_DECODE: "ar_decode_ordered_onset_match",
    AUX_F1_HUNGARIAN_AR_DECODE: "ar_decode_event_f1",
}

# checkpoint_metric config values (with val_) -> canonical (no val_).
LEGACY_CHECKPOINT_ALIASES: dict[str, str] = {
    "val_overfit_gate": GATE_TEACHER,
    "val_ordered_onset_match": TIMING_MATCH_TEACHER,
    "val_event_onset_f1": AUX_F1_HUNGARIAN,
    "val_ar_decode_ordered_onset_match": TIMING_MATCH_AR_DECODE,
    "val_ar_decode_event_f1": AUX_F1_HUNGARIAN_AR_DECODE,
    "val_onset_f1_score": "frame_onset_f1",
}

# Eval JSON top-level keys -> tier.
EVAL_JSON_TIERS: dict[str, str] = {
    "timing_match": "primary",
    "ordered_onset_match": "primary",
    TIMING_MATCH_TEACHER: "primary",
    "chart_ordered_onset_match": "aux",
    AUX_TIMING_MATCH_CHART: "aux",
    "event_f1": "aux",
    AUX_F1_HUNGARIAN: "aux",
    "true_positives": "aux",
    "false_positives": "aux",
    "false_negatives": "aux",
    "abs_error_ms": "diag",
    "residual_error_ms": "diag",
    "n_within_tolerance": "diag",
    "n_patch_wrong": "diag",
    "n_patch_ok_timing_wrong": "diag",
    "worst_onsets": "diag",
    "eval_elapsed_sec": "diag",
}


def val_name(metric: str) -> str:
    """Keras validation log key for a canonical metric name."""
    return metric if metric.startswith("val_") else f"val_{metric}"


def resolve_checkpoint_metric(name: str) -> str:
    """Map config ``checkpoint_metric`` to the Keras ``val_*`` monitor name."""
    canonical = LEGACY_CHECKPOINT_ALIASES.get(name, name.removeprefix("val_"))
    legacy = CANONICAL_TO_LEGACY_METRIC.get(canonical)
    if legacy is not None:
        return val_name(legacy)
    return name if name.startswith("val_") else val_name(canonical)


def canonical_metric_name(keras_metric_name: str) -> str:
    """Resolve a Keras metric name (no ``val_``) to its canonical form."""
    return LEGACY_METRIC_ALIASES.get(keras_metric_name, keras_metric_name)


def publish_legacy_val_aliases(logs: dict, *, val_prefix: bool = True) -> None:
    """Copy canonical ``val_*`` keys to legacy names still used in old configs/logs."""
    prefix = "val_" if val_prefix else ""
    for canonical, legacy in CANONICAL_TO_LEGACY_METRIC.items():
        canonical_key = f"{prefix}{canonical}"
        legacy_key = f"{prefix}{legacy}"
        if canonical_key in logs and legacy_key not in logs:
            logs[legacy_key] = logs[canonical_key]
        if legacy_key in logs and canonical_key not in logs:
            logs[canonical_key] = logs[legacy_key]


def tier_for_eval_key(key: str) -> str:
    """Return ``primary``, ``aux``, or ``diag`` for an eval JSON field."""
    return EVAL_JSON_TIERS.get(key, "diag")
