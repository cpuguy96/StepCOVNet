"""Diagnostics for single-song overfit checkpoints."""

import dataclasses

import numpy as np

from stepcovnet.onset_events import config, matching, metrics


def uniform_grid_ref_times_sec(
    num_queries: int,
    duration_sec: float,
) -> np.ndarray:
    """Return uniform query anchor times in seconds for ``num_queries`` slots."""
    ref_norm = (np.arange(num_queries, dtype=np.float64) + 0.5) / float(num_queries)
    return ref_norm * duration_sec


def oracle_uniform_grid_coverage(
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    duration_sec: float,
    num_queries: int,
    tolerance_sec: float,
) -> dict[str, int | float]:
    """Count GT onsets matchable in principle by nearest uniform grid slot."""
    valid = gt_mask.astype(bool)
    gt_values = gt_times[valid]
    if gt_values.size == 0 or duration_sec <= 0.0:
        return {
            "num_gt": 0,
            "grid_matchable": 0,
            "grid_matchable_fraction": 0.0,
        }
    grid_times = uniform_grid_ref_times_sec(num_queries, duration_sec)
    matchable = 0
    for gt_time in gt_values:
        diffs = np.abs(grid_times - float(gt_time))
        if float(np.min(diffs)) <= tolerance_sec:
            matchable += 1
    num_gt = int(gt_values.size)
    return {
        "num_gt": num_gt,
        "grid_matchable": matchable,
        "grid_matchable_fraction": float(matchable) / float(num_gt),
    }


def confidence_stats(confidence: np.ndarray) -> dict[str, float]:
    """Summarize slot confidence values."""
    flat = np.asarray(confidence, dtype=np.float64).reshape(-1)
    above = flat >= 0.5
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "count_ge_0.5": int(np.sum(above)),
        "count_ge_0.1": int(np.sum(flat >= 0.1)),
        "count_ge_0.01": int(np.sum(flat >= 0.01)),
    }


def _within_tol_count(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    result: matching.MatchResult,
    tolerance_sec: float,
    batch_idx: int = 0,
) -> int:
    count = 0
    for match_idx in range(int(result.num_matches[batch_idx])):
        pred_idx = int(result.matched_pred_indices[batch_idx, match_idx])
        gt_idx = int(result.matched_gt_indices[batch_idx, match_idx])
        if pred_idx < 0:
            continue
        diff = abs(
            float(pred_times[batch_idx, pred_idx]) - float(gt_times[batch_idx, gt_idx])
        )
        if diff <= tolerance_sec:
            count += 1
    return count


def assignment_summary(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
) -> dict[str, int]:
    """Compare ordered, Hungarian-L1, and eval assignment on one batch."""
    ordered = matching.assign_onset_pairs_ordered_numpy(pred_times, gt_times, gt_mask)
    hungarian_l1 = matching.assign_onset_pairs_l1_numpy(pred_times, gt_times, gt_mask)
    hungarian_eval = matching.match_onsets_numpy(
        pred_times, gt_times, gt_mask, tolerance_sec=tolerance_sec
    )
    return {
        "num_gt": int(np.sum(gt_mask)),
        "ordered_pairs": int(ordered.num_matches[0]),
        "hungarian_l1_pairs": int(hungarian_l1.num_matches[0]),
        "hungarian_eval_pairs": int(hungarian_eval.num_matches[0]),
        "ordered_within_tol": _within_tol_count(
            pred_times, gt_times, ordered, tolerance_sec
        ),
        "hungarian_l1_within_tol": _within_tol_count(
            pred_times, gt_times, hungarian_l1, tolerance_sec
        ),
        "hungarian_eval_within_tol": _within_tol_count(
            pred_times, gt_times, hungarian_eval, tolerance_sec
        ),
    }


@dataclasses.dataclass
class OverfitDiagnosticReport:
    """Structured diagnostics for one overfit forward pass."""

    model_path: str
    frontend: str
    num_queries: int
    duration_sec: float
    confidence_threshold: float
    tolerance_sec: float
    confidence: dict[str, float]
    assignment: dict[str, int]
    eval_f1: float
    eval_tp: float
    eval_fp: float
    eval_fn: float
    pred_time_min_sec: float
    pred_time_max_sec: float

    def as_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable mapping."""
        return {
            "model_path": self.model_path,
            "frontend": self.frontend,
            "num_queries": self.num_queries,
            "duration_sec": self.duration_sec,
            "confidence_threshold": self.confidence_threshold,
            "tolerance_sec": self.tolerance_sec,
            "confidence": self.confidence,
            "assignment": self.assignment,
            "eval": {
                "f1": self.eval_f1,
                "tp": self.eval_tp,
                "fp": self.eval_fp,
                "fn": self.eval_fn,
            },
            "pred_time_range_sec": [
                self.pred_time_min_sec,
                self.pred_time_max_sec,
            ],
        }


def diagnose_overfit_outputs(
    *,
    model_path: str,
    experiment: config.OnsetEventExperimentConfig,
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    duration_sec: float,
) -> OverfitDiagnosticReport:
    """Build a diagnostic report from model outputs and ground truth."""
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred_times,
        pred_confidence,
        gt_times,
        gt_mask,
        experiment.run.tolerance_sec,
        experiment.run.confidence_threshold,
    )
    _precision, _recall, f1 = metrics.event_onset_f1_numpy(
        pred_times,
        pred_confidence,
        gt_times,
        gt_mask,
        experiment.run.tolerance_sec,
        experiment.run.confidence_threshold,
    )
    return OverfitDiagnosticReport(
        model_path=model_path,
        frontend=experiment.model.frontend,
        num_queries=experiment.model.num_queries,
        duration_sec=duration_sec,
        confidence_threshold=experiment.run.confidence_threshold,
        tolerance_sec=experiment.run.tolerance_sec,
        confidence=confidence_stats(pred_confidence),
        assignment=assignment_summary(
            pred_times,
            gt_times,
            gt_mask,
            experiment.run.tolerance_sec,
        ),
        eval_f1=float(f1),
        eval_tp=float(tp),
        eval_fp=float(fp),
        eval_fn=float(fn),
        pred_time_min_sec=float(np.min(pred_times)),
        pred_time_max_sec=float(np.max(pred_times)),
    )


def sweep_confidence_thresholds(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
    thresholds: tuple[float, ...] | list[float],
    min_onset_distance_ms: float | None = None,
) -> list[dict[str, float]]:
    """Evaluate F1 at multiple confidence thresholds without retraining."""
    min_gap = 0.0 if min_onset_distance_ms is None else min_onset_distance_ms
    results: list[dict[str, float]] = []
    for threshold in thresholds:
        tp, fp, fn = metrics.count_event_onset_errors_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec,
            float(threshold),
            min_gap,
        )
        _precision, _recall, f1 = metrics.event_onset_f1_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec,
            float(threshold),
            min_gap,
        )
        results.append(
            {
                "confidence_threshold": float(threshold),
                "f1": float(f1),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )
    return results
