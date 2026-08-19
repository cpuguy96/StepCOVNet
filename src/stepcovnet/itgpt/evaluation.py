"""``M-slot48`` eval for ITGPT full-chart placement (`omalley2026itgpt`)."""

from __future__ import annotations

import numpy as np

from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import evaluation as ddcl_eval
from stepcovnet.itgpt import constants, datasets


def predict_chart_slots(
    model: ddcl_eval.SlotPredictor, chart: ddcl_datasets.DdclChart
) -> np.ndarray:
    """Return slot probabilities for one chart (true beats only).

    Args:
        model: Keras ITGPT placement model.
        chart: Loaded beat-grid chart.

    Returns:
        Probabilities ``(n_beats, 48)``.
    """
    max_beats = max(
        datasets.pad_length(chart.n_beats, chart.n_beats), constants.CHUNK_ALIGN
    )
    # Use a cap large enough for this chart; trainers pass the config cap.
    inputs, _, mask = datasets.pack_chart(
        chart, max_beats=max(max_beats, constants.CHUNK_ALIGN)
    )
    pred = np.asarray(model.predict(inputs, verbose=0), dtype=np.float32)
    if pred.ndim == 2:
        pred = pred[None, ...]
    n_beats = int(mask[0].sum())
    return pred[0, :n_beats]


def evaluate_slot48(
    model: ddcl_eval.SlotPredictor,
    charts: list[ddcl_datasets.DdclChart],
    *,
    seed: int = 42,
    max_beats: int = constants.MAX_BEATS,
) -> ddcl_eval.Slot48EvalReport:
    """Pool ``M-slot48`` over full-chart ITGPT predictions.

    Args:
        model: Trained placement model.
        charts: Loaded val charts.
        seed: Null-shuffle seed.
        max_beats: Pad cap used at train time.

    Returns:
        Pooled report.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    rng = np.random.default_rng(seed)
    pred_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    for chart in charts:
        inputs, _, mask = datasets.pack_chart(chart, max_beats=max_beats)
        pred = np.asarray(model.predict(inputs, verbose=0), dtype=np.float32)
        if pred.ndim == 2:
            pred = pred[None, ...]
        n_beats = int(mask[0].sum())
        pred_parts.append(pred[0, :n_beats])
        tgt_parts.append(chart.slots[:n_beats])
    pred_all = np.concatenate(pred_parts, axis=0)
    tgt_all = np.concatenate(tgt_parts, axis=0)
    counts_05 = ddcl_eval.counts_at_threshold(pred_all, tgt_all, constants.THRESHOLD_05)
    best_threshold = constants.THRESHOLD_05
    best_f1 = counts_05.f_score
    for threshold in ddcl_eval.DEFAULT_THRESHOLDS:
        score = ddcl_eval.counts_at_threshold(pred_all, tgt_all, threshold).f_score
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    null_pred = ddcl_eval.shuffle_slot_null(tgt_all, rng)
    null_counts = ddcl_eval.counts_at_threshold(
        null_pred, tgt_all, constants.THRESHOLD_05
    )
    return ddcl_eval.Slot48EvalReport(
        f1_at_05=counts_05.f_score,
        f1_max=best_f1,
        best_threshold=best_threshold,
        null_f1_at_05=null_counts.f_score,
        n_charts=len(charts),
        n_beats=int(tgt_all.shape[0]),
        counts_at_05=counts_05,
    )


def report_as_dict(report: ddcl_eval.Slot48EvalReport, *, weights: str) -> dict:
    """Serialize a report with ITGPT Table 2 citations.

    Args:
        report: Pooled ``M-slot48`` scores.
        weights: ``last`` or ``best``.

    Returns:
        JSON-serializable mapping.
    """
    payload = report.as_dict()
    payload["citation"] = "omalley2026itgpt"
    payload["published_f1_at_05_expanded_fraxtil"] = (
        constants.PUBLISHED_F1_AT_05_EXPANDED
    )
    payload["published_f1_max_expanded_fraxtil"] = constants.PUBLISHED_F1_MAX_EXPANDED
    payload["weights"] = weights
    return payload
