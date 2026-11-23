"""``M-slot48`` eval for ITGPT full-chart placement (`omalley2026itgpt`)."""

from __future__ import annotations

import numpy as np

from stepcovnet import onset_null_baseline, timing_match
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import evaluation as ddcl_eval
from stepcovnet.itgpt import constants, datasets


def predict_chart_slots(
    model: ddcl_eval.SlotPredictor,
    chart: ddcl_datasets.DdclChart,
    *,
    max_beats: int | None = None,
) -> np.ndarray:
    """Return slot probabilities for one chart (true beats only).

    Args:
        model: Keras ITGPT placement model.
        chart: Loaded beat-grid chart.
        max_beats: Optional pad cap used at train time. Defaults to the
            chart's chunk-aligned length.

    Returns:
        Probabilities ``(n_beats, 48)``.
    """
    cap = (
        max(datasets.pad_length(chart.n_beats, chart.n_beats), constants.CHUNK_ALIGN)
        if max_beats is None
        else max(int(max_beats), constants.CHUNK_ALIGN)
    )
    inputs, _, mask = datasets.pack_chart(chart, max_beats=cap)
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


def evaluate_audio_grounding(
    model: ddcl_eval.SlotPredictor,
    charts: list[ddcl_datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = ddcl_eval.DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    max_beats: int = constants.MAX_BEATS,
    seed: int = onset_null_baseline.DEFAULT_SEED,
) -> ddcl_eval.AudioGroundingReport:
    """Run ``none``, ``shuffle``, and ``zeros`` beat-audio grounding checks.

    Corrupts ``beat_audio`` while leaving scalar ``bpm``, ``difficulty``, and
    positional embeddings intact to measure how much ITGPT's slot and decoded
    ±20 ms skill relies on audio rather than priors.

    Args:
        model: Keras ITGPT placement model.
        charts: Loaded beat-grid charts with BPM segments and onset times.
        threshold: Occupancy cutoff for slot F1, decoded timing, and flip rates.
        thresholds: Sweep for ``slot_f1_max``.
        tolerance_sec: Match window in seconds.
        max_beats: Pad cap used at train time.
        seed: Base RNG seed for shuffling and null baselines.

    Returns:
        Audio-grounding ablation report across all corruption modes.

    Raises:
        ValueError: If ``charts`` is empty or any chart exceeds ``max_beats``.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    for chart in charts:
        if chart.n_beats > max_beats:
            raise ValueError(
                f"chart {chart.song_key} has {chart.n_beats} beats > "
                f"max_beats={max_beats}"
            )
    predictions_by_corruption: dict[str, list[np.ndarray]] = {}
    for kind in ddcl_eval.CORRUPTION_KINDS:
        predictions_by_corruption[kind] = [
            predict_chart_slots(
                model,
                ddcl_eval.corrupt_chart_audio(chart, kind, seed=seed + idx),
                max_beats=max_beats,
            )
            for idx, chart in enumerate(charts)
        ]
    return ddcl_eval.grounding_report_from_predictions(
        charts,
        predictions_by_corruption,
        threshold=threshold,
        thresholds=thresholds,
        tolerance_sec=tolerance_sec,
        seed=seed,
    )
