"""``M-slot48`` evaluation and audio-grounding ablations for ITGPT (PyTorch)."""

from __future__ import annotations

from stepcovnet.pytorch.itgpt.evaluation import (
    CORRUPTION_KINDS,
    DEFAULT_THRESHOLDS,
    AudioGroundingConditionReport,
    AudioGroundingReport,
    Slot48Counts,
    Slot48EvalReport,
    SlotPredictor,
    TorchSlotPredictor,
    corrupt_chart_audio,
    counts_at_threshold,
    evaluate_audio_grounding,
    evaluate_slot48,
    evaluate_slot48_slices,
    predict_chart_slots,
    report_as_dict,
    shuffle_slot_null,
)

__all__ = [
    "CORRUPTION_KINDS",
    "DEFAULT_THRESHOLDS",
    "AudioGroundingConditionReport",
    "AudioGroundingReport",
    "Slot48Counts",
    "Slot48EvalReport",
    "SlotPredictor",
    "TorchSlotPredictor",
    "corrupt_chart_audio",
    "counts_at_threshold",
    "evaluate_audio_grounding",
    "evaluate_slot48",
    "evaluate_slot48_slices",
    "predict_chart_slots",
    "report_as_dict",
    "shuffle_slot_null",
]
