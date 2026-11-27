"""ITGPT hierarchical transformer beat-slot placement models (PyTorch)."""

from __future__ import annotations

from stepcovnet.pytorch.itgpt.models import (
    ItgptBeatAuxHead,
    ItgptBeatCNNEncoder,
    ItgptBeatMertEncoder,
    ItgptConditionFilm,
    ItgptConditionProj,
    ItgptConvHead,
    ItgptEncoderBlock,
    ItgptGatedCompression,
    ItgptHierarchicalAttnBlock,
    ItgptNormlessEncoderBlock,
    ItgptPlacementLoss,
    ItgptPlacementModel,
    ItgptRmsNorm,
    build_itgpt_placement_model,
    compute_beat_aux_loss,
    compute_bernoulli_rdrop_kl,
    grid_importance_weights,
)

__all__ = [
    "ItgptBeatAuxHead",
    "ItgptBeatCNNEncoder",
    "ItgptBeatMertEncoder",
    "ItgptConditionFilm",
    "ItgptConditionProj",
    "ItgptConvHead",
    "ItgptEncoderBlock",
    "ItgptGatedCompression",
    "ItgptHierarchicalAttnBlock",
    "ItgptNormlessEncoderBlock",
    "ItgptPlacementLoss",
    "ItgptPlacementModel",
    "ItgptRmsNorm",
    "build_itgpt_placement_model",
    "compute_beat_aux_loss",
    "compute_bernoulli_rdrop_kl",
    "grid_importance_weights",
]
