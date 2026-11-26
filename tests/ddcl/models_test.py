"""Canonical PyTorch DDCL placement model tests via ``stepcovnet.ddcl.models``."""

from __future__ import annotations

import torch

from stepcovnet.ddcl import constants, models


def test_canonical_ddcl_convlstm_forward() -> None:
    memlen = 1
    seq = memlen + 1
    model = models.build_convlstm_placement_model(
        memlen=memlen,
        n_frames=12,
        lstm_units=8,
        dropout_rate=0.0,
        dense_sizes=(16,),
    )
    model.eval()
    inputs = {
        "audio_fwd": torch.randn(2, seq, 12, constants.N_MELS, constants.N_CHANNELS),
        "audio_bwd": torch.randn(2, seq, 12, constants.N_MELS, constants.N_CHANNELS),
        "stream_fwd": torch.randn(2, seq, constants.STREAM_DIM),
        "stream_bwd": torch.randn(2, seq, constants.STREAM_DIM),
    }
    with torch.no_grad():
        out = model(inputs)
    assert tuple(out.shape) == (2, constants.N_SLOTS)
    assert bool((out >= 0.0).all() and (out <= 1.0).all())


def test_canonical_ddcl_mert_forward() -> None:
    memlen = 1
    seq = memlen + 1
    model = models.build_mert_placement_model(
        memlen=memlen,
        n_frames=8,
        feature_dim=32,
        proj_dim=16,
        conv_filters=16,
        lstm_units=8,
        dropout_rate=0.0,
        dense_sizes=(16,),
    )
    model.eval()
    inputs = {
        "audio_fwd": torch.randn(2, seq, 8, 32),
        "audio_bwd": torch.randn(2, seq, 8, 32),
        "stream_fwd": torch.randn(2, seq, constants.STREAM_DIM),
        "stream_bwd": torch.randn(2, seq, constants.STREAM_DIM),
    }
    with torch.no_grad():
        out = model(inputs)
    assert tuple(out.shape) == (2, constants.N_SLOTS)
    assert bool((out >= 0.0).all() and (out <= 1.0).all())
