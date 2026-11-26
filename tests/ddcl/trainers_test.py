"""Canonical PyTorch DDCL checkpoint save/load tests via ``stepcovnet.ddcl.trainers``."""

from __future__ import annotations

import pathlib

import torch

from stepcovnet.ddcl import constants, models, trainers


def test_canonical_ddcl_save_and_load_pt_checkpoint(tmp_path: pathlib.Path) -> None:
    memlen = 1
    seq = memlen + 1
    model = models.build_mert_placement_model(
        memlen=memlen,
        n_frames=8,
        feature_dim=16,
        proj_dim=8,
        conv_filters=8,
        lstm_units=8,
        dropout_rate=0.0,
        dense_sizes=(8,),
    )
    ckpt = tmp_path / "ddcl_test.pt"
    trainers.save_checkpoint(model, ckpt)
    reloaded = trainers.load_placement_model(ckpt, device="cpu")
    inputs = {
        "audio_fwd": torch.randn(1, seq, 8, 16),
        "audio_bwd": torch.randn(1, seq, 8, 16),
        "stream_fwd": torch.randn(1, seq, constants.STREAM_DIM),
        "stream_bwd": torch.randn(1, seq, constants.STREAM_DIM),
    }
    model.eval()
    with torch.no_grad():
        a = model(inputs)
        b = reloaded(inputs)
    assert torch.allclose(a, b, atol=1e-6)
