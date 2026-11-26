"""Canonical PyTorch AR onset smoke tests via ``stepcovnet.onset_ar``."""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from stepcovnet.onset_ar import config, datasets, inference, losses, models, trainers
from stepcovnet.pytorch.onset_ar import targets as ar_targets


def test_canonical_onset_ar_forward_loss_and_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    exp = config.ArExperimentConfig(
        dataset=config.ArDatasetConfig(max_steps_per_chart=12),
        model=config.ArModelConfig(
            d_model=16,
            num_heads=2,
            n_enc_layers=1,
            n_dec_layers=1,
            patch_frames=4,
            dropout_rate=0.0,
            mert_hidden_size=16,
        ),
        run=config.ArRunConfig(epochs=1, model_output_dir=str(tmp_path)),
    )
    model = models.build_ar_onset_model(exp)
    vocab = exp.build_vocab()
    gap_vocab = exp.build_gap_vocab()
    times = np.array([0.2, 0.6, 1.0], dtype=np.float64)
    token_seq = ar_targets.encode_onset_times(
        times,
        duration_sec=2.0,
        hop_sec=exp.dataset.hop_sec,
        patch_frames=exp.model.patch_frames,
        vocab=vocab,
        gap_vocab=gap_vocab,
    )
    sample = datasets.ArSample(
        mert_patches=np.random.randn(20, exp.patch_input_dim()).astype(np.float32),
        n_patches=20,
        n_frames=80,
        duration_sec=2.0,
        token_seq=token_seq,
        gt_times_sec=times,
        audio_path="a.ogg",
        chart_path="a.txt",
    )
    batch_t = datasets.collate_ar_samples([sample])
    out = model(batch_t)
    total_loss, _ = losses.compute_ar_onset_loss(
        out,
        batch_t,
        patch_frames=exp.model.patch_frames,
        hop_sec=exp.dataset.hop_sec,
    )
    assert torch.isfinite(total_loss)

    ckpt = tmp_path / "ar.pt"
    trainers.save_checkpoint(model, ckpt)
    reloaded = trainers.load_ar_onset_model(ckpt, device="cpu")
    stats = inference.decode_autoregressive_kv(
        reloaded,
        sample.mert_patches,
        np.ones((sample.n_patches,), dtype=np.float32),
        max_decoder_len=6,
    )
    assert stats.n_forward_steps >= 1
