"""Unit tests for the PyTorch AR onset Transformer stack and O(L) KV cache."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from stepcovnet.pytorch.onset_ar import (
    config as ar_config,
)
from stepcovnet.pytorch.onset_ar import (
    datasets as ar_datasets,
)
from stepcovnet.pytorch.onset_ar import (
    eval as ar_eval_cli,
)
from stepcovnet.pytorch.onset_ar import (
    inference as ar_inference,
)
from stepcovnet.pytorch.onset_ar import (
    losses as ar_losses,
)
from stepcovnet.pytorch.onset_ar import (
    models as ar_models,
)
from stepcovnet.pytorch.onset_ar import (
    targets as ar_targets,
)
from stepcovnet.pytorch.onset_ar import (
    train as ar_train_cli,
)
from stepcovnet.pytorch.onset_ar import (
    trainers as ar_trainers,
)


def _tiny_config(
    *,
    alignment: str = "pointer_residual",
    density_conditioning: str = "none",
    pointer_head: str = "content",
    gap_head: str = "content",
    decoder_cross_content_only: bool = False,
) -> ar_config.ArExperimentConfig:
    return ar_config.ArExperimentConfig(
        dataset=ar_config.ArDatasetConfig(
            batch_size=1,
            max_audio_seconds=4.0,
            max_steps_per_chart=16,
            hop_sec=0.02,
        ),
        model=ar_config.ArModelConfig(
            patch_frames=4,
            d_model=32,
            n_enc_layers=2,
            n_dec_layers=2,
            num_heads=4,
            dropout_rate=0.0,
            alignment=alignment,
            pointer_head=pointer_head,
            gap_head=gap_head,
            max_decode_steps=16,
            delta_max_dense=16,
            n_log_buckets=4,
            n_first_abs_bins=8,
            patch_delta_max_dense=16,
            patch_n_log_buckets=4,
            density_conditioning=density_conditioning,
            decoder_cross_content_only=decoder_cross_content_only,
            mert_hidden_size=8,
        ),
        run=ar_config.ArRunConfig(
            epochs=3,
            overfit_one_song=True,
            learning_rate=5e-3,
            lambda_time=0.5,
            lambda_residual=1.0,
            seed=7,
        ),
    )


def _make_synthetic_sample(
    cfg: ar_config.ArExperimentConfig,
    *,
    n_patches: int = 12,
    seed: int = 0,
) -> ar_datasets.ArSample:
    rng = np.random.default_rng(seed)
    patch_dim = cfg.patch_input_dim()
    mert_patches = rng.standard_normal((n_patches, patch_dim)).astype(np.float32)
    n_frames = n_patches * cfg.model.patch_frames
    duration_sec = float(n_frames) * cfg.dataset.hop_sec
    gt_times = np.array([0.12, 0.28, 0.44, 0.62, 0.80], dtype=np.float32)
    token_seq = ar_targets.encode_onset_times(
        gt_times,
        duration_sec=duration_sec,
        hop_sec=cfg.dataset.hop_sec,
        patch_frames=cfg.model.patch_frames,
        vocab=cfg.build_vocab(),
        gap_vocab=cfg.build_gap_vocab(),
        max_steps=cfg.dataset.max_steps_per_chart,
    )
    return ar_datasets.ArSample(
        mert_patches=mert_patches,
        n_patches=n_patches,
        n_frames=n_frames,
        duration_sec=duration_sec,
        token_seq=token_seq,
        gt_times_sec=gt_times,
        audio_path="synthetic.wav",
        chart_path="synthetic.json",
        meter=10,
        density_scalar=0.4,
    )


def test_vocab_and_targets_round_trip() -> None:
    """DeltaBucketVocab and PatchGapVocab round-trip dense and overflow targets."""
    vocab = ar_targets.DeltaBucketVocab(
        delta_max_dense=32, n_log_buckets=8, n_first_abs_bins=16, hop_sec=0.02
    )
    gap_vocab = ar_targets.PatchGapVocab(delta_max_dense=32, n_log_buckets=8)
    times = np.array([0.10, 0.24, 0.50, 1.20], dtype=np.float64)
    seq = ar_targets.encode_onset_times(
        times,
        duration_sec=2.0,
        hop_sec=0.02,
        patch_frames=4,
        vocab=vocab,
        gap_vocab=gap_vocab,
    )
    assert seq.n_steps == 4
    assert int(seq.decoder_input_ids[0]) == ar_targets.BOS_ID
    assert int(seq.decoder_target_ids[-1]) == ar_targets.EOS_ID
    decoded_patches = ar_targets.decode_gap_ids_to_patch_indices(
        seq.gap_token_ids, gap_vocab=gap_vocab, max_patch=100
    )
    np.testing.assert_array_equal(decoded_patches, seq.patch_indices)
    decoded_times = ar_targets.decode_token_sequence_to_times(
        seq.token_ids, hop_sec=0.02, vocab=vocab, max_frame=100
    )
    assert decoded_times.size == 4


def test_existing_repo_ar_configs_parse() -> None:
    """Existing repo configs/ar JSON files load cleanly with PyTorch ArExperimentConfig."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for rel in ("configs/ar/tide_overfit.json", "configs/ar/tide_gap_residual.json"):
        cfg_path = repo_root / rel
        if cfg_path.is_file():
            exp = ar_config.ArExperimentConfig.from_json(cfg_path)
            assert exp.model.d_model > 0
            assert exp.build_vocab().vocab_size > 10


@pytest.mark.parametrize(
    ("alignment", "density"),
    [
        ("pointer_residual", "none"),
        ("pointer_residual", "onset_density"),
        ("gap_residual", "none"),
        ("gap_residual", "meter"),
    ],
)
def test_kv_cache_matches_full_parallel_forward_exact(
    alignment: str,
    density: str,
) -> None:
    """O(L) KV-cached step-by-step decode matches full parallel forward within 1e-5."""
    torch.manual_seed(123)
    cfg = _tiny_config(alignment=alignment, density_conditioning=density)
    model = ar_models.build_ar_onset_model(cfg)
    model.eval()

    sample = _make_synthetic_sample(cfg, n_patches=10, seed=3)
    batch = ar_datasets.collate_ar_samples([sample], dynamic_padding=True)
    # Add 2 padded encoder patches to ensure padding masks match identically.
    pad_mert = torch.cat(
        [batch["mert_patches"], torch.randn(1, 2, cfg.patch_input_dim())], dim=1
    )
    pad_mask = torch.cat([batch["patch_mask"], torch.zeros(1, 2)], dim=1)
    batch["mert_patches"] = pad_mert
    batch["patch_mask"] = pad_mask

    with torch.no_grad():
        full_out = model(batch)
        cache = model.init_kv_cache(
            pad_mert,
            pad_mask,
            max_decoder_len=int(batch["decoder_input_ids"].shape[1]),
            density_scalar=batch["density_scalar"] if density != "none" else None,
        )
        seq_len = int(batch["decoder_mask"][0].sum().item())
        for pos in range(seq_len):
            tok = batch["decoder_input_ids"][:, pos : pos + 1]
            prev = (
                batch["prev_patch_indices"][:, pos : pos + 1]
                if model.content_gap
                else None
            )
            step_out = model.decode_step_cached(
                tok, pos, cache, prev_patch_indices=prev
            )
            torch.testing.assert_close(
                step_out["token_logits"][:, 0, :],
                full_out["token_logits"][:, pos, :],
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                step_out["residual_sec"][:, 0],
                full_out["residual_sec"][:, pos],
                atol=1e-5,
                rtol=1e-5,
            )
            if "pointer_logits" in full_out:
                torch.testing.assert_close(
                    step_out["pointer_logits"][:, 0, :],
                    full_out["pointer_logits"][:, pos, :],
                    atol=1e-5,
                    rtol=1e-5,
                )
            if "gap_logits" in full_out:
                torch.testing.assert_close(
                    step_out["gap_logits"][:, 0, :],
                    full_out["gap_logits"][:, pos, :],
                    atol=1e-5,
                    rtol=1e-5,
                )


def test_losses_and_gradients_across_variants() -> None:
    """Forward, loss, and backward work for pointer STE, local CE, and gap modes."""
    for alignment, ste, radius, anchor, alpha in (
        ("pointer_residual", True, 0, "target", 0.05),
        ("pointer_residual", False, 4, "prev", 0.0),
        ("gap_residual", False, 0, "target", 0.02),
    ):
        cfg = _tiny_config(alignment=alignment)
        model = ar_models.build_ar_onset_model(cfg)
        sample = _make_synthetic_sample(cfg, seed=5)
        batch = ar_datasets.collate_ar_samples([sample], dynamic_padding=True)
        outputs = model(batch)
        total, parts = ar_losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=cfg.model.patch_frames,
            hop_sec=cfg.dataset.hop_sec,
            lambda_time=0.5,
            lambda_residual=1.0,
            use_ste_pointer_time=ste,
            pointer_local_ce_radius=radius,
            pointer_local_ce_anchor=anchor,
            pointer_soft_distance_alpha=alpha,
            monotonic_pointer=True,
            gap_alignment=model.gap_alignment,
            gap_delta_lookup=model.gap_delta_lookup,
        )
        assert torch.isfinite(total)
        assert all(torch.isfinite(v) for v in parts.values())
        total.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )


def test_train_ar_onset_and_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    """train_ar_onset runs CPU smoke training, saves checkpoints, and evaluates KV decode."""
    cfg = _tiny_config(alignment="pointer_residual")
    cfg.run.model_output_dir = str(tmp_path / "out")
    cfg.run.callback_root_dir = str(tmp_path / "cb")
    cfg.run.run_label = "ar_smoke"
    cfg.run.epochs = 3
    sample = _make_synthetic_sample(cfg, seed=9)

    res = ar_trainers.train_ar_onset(
        cfg,
        train_samples=[sample],
        val_samples=[sample],
        device="cpu",
        eval_val=True,
    )
    assert len(res.history["loss"]) == 3
    assert res.history["loss"][-1] < res.history["loss"][0]
    assert res.val_report is not None
    assert (tmp_path / "out" / ar_trainers.BEST_CHECKPOINT_FILENAME).is_file()
    assert (tmp_path / "out" / ar_trainers.LAST_CHECKPOINT_FILENAME).is_file()
    assert (tmp_path / "out" / ar_trainers.EVAL_REPORT_FILENAME).is_file()

    loaded = ar_trainers.load_ar_onset_model(
        tmp_path / "out" / ar_trainers.LAST_CHECKPOINT_FILENAME, device="cpu"
    )
    patch_mask = np.ones((sample.n_patches,), dtype=np.float32)
    stats_a = ar_inference.decode_autoregressive_kv(
        res.model,
        sample.mert_patches,
        patch_mask,
        max_decoder_len=10,
        device=torch.device("cpu"),
    )
    stats_b = ar_inference.decode_autoregressive_kv(
        loaded,
        sample.mert_patches,
        patch_mask,
        max_decoder_len=10,
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(stats_a.times, stats_b.times, atol=1e-5)


def test_onset_ar_cli_and_no_tensorflow_imported() -> None:
    """CLI parsers work and importing stepcovnet.pytorch.onset_ar pulls zero TF/Keras."""
    tr_args = ar_train_cli._build_parser().parse_args(
        ["--config", "cfg.json", "--epochs", "2", "--device", "cpu", "--no_eval_val"]
    )
    assert tr_args.epochs == 2
    assert tr_args.no_eval_val is True

    ev_args = ar_eval_cli._build_parser().parse_args(
        ["--model", "best.pt", "--config", "cfg.json", "--split", "val"]
    )
    assert ev_args.model_path == "best.pt"

    code = (
        "import sys; "
        "import stepcovnet.pytorch.onset_ar; "
        "assert 'tensorflow' not in sys.modules; "
        "assert 'keras' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
