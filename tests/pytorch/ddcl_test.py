"""Unit tests for the PyTorch DDCL ConvLSTM and MERT placement stack."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as nn_func

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import config as ddcl_config
from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.pytorch.ddcl import eval as ddcl_eval_cli
from stepcovnet.pytorch.ddcl import evaluation as ddcl_eval
from stepcovnet.pytorch.ddcl import models as ddcl_models
from stepcovnet.pytorch.ddcl import train as ddcl_train_cli
from stepcovnet.pytorch.ddcl import trainers as ddcl_trainers


def _make_synthetic_ddcl_chart(
    *,
    song_key: str = "pack/song_a",
    difficulty: str = "challenge",
    n_beats: int = 12,
    n_frames: int = 10,
    memlen: int = 3,
    feature_type: str = "logmel",
    mert_dim: int = 16,
    seed: int = 0,
) -> ddcl_datasets.DdclChart:
    """Create a tiny synthetic beat-aligned DdclChart for tests."""
    rng = np.random.default_rng(seed)
    if feature_type == "mert":
        beat_audio = rng.standard_normal((n_beats, n_frames, mert_dim)).astype(
            np.float32
        )
    else:
        beat_audio = rng.standard_normal(
            (n_beats, n_frames, ddcl_constants.N_MELS, ddcl_constants.N_CHANNELS)
        ).astype(np.float32)
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 0] = 0.5
    slots = np.zeros((n_beats, ddcl_constants.N_SLOTS), dtype=np.float32)
    slots[1::2, 0] = 1.0
    slots[2::3, 24] = 1.0
    gt_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    return ddcl_datasets.DdclChart(
        song_key=song_key,
        difficulty=difficulty,
        meter=10,
        beat_audio=beat_audio,
        stream=stream,
        slots=slots,
        memlen=memlen,
        offset_sec=0.0,
        bpm_segments=(prep_models.BpmSegment(start_beat=0.0, bpm=120.0),),
        gt_times_sec=gt_times,
    )


def test_convlstm2d_forward_backward_both_directions() -> None:
    """ConvLSTM2d produces expected valid-padding shapes and gradients."""
    for go_backwards in (False, True):
        layer = ddcl_models.ConvLSTM2d(3, 4, (3, 3), go_backwards=go_backwards)
        x = torch.randn(2, 4, 3, 6, 8)
        y = layer(x)
        # Valid padding with 3x3 kernel shrinks H and W by 2 each.
        assert y.shape == (2, 4, 4, 4, 6)
        y.pow(2).mean().backward()
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in layer.parameters()
        )


def test_ddcl_convlstm_model_forward_backward_and_round_trip(
    tmp_path: pathlib.Path,
) -> None:
    """DdclConvLstmPlacementModel forward/backward and checkpoint round-trip work."""
    model = ddcl_models.build_convlstm_placement_model(
        memlen=3,
        n_frames=16,
        lstm_units=12,
        dropout_rate=0.0,
        dense_sizes=(16, 8),
    )
    inputs = {
        "audio_fwd": torch.randn(2, 3, 16, ddcl_constants.N_MELS, 3),
        "audio_bwd": torch.randn(2, 3, 16, ddcl_constants.N_MELS, 3),
        "stream_fwd": torch.randn(2, 3, ddcl_constants.STREAM_DIM),
        "stream_bwd": torch.randn(2, 3, ddcl_constants.STREAM_DIM),
    }
    logits = model.forward_logits(inputs)
    probs = model(inputs)
    assert logits.shape == (2, ddcl_constants.N_SLOTS)
    assert logits.dtype == torch.float32
    assert probs.shape == (2, ddcl_constants.N_SLOTS)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    logits.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )

    model.eval()
    ckpt = ddcl_trainers.save_checkpoint(model, tmp_path / "ddcl_convlstm.pt")
    loaded = ddcl_trainers.load_placement_model(ckpt, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.forward_logits(inputs), model.forward_logits(inputs)
        )


def test_ddcl_mert_model_forward_backward_and_round_trip(
    tmp_path: pathlib.Path,
) -> None:
    """DdclMertPlacementModel forward/backward, noise, and checkpoint round-trip work."""
    model = ddcl_models.build_mert_placement_model(
        memlen=3,
        n_frames=6,
        feature_dim=16,
        proj_dim=12,
        conv_filters=8,
        lstm_units=12,
        dropout_rate=0.0,
        dense_sizes=(16,),
        noise_std=0.05,
    )
    inputs = {
        "audio_fwd": torch.randn(2, 3, 6, 16),
        "audio_bwd": torch.randn(2, 3, 6, 16),
        "stream_fwd": torch.randn(2, 3, ddcl_constants.STREAM_DIM),
        "stream_bwd": torch.randn(2, 3, ddcl_constants.STREAM_DIM),
    }
    model.train()
    logits = model.forward_logits(inputs)
    assert logits.shape == (2, ddcl_constants.N_SLOTS)
    logits.pow(2).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
    )

    model.eval()
    ckpt = ddcl_trainers.save_checkpoint(model, tmp_path / "ddcl_mert.pt")
    loaded = ddcl_trainers.load_placement_model(ckpt, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.forward_logits(inputs), model.forward_logits(inputs)
        )


def test_binary_focal_with_logits_loss() -> None:
    """BinaryFocalWithLogitsLoss matches BCE at gamma=0 and downweights easy samples."""
    logits = torch.tensor([[2.0, -2.0, 0.5, -0.5]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    bce_like = ddcl_models.BinaryFocalWithLogitsLoss(
        gamma=0.0,
        apply_class_balancing=False,
    )
    expected = nn_func.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(bce_like(logits, targets), expected)

    focal = ddcl_models.BinaryFocalWithLogitsLoss(
        gamma=2.0,
        apply_class_balancing=False,
    )
    assert focal(logits, targets).item() < expected.item()

    balanced = ddcl_models.BinaryFocalWithLogitsLoss(
        gamma=2.0,
        alpha=0.25,
        apply_class_balancing=True,
    )
    assert balanced(logits, targets).item() < focal(logits, targets).item()


def test_ddcl_predict_and_evaluate_slot48_mert() -> None:
    """predict_chart_slots and evaluate_slot48 run cleanly on synthetic MERT charts."""
    model = ddcl_models.build_mert_placement_model(
        memlen=3,
        n_frames=6,
        feature_dim=16,
        proj_dim=12,
        conv_filters=8,
        lstm_units=10,
        dropout_rate=0.0,
        dense_sizes=(12,),
    )
    charts = [
        _make_synthetic_ddcl_chart(
            song_key="p/a",
            difficulty="challenge",
            n_beats=10,
            n_frames=6,
            memlen=3,
            feature_type="mert",
            mert_dim=16,
            seed=1,
        ),
        _make_synthetic_ddcl_chart(
            song_key="p/b",
            difficulty="hard",
            n_beats=10,
            n_frames=6,
            memlen=3,
            feature_type="mert",
            mert_dim=16,
            seed=2,
        ),
    ]
    probs = ddcl_eval.predict_chart_slots(
        model,
        charts[0],
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert probs.shape == (10, ddcl_constants.N_SLOTS)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    report = ddcl_eval.evaluate_slot48(
        model,
        charts,
        thresholds=(0.2, 0.5, 0.8),
        device=torch.device("cpu"),
    )
    payload = report.as_dict()
    assert payload["n_charts"] == 2
    assert 0.0 <= payload["f1_at_05"] <= 1.0
    assert 0.0 <= payload["f1_max"] <= 1.0


def test_ddcl_train_placement_and_resume_smoke(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_placement trains, checkpoints, resumes, and evaluates on synthetic charts."""
    train_charts = [
        _make_synthetic_ddcl_chart(
            song_key="p/tr",
            n_beats=10,
            n_frames=6,
            memlen=3,
            feature_type="mert",
            mert_dim=16,
            seed=11,
        )
    ]
    val_charts = [
        _make_synthetic_ddcl_chart(
            song_key="p/va",
            n_beats=10,
            n_frames=6,
            memlen=3,
            feature_type="mert",
            mert_dim=16,
            seed=22,
        )
    ]
    monkeypatch.setattr(
        ddcl_datasets,
        "load_split_charts",
        lambda _cfg, split, **_kw: train_charts if split == "train" else val_charts,
    )
    exp = ddcl_config.DdclExperimentConfig(
        dataset=ddcl_config.DdclDatasetConfig(
            training_index_path=str(tmp_path / "unused.json"),
            data_root=str(tmp_path),
            batch_size=4,
            memlen=3,
            audio_feature_type="mert",
            n_frames_per_beat=6,
        ),
        model=ddcl_config.DdclModelConfig(
            lstm_units=8,
            dropout_rate=0.0,
            dense_sizes=(12,),
            mert_proj_dim=10,
            mert_conv_filters=6,
        ),
        run=ddcl_config.DdclRunConfig(
            epoch=2,
            steps_per_epoch=2,
            validation_steps=2,
            learning_rate=1e-2,
            eval_slot_f1_every=1,
            model_output_dir=str(tmp_path / "out"),
            callback_root_dir=str(tmp_path / "cb"),
            model_name="ddcl_mert_smoke",
            seed=5,
        ),
    )
    model1 = ddcl_trainers.train_placement(exp, device="cpu")
    assert isinstance(model1, ddcl_models.DdclMertPlacementModel)
    out_dir = tmp_path / "out"
    assert (out_dir / ddcl_trainers.BEST_CHECKPOINT_FILENAME).is_file()
    assert (out_dir / ddcl_trainers.BEST_SLOT_F1_CHECKPOINT_FILENAME).is_file()
    assert (out_dir / ddcl_trainers.LAST_CHECKPOINT_FILENAME).is_file()
    assert (out_dir / ddcl_trainers.LAST_EVAL_FILENAME).is_file()

    # Resume for 1 additional epoch using backup/resume.pt created during run 1.
    exp_resume = ddcl_config.DdclExperimentConfig(
        dataset=exp.dataset,
        model=exp.model,
        run=ddcl_config.DdclRunConfig(
            epoch=3,
            steps_per_epoch=2,
            validation_steps=2,
            learning_rate=1e-2,
            eval_slot_f1_every=1,
            model_output_dir=str(tmp_path / "out"),
            callback_root_dir=str(tmp_path / "cb"),
            model_name="ddcl_mert_smoke",
            seed=5,
            resume=True,
        ),
    )
    model2 = ddcl_trainers.train_placement(exp_resume, device="cpu")
    assert isinstance(model2, ddcl_models.DdclMertPlacementModel)


def test_ddcl_cli_parsers() -> None:
    """CLI argument parsers accept expected flags."""
    train_args = ddcl_train_cli._build_parser().parse_args(
        ["--config", "cfg.json", "--device", "cpu", "--fresh"]
    )
    assert train_args.config == "cfg.json"
    assert train_args.fresh is True

    eval_args = ddcl_eval_cli._build_parser().parse_args(
        ["--model", "best.pt", "--config", "cfg.json", "--split", "val"]
    )
    assert eval_args.model_path == "best.pt"
    assert eval_args.split == "val"


def test_ddcl_no_tensorflow_imported() -> None:
    """Importing stepcovnet.pytorch.ddcl modules never imports TensorFlow or Keras."""
    code = (
        "import sys; "
        "import stepcovnet.pytorch.ddcl.models; "
        "import stepcovnet.pytorch.ddcl.evaluation; "
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
