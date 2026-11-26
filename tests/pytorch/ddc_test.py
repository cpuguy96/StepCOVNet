"""Unit tests for the PyTorch DDC C-LSTM placement stack."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from stepcovnet.ddc import config as ddc_config
from stepcovnet.ddc import constants as ddc_constants
from stepcovnet.ddc import datasets as ddc_datasets
from stepcovnet.pytorch.ddc import eval as ddc_eval_cli
from stepcovnet.pytorch.ddc import evaluation as ddc_eval
from stepcovnet.pytorch.ddc import models as ddc_models
from stepcovnet.pytorch.ddc import train as ddc_train_cli
from stepcovnet.pytorch.ddc import trainers as ddc_trainers


def _make_synthetic_chart(
    *,
    song_key: str = "pack/song_a",
    difficulty: str = "challenge",
    n_frames: int = 80,
    seed: int = 0,
) -> ddc_datasets.PlacementChart:
    """Create a tiny synthetic 10 ms PlacementChart for unit tests."""
    rng = np.random.default_rng(seed)
    spec = rng.standard_normal((n_frames, ddc_constants.N_MELS, 3)).astype(np.float32)
    target = np.zeros((n_frames,), dtype=np.float32)
    onset_idx = np.array([15, 35, 55, 65], dtype=np.int64)
    target[onset_idx] = 1.0
    gt_times = onset_idx.astype(np.float64) * ddc_constants.HOP_SEC
    return ddc_datasets.PlacementChart(
        song_key=song_key,
        difficulty=difficulty,
        spec=spec,
        target=target,
        gt_times=gt_times,
        first_onset=int(onset_idx[0]),
        last_onset=int(onset_idx[-1]),
    )


def test_ddc_model_forward_backward_and_dynamic_length() -> None:
    """DdcClstmPlacementModel supports dynamic time lengths and flows gradients."""
    model = ddc_models.build_clstm_placement_model(
        lstm_units=16,
        dnn_sizes=(24, 12),
        dropout_rate=0.1,
    )
    for seq_len in (6, 20):
        audio = torch.randn(2, seq_len, 15, ddc_constants.N_MELS, 3)
        diff = torch.zeros(2, seq_len, ddc_constants.N_DIFFICULTIES)
        diff[:, :, 2] = 1.0
        inputs = {"audio": audio, "difficulty": diff}
        logits = model.forward_logits(inputs)
        probs = model(inputs)
        assert logits.shape == (2, seq_len, 1)
        assert logits.dtype == torch.float32
        assert probs.shape == (2, seq_len, 1)
        assert torch.all((probs >= 0.0) & (probs <= 1.0))
        loss = logits.pow(2).mean()
        loss.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        model.zero_grad()


def test_ddc_checkpoint_round_trip(tmp_path: pathlib.Path) -> None:
    """save_checkpoint and load_placement_model reproduce exact outputs."""
    model = ddc_models.build_clstm_placement_model(
        lstm_units=12,
        dnn_sizes=(16,),
        dropout_rate=0.0,
    )
    model.eval()
    ckpt_path = ddc_trainers.save_checkpoint(model, tmp_path / "ddc.pt")
    loaded = ddc_trainers.load_placement_model(ckpt_path, device="cpu")

    inputs = {
        "audio": torch.randn(1, 8, 15, ddc_constants.N_MELS, 3),
        "difficulty": torch.ones(1, 8, ddc_constants.N_DIFFICULTIES) / 5.0,
    }
    with torch.no_grad():
        ref = model.forward_logits(inputs)
        got = loaded.forward_logits(inputs)
    torch.testing.assert_close(got, ref)


def test_ddc_predict_and_evaluate_placement() -> None:
    """predict_chart_salience and evaluate_placement run cleanly on synthetic charts."""
    model = ddc_models.build_clstm_placement_model(
        lstm_units=8,
        dnn_sizes=(16,),
        dropout_rate=0.0,
    )
    charts = [
        _make_synthetic_chart(song_key="p/a", difficulty="challenge", seed=1),
        _make_synthetic_chart(song_key="p/b", difficulty="beginner", seed=2),
    ]
    salience = ddc_eval.predict_chart_salience(
        model,
        charts[0],
        device=torch.device("cpu"),
    )
    assert salience.shape == (charts[0].n_frames,)
    assert np.all((salience >= 0.0) & (salience <= 1.0))

    report = ddc_eval.evaluate_placement(
        model,
        charts,
        thresholds=(0.1, 0.3, 0.5, 0.7),
        device=torch.device("cpu"),
    )
    payload = report.as_dict()
    assert payload["n_charts"] == 2
    assert 0.0 <= payload["f_score_m"] <= 1.0
    assert 0.0 <= payload["f_score_c"] <= 1.0
    assert set(payload["per_difficulty_threshold"].keys()) == {"beginner", "challenge"}


def test_ddc_train_placement_smoke(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train_placement runs a 2-epoch CPU smoke loop and writes checkpoints/reports."""
    train_charts = [_make_synthetic_chart(song_key="p/tr", seed=10)]
    val_charts = [_make_synthetic_chart(song_key="p/va", seed=20)]

    monkeypatch.setattr(
        ddc_datasets,
        "load_split_charts",
        lambda _cfg, split, **_kw: train_charts if split == "train" else val_charts,
    )
    exp = ddc_config.PlacementExperimentConfig(
        dataset=ddc_config.PlacementDatasetConfig(
            training_index_path=str(tmp_path / "unused.json"),
            data_root=str(tmp_path),
            batch_size=2,
            nunroll=12,
        ),
        model=ddc_config.PlacementModelConfig(
            lstm_units=8,
            dnn_sizes=(12,),
            dropout_rate=0.0,
        ),
        run=ddc_config.PlacementRunConfig(
            epoch=2,
            steps_per_epoch=2,
            validation_steps=2,
            learning_rate=1e-2,
            model_output_dir=str(tmp_path / "out"),
            callback_root_dir=str(tmp_path / "cb"),
            model_name="ddc_smoke",
            seed=7,
        ),
    )
    trained = ddc_trainers.train_placement(exp, device="cpu")
    assert isinstance(trained, ddc_models.DdcClstmPlacementModel)
    out_dir = tmp_path / "out"
    assert (out_dir / ddc_trainers.BEST_CHECKPOINT_FILENAME).is_file()
    assert (out_dir / ddc_trainers.LAST_EVAL_FILENAME).is_file()
    assert (out_dir / ddc_trainers.BEST_EVAL_FILENAME).is_file()


def test_ddc_cli_parsers() -> None:
    """CLI argument parsers accept expected flags."""
    train_args = ddc_train_cli._build_parser().parse_args(
        ["--config", "cfg.json", "--device", "cpu", "--epochs", "4"]
    )
    assert train_args.config == "cfg.json"
    assert train_args.device == "cpu"
    assert train_args.epochs == 4

    eval_args = ddc_eval_cli._build_parser().parse_args(
        ["--model", "best.pt", "--config", "cfg.json", "--split", "val"]
    )
    assert eval_args.model_path == "best.pt"
    assert eval_args.split == "val"


def test_ddc_no_tensorflow_imported() -> None:
    """Importing stepcovnet.pytorch.ddc modules never imports TensorFlow or Keras."""
    code = (
        "import sys; "
        "import stepcovnet.pytorch.ddc.models; "
        "import stepcovnet.pytorch.ddc.evaluation; "
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
