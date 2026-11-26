"""Tests for PyTorch DDC placement trainers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as nn_func

from stepcovnet.ddc import config, datasets, models, trainers
from stepcovnet.pytorch import runtime


def _synthetic_chart() -> datasets.PlacementChart:
    """Return a short in-memory chart for trainer tests."""
    n_frames = 80
    spec = np.zeros((n_frames, 80, 3), dtype=np.float32)
    target = np.zeros((n_frames,), dtype=np.float32)
    target[10] = 1.0
    target[70] = 1.0
    return datasets.PlacementChart(
        song_key="bundle/song",
        difficulty="easy",
        spec=spec,
        target=target,
        gt_times=np.array([0.10, 0.70], dtype=np.float32),
        first_onset=10,
        last_onset=70,
    )


class DdcTrainersTest(unittest.TestCase):
    def test_train_step_and_checkpoint_round_trip(self) -> None:
        runtime.set_seed(0)
        model = models.build_clstm_placement_model(
            lstm_units=8,
            lstm_layers=1,
            dropout_rate=0.0,
            dnn_sizes=(8,),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        rng = np.random.default_rng(0)
        inputs_np, labels_np = datasets.sample_train_batch(
            [_synthetic_chart()],
            batch_size=2,
            nunroll=32,
            rng=rng,
        )
        inputs = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in inputs_np.items()
        }
        labels = torch.as_tensor(labels_np, dtype=torch.float32)
        optimizer.zero_grad()
        logits = model.forward_logits(inputs)
        loss = nn_func.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        self.assertTrue(bool(torch.isfinite(loss)))

        model.eval()
        with torch.no_grad():
            expected = model(inputs).numpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = trainers.save_checkpoint(model, pathlib.Path(tmpdir) / "ddc.pt")
            reloaded = trainers.load_placement_model(ckpt, device="cpu")
            with torch.no_grad():
                actual = reloaded(inputs).numpy()
        np.testing.assert_allclose(expected, actual, atol=1e-6)

    def test_tensorboard_run_log_dir_is_timestamped_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = trainers.tensorboard_run_log_dir(tmpdir, "smoke")
            self.assertEqual(log_dir.parent, pathlib.Path(tmpdir) / "logs")
            self.assertTrue(log_dir.name.endswith("-smoke"))
            self.assertTrue(log_dir.is_dir())

    def test_train_placement_writes_artifacts(self) -> None:
        chart = _synthetic_chart()
        report = mock.Mock()
        report.as_dict.return_value = {"f_score_c": 0.1, "f_score_m": 0.2}
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.PlacementExperimentConfig(
                dataset=config.PlacementDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    batch_size=2,
                    nunroll=32,
                ),
                model=config.PlacementModelConfig(
                    lstm_units=8,
                    lstm_layers=1,
                    dropout_rate=0.0,
                    dnn_sizes=[8],
                ),
                run=config.PlacementRunConfig(
                    epoch=1,
                    steps_per_epoch=1,
                    validation_steps=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir=str(pathlib.Path(tmpdir) / "callbacks"),
                    model_name="smoke",
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.ddc.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.pytorch.ddc.trainers.evaluation.evaluate_placement",
                    return_value=report,
                ) as mock_eval,
            ):
                trainers.train_placement(experiment, device="cpu")
            saved = pathlib.Path(tmpdir) / "models" / "smoke.pt"
            best = pathlib.Path(tmpdir) / "models" / "best.pt"
            eval_path = pathlib.Path(tmpdir) / "models" / "eval_val_ddc.json"
            best_eval = pathlib.Path(tmpdir) / "models" / "eval_val_ddc_best.json"
            self.assertTrue(saved.is_file())
            self.assertTrue(best.is_file())
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["f_score_m"], 0.2)
            self.assertEqual(payload["weights"], "last")
            best_payload = json.loads(best_eval.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["weights"], "best")
            self.assertEqual(mock_eval.call_count, 2)
            log_root = pathlib.Path(tmpdir) / "callbacks" / "logs"
            run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.endswith("-smoke"))


if __name__ == "__main__":
    unittest.main()
