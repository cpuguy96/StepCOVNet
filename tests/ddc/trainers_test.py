"""Tests for DDC placement trainers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet.ddc import config, datasets, models, trainers


def _synthetic_chart() -> datasets.PlacementChart:
    """Return a short in-memory chart for trainer tests.

    Returns:
        Placement chart with a labeled span long enough for nunroll=32.
    """
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
    def test_compile_and_train_on_batch(self):
        trainers.set_seed(0)
        model = trainers.compile_placement_model(
            models.build_clstm_placement_model(
                lstm_units=8,
                lstm_layers=1,
                dropout_rate=0.0,
                dnn_sizes=(8,),
                model_name="ddc_train_test",
            ),
            config.PlacementRunConfig(learning_rate=0.1, clipnorm=5.0),
        )
        rng = np.random.default_rng(0)
        inputs, labels = datasets.sample_train_batch(
            [_synthetic_chart()],
            batch_size=2,
            nunroll=32,
            rng=rng,
        )
        loss = model.train_on_batch(inputs, labels)
        self.assertTrue(np.isfinite(loss))

    def test_train_placement_writes_artifacts(self):
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
                    "stepcovnet.ddc.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.ddc.trainers.evaluation.evaluate_placement",
                    return_value=report,
                ),
            ):
                trainers.train_placement(experiment)
            saved = pathlib.Path(tmpdir) / "models" / "smoke.keras"
            eval_path = pathlib.Path(tmpdir) / "models" / "eval_val_ddc.json"
            self.assertTrue(saved.is_file())
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["f_score_m"], 0.2)


if __name__ == "__main__":
    unittest.main()
