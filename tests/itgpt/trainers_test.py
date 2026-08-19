"""Tests for ITGPT placement trainers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants, trainers


def _synthetic_chart(*, n_beats: int = 6) -> ddcl_datasets.DdclChart:
    """Return a short in-memory chart for trainer tests.

    Args:
        n_beats: Integer-beat length.

    Returns:
        DdclChart with BPM in stream column 1.
    """
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 1] = 140.0
    return ddcl_datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=5,
        beat_audio=np.zeros(
            (
                n_beats,
                constants.N_FRAMES_PER_BEAT,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        ),
        stream=stream,
        slots=slots,
        memlen=0,
    )


class ItgptTrainersTest(unittest.TestCase):
    def test_tensorboard_run_log_dir_is_timestamped_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = trainers.tensorboard_run_log_dir(tmpdir, "smoke")
            self.assertEqual(log_dir.parent, pathlib.Path(tmpdir) / "logs")
            self.assertTrue(log_dir.name.endswith("-smoke"))
            self.assertTrue(log_dir.is_dir())

    def test_train_placement_writes_artifacts(self):
        chart = _synthetic_chart()
        report = mock.Mock()
        report.as_dict.return_value = {"f1_at_05": 0.1, "f1_max": 0.2}
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
                model=config.ItgptModelConfig(
                    d_model=32,
                    n_heads=4,
                    n_enc_layers=1,
                    cnn_hidden=8,
                    dropout_rate=0.0,
                ),
                run=config.ItgptRunConfig(
                    epoch=1,
                    steps_per_epoch=1,
                    validation_steps=1,
                    model_output_dir=str(pathlib.Path(tmpdir) / "models"),
                    callback_root_dir=str(pathlib.Path(tmpdir) / "callbacks"),
                    model_name="smoke",
                    resume=False,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.itgpt.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ) as mock_eval,
            ):
                trainers.train_placement(experiment)
            saved = pathlib.Path(tmpdir) / "models" / "smoke.keras"
            best = pathlib.Path(tmpdir) / "models" / "best.keras"
            last = pathlib.Path(tmpdir) / "models" / "last.keras"
            eval_path = pathlib.Path(tmpdir) / "models" / "eval_val_slot48.json"
            best_eval = pathlib.Path(tmpdir) / "models" / "eval_val_slot48_best.json"
            self.assertTrue(saved.is_file())
            self.assertTrue(best.is_file())
            self.assertTrue(last.is_file())
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["f1_max"], 0.2)
            self.assertEqual(payload["weights"], "last")
            best_payload = json.loads(best_eval.read_text(encoding="utf-8"))
            self.assertEqual(best_payload["weights"], "best")
            self.assertEqual(mock_eval.call_count, 2)
            log_root = pathlib.Path(tmpdir) / "callbacks" / "logs"
            run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            self.assertTrue(run_dirs[0].name.endswith("-smoke"))

    def test_train_placement_rejects_empty_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.ItgptExperimentConfig(
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
            )
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    return_value=[],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment)
            with (
                mock.patch(
                    "stepcovnet.itgpt.trainers.datasets.load_split_charts",
                    side_effect=[[_synthetic_chart()], []],
                ),
                self.assertRaises(ValueError),
            ):
                trainers.train_placement(experiment)


if __name__ == "__main__":
    unittest.main()
