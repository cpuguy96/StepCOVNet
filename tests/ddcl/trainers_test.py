"""Tests for DDCL placement trainers."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet.ddcl import config, constants, datasets, models, trainers


def _synthetic_chart(*, memlen: int = 2) -> datasets.DdclChart:
    """Return a short in-memory chart for trainer tests.

    Args:
        memlen: Context length used to size windows.

    Returns:
        DdclChart.
    """
    n_beats = 6
    context = memlen + 1
    n_frames = constants.N_FRAMES_PER_BEAT
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    slots[3, 24] = 1.0
    audio = np.zeros(
        (n_beats, context, n_frames, constants.N_MELS, constants.N_CHANNELS),
        dtype=np.float32,
    )
    stream = np.zeros((n_beats, context, constants.STREAM_DIM), dtype=np.float32)
    return datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=5,
        beat_audio=np.zeros(
            (n_beats, n_frames, constants.N_MELS, constants.N_CHANNELS),
            dtype=np.float32,
        ),
        stream=np.zeros((n_beats, constants.STREAM_DIM), dtype=np.float32),
        slots=slots,
        audio_fwd=audio,
        audio_bwd=audio,
        stream_fwd=stream,
        stream_bwd=stream,
    )


class DdclTrainersTest(unittest.TestCase):
    def test_compile_and_train_on_batch(self):
        trainers.set_seed(0)
        memlen = 2
        model = trainers.compile_placement_model(
            models.build_convlstm_placement_model(
                memlen=memlen,
                lstm_units=8,
                dropout_rate=0.0,
                dense_sizes=(16, 8),
                model_name="ddcl_train_test",
            ),
            config.DdclRunConfig(learning_rate=1e-4, clipnorm=1.0),
        )
        rng = np.random.default_rng(0)
        inputs, labels = datasets.sample_train_batch(
            [_synthetic_chart(memlen=memlen)],
            batch_size=2,
            rng=rng,
        )
        loss = model.train_on_batch(inputs, labels)
        self.assertTrue(np.isfinite(loss))

    def test_tensorboard_run_log_dir_is_timestamped_child(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = trainers.tensorboard_run_log_dir(tmpdir, "smoke")
            self.assertEqual(log_dir.parent, pathlib.Path(tmpdir) / "logs")
            self.assertTrue(log_dir.name.endswith("-smoke"))
            self.assertTrue(log_dir.is_dir())

    def test_train_placement_writes_artifacts(self):
        memlen = 2
        chart = _synthetic_chart(memlen=memlen)
        report = mock.Mock()
        report.as_dict.return_value = {"f1_at_05": 0.1, "f1_max": 0.2}
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = config.DdclExperimentConfig(
                dataset=config.DdclDatasetConfig(
                    training_index_path=str(pathlib.Path(tmpdir) / "missing.json"),
                    batch_size=2,
                    memlen=memlen,
                ),
                model=config.DdclModelConfig(
                    lstm_units=8,
                    dropout_rate=0.0,
                    dense_sizes=[16, 8],
                ),
                run=config.DdclRunConfig(
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
                    "stepcovnet.ddcl.trainers.datasets.load_split_charts",
                    return_value=[chart],
                ),
                mock.patch(
                    "stepcovnet.ddcl.trainers.evaluation.evaluate_slot48",
                    return_value=report,
                ) as mock_eval,
            ):
                trainers.train_placement(experiment)
            saved = pathlib.Path(tmpdir) / "models" / "smoke.keras"
            best = pathlib.Path(tmpdir) / "models" / "best.keras"
            eval_path = pathlib.Path(tmpdir) / "models" / "eval_val_slot48.json"
            best_eval = pathlib.Path(tmpdir) / "models" / "eval_val_slot48_best.json"
            self.assertTrue(saved.is_file())
            self.assertTrue(best.is_file())
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


if __name__ == "__main__":
    unittest.main()
