import pathlib
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from stepcovnet import config, constants, dense_overfit_eval, trainers


def _write_tiny_dataset(root: pathlib.Path, stem: str = "song1") -> pathlib.Path:
    song_dir = root / stem
    song_dir.mkdir(parents=True, exist_ok=True)
    sr = constants.TARGET_SR
    duration = 0.8
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    wav = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(song_dir / f"{stem}.wav"), wav, sr)
    (song_dir / f"{stem}.txt").write_text(
        f"TITLE {stem}\n"
        "BPM 120.0\n"
        "NOTES\n"
        "DIFFICULTY Challenge\n"
        "1000 0.15\n"
        "0100 0.35\n"
        "2000 0.50\n"
        "3000 0.65\n",
        encoding="utf-8",
    )
    return root


class TrainersHelpersTest(unittest.TestCase):
    def test_experiment_naming(self):
        onset_name = trainers._get_onset_experiment_name(
            take_count=5,
            apply_temporal_augment=True,
            should_apply_spec_augment=True,
            use_gaussian_target=True,
            gaussian_sigma=1.5,
            model_params=config.OnsetModelConfig(
                onset_architecture=config.OnsetArchitecture.TCN,
                initial_filters=16,
                tcn_blocks=2,
            ),
            feature_source=config.FeatureSource.MERT,
        )
        self.assertIn("ONSET-mert-take_5-sigma_1_5", onset_name)
        self.assertIn("tcn", onset_name)

        arrow_cfg = config.ArrowExperimentConfig(
            dataset=config.ArrowDatasetConfig(data_dir="tr", val_data_dir="va"),
            model=config.ArrowModelConfig.from_dict(
                {"model_type": "lstm", "lstm": {"units": 32}}
            ),
            run=config.ArrowRunConfig(epoch=1, take_count=1, model_output_dir="out"),
        )
        arrow_name = trainers._get_arrow_experiment_name(arrow_cfg)
        self.assertTrue(arrow_name.startswith("ARROW-lstm-"))

    def test_monitored_checkpoint_discovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = pathlib.Path(tmpdir) / "models" / "run1"
            run_dir.mkdir(parents=True)
            for fname in (
                "VAL_ONSET_F1_SCORE-0.20000.pt",
                "VAL_ONSET_F1_SCORE-0.10000.pt",
                "VAL_ONSET_F1_SCORE-notanumber.pt",
                "other.pt",
            ):
                (run_dir / fname).write_bytes(b"pt")
            found = trainers._list_monitored_checkpoints(
                tmpdir, trainers.ONSET_CHECKPOINT_MONITOR
            )
            self.assertEqual(len(found), 2)
            best = trainers._latest_monitored_checkpoint(
                tmpdir, trainers.ONSET_CHECKPOINT_MONITOR
            )
            assert best is not None
            self.assertTrue(best.endswith("VAL_ONSET_F1_SCORE-0.20000.pt"))

    def test_cosine_warmup_schedule(self):
        sched = trainers._build_cosine_warmup_schedule(
            total_epochs=10, warmup_epochs=3, lr_peak=1e-3, lr_min=1e-5
        )
        self.assertAlmostEqual(sched(0), 1e-5, places=8)
        self.assertAlmostEqual(sched(2), 1e-3, places=8)
        self.assertLess(sched(9), 1e-3)

    def test_dense_val_event_f1_callback(self):
        y_true = np.zeros((1, 40, 1), dtype=np.float32)
        y_true[0, 10, 0] = 1.0
        y_pred = np.zeros((1, 40, 1), dtype=np.float32)
        y_pred[0, 10, 0] = 0.9

        class _ConstModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, _x: torch.Tensor) -> torch.Tensor:
                return torch.from_numpy(y_pred)

        val_batches = [(torch.zeros((1, 40, 8)), torch.from_numpy(y_true))]
        cb = dense_overfit_eval.DenseValEventF1Callback(
            val_batches,
            confidence_threshold=0.5,
            model=_ConstModel(),
        )
        logs: dict[str, float] = {}
        cb.on_epoch_end(0, logs)
        self.assertIn("val_dense_event_onset_f1", logs)
        self.assertAlmostEqual(logs["val_dense_event_onset_f1"], 1.0, places=5)


class TrainersEndToEndPyTorchTest(unittest.TestCase):
    def test_run_train_saves_and_reloads_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            data_dir = _write_tiny_dataset(root / "train", "s1")
            val_dir = _write_tiny_dataset(root / "val", "s2")
            out_dir = root / "models"
            cb_dir = root / "callbacks"

            model, history = trainers.run_train(
                data_dir=str(data_dir),
                val_data_dir=str(val_dir),
                batch_size=1,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=1.0,
                take_count=1,
                val_take_count=1,
                epoch=1,
                model_output_dir=str(out_dir),
                callback_root_dir=str(cb_dir),
                model_name="tiny_onset",
                model_params={
                    "initial_filters": 8,
                    "depth": 1,
                    "dilation_rates": [1],
                },
            )
            self.assertIsInstance(model, torch.nn.Module)
            self.assertIn("loss", history.history)
            self.assertIn("val_onset_f1_score", history.history)
            saved = out_dir / "tiny_onset.pt"
            self.assertTrue(saved.is_file())
            reloaded = trainers.load_onset_model(saved)
            self.assertIsInstance(reloaded, torch.nn.Module)

    def test_run_arrow_train_saves_and_reloads_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            data_dir = _write_tiny_dataset(root / "train", "s1")
            val_dir = _write_tiny_dataset(root / "val", "s2")
            out_dir = root / "models"
            cb_dir = root / "callbacks"

            exp = config.ArrowExperimentConfig(
                dataset=config.ArrowDatasetConfig(
                    data_dir=str(data_dir),
                    val_data_dir=str(val_dir),
                    batch_size=1,
                    use_interval=True,
                    use_aux_interval_target=True,
                ),
                model=config.ArrowModelConfig.from_dict(
                    {
                        "model_type": "transformer",
                        "transformer": {
                            "num_layers": 1,
                            "d_model": 16,
                            "num_heads": 2,
                            "ff_dim": 32,
                        },
                    }
                ),
                run=config.ArrowRunConfig(
                    epoch=2,
                    take_count=1,
                    val_take_count=1,
                    model_output_dir=str(out_dir),
                    callback_root_dir=str(cb_dir),
                    model_name="tiny_arrow",
                    aux_interval_weight=0.5,
                    warmup_epochs=1,
                    seed=42,
                ),
            )
            model, history = trainers.run_arrow_train_from_config(exp)
            self.assertIsInstance(model, torch.nn.Module)
            self.assertIn("val_output_probabilities_main_loss", history.history)
            saved = out_dir / "tiny_arrow.pt"
            self.assertTrue(saved.is_file())
            reloaded = trainers.load_arrow_model(saved)
            self.assertIsInstance(reloaded, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
