import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_dense_onset  # noqa: E402


class EvalDenseOnsetScriptTest(unittest.TestCase):
    def test_main_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "models")
            os.makedirs(model_dir)
            model_path = os.path.join(model_dir, "best.keras")
            with open(model_path, "wb") as model_file:
                model_file.write(b"keras")

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(
                    {
                        "dataset": {
                            "data_dir": "data/v2/train",
                            "val_data_dir": "data/v2/val",
                            "feature_source": "mert",
                            "mert_features_dir": "data/mert/v2",
                        },
                        "model": {"input_features": 1024},
                        "run": {
                            "epoch": 1,
                            "take_count": -1,
                            "model_output_dir": model_dir,
                            "callback_root_dir": "callbacks/test",
                        },
                    },
                    config_file,
                )

            output_path = os.path.join(tmpdir, "eval.json")
            fake_report = {
                "eval_split": "data/v2/val",
                "num_songs": 2,
                "mean_event_f1": 0.5,
                "micro_event_f1": 0.55,
                "micro_precision": 0.6,
                "micro_recall": 0.5,
                "micro_tp": 10.0,
                "micro_fp": 5.0,
                "micro_fn": 5.0,
                "eval_kwargs": {"confidence_threshold": 0.5},
                "per_song": {},
            }
            stub_model = mock.Mock()
            argv = [
                f"--config={config_path}",
                f"--model_path={model_path}",
                "--threshold=0.5",
                f"--output={output_path}",
            ]
            with (
                mock.patch.object(
                    eval_dense_onset.tf.keras.models,
                    "load_model",
                    return_value=stub_model,
                    autospec=True,
                ),
                mock.patch.object(
                    eval_dense_onset.dense_overfit_eval,
                    "eval_dense_val_event_f1",
                    return_value=fake_report,
                    autospec=True,
                ) as eval_mock,
            ):
                exit_code = eval_dense_onset.main(argv)

            self.assertEqual(exit_code, 0)
            eval_mock.assert_called_once()
            self.assertEqual(eval_mock.call_args.kwargs["confidence_threshold"], 0.5)
            with open(output_path, encoding="utf-8") as out_file:
                report = json.load(out_file)
            self.assertEqual(report["model_path"], model_path)
            self.assertEqual(report["config_path"], config_path)
            self.assertEqual(report["micro_event_f1"], 0.55)

    def test_find_saved_model_path_requires_single_keras(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                eval_dense_onset._find_saved_model_path(tmpdir)

            with open(os.path.join(tmpdir, "a.keras"), "wb") as model_file:
                model_file.write(b"a")
            with open(os.path.join(tmpdir, "b.keras"), "wb") as model_file:
                model_file.write(b"b")
            with self.assertRaises(FileNotFoundError):
                eval_dense_onset._find_saved_model_path(tmpdir)

            os.remove(os.path.join(tmpdir, "b.keras"))
            resolved = eval_dense_onset._find_saved_model_path(tmpdir)
            self.assertTrue(resolved.endswith("a.keras"))
