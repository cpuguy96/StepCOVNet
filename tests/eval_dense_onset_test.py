import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import torch

_SCRIPT_DIR = str(
    pathlib.Path(pathlib.Path(__file__).resolve()).resolve().parent.parent / "scripts"
)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_dense_onset  # noqa: E402


def _torch_model_stub():
    return mock.create_autospec(torch.nn.Module, instance=True)


class EvalDenseOnsetScriptTest(unittest.TestCase):
    def test_main_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = pathlib.Path(tmpdir) / "models"
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
            model_path = pathlib.Path(model_dir) / "best.pt"
            model_path.write_bytes(b"pt")

            config_path = pathlib.Path(tmpdir) / "config.json"
            with pathlib.Path(config_path).open("w", encoding="utf-8") as config_file:
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
                            "model_output_dir": str(model_dir),
                            "callback_root_dir": "callbacks/test",
                        },
                    },
                    config_file,
                )

            output_path = pathlib.Path(tmpdir) / "eval.json"
            fake_report = {
                "eval_split": "data/v2/val",
                "corruption": "none",
                "null_floors": {
                    "by_kind": {},
                    "strongest": {
                        "event_f1": "ioi_shuffle",
                        "timing_match": "ioi_shuffle",
                    },
                    "event_f1_floor": 0.25,
                    "timing_match_floor": 0.01,
                    "skill_event_f1": 0.4,
                    "skill_timing_match": 0.89,
                },
                "num_songs": 2,
                "mean_event_f1": 0.5,
                "micro_event_f1": 0.55,
                "micro_timing_match": 0.9,
                "timing_match_n_matched": 9.0,
                "timing_match_n_pred": 10.0,
                "timing_match_n_ref": 10.0,
                "timing_match_n_denom": 10.0,
                "micro_precision": 0.6,
                "micro_recall": 0.5,
                "micro_tp": 10.0,
                "micro_fp": 5.0,
                "micro_fn": 5.0,
                "eval_kwargs": {"confidence_threshold": 0.5},
                "per_song": {},
            }
            stub_model = _torch_model_stub()
            argv = [
                f"--config={config_path}",
                f"--model_path={model_path}",
                "--threshold=0.5",
                f"--output={output_path}",
            ]
            with (
                mock.patch.object(
                    eval_dense_onset.trainers,
                    "load_onset_model",
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
            with pathlib.Path(output_path).open(encoding="utf-8") as out_file:
                report = json.load(out_file)
            self.assertEqual(report["model_path"], str(model_path))
            self.assertEqual(report["config_path"], str(config_path))
            self.assertEqual(report["micro_event_f1"], 0.55)

    def test_find_saved_model_path_requires_single_pt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                eval_dense_onset._find_saved_model_path(tmpdir)

            (pathlib.Path(tmpdir) / "a.pt").write_bytes(b"a")
            (pathlib.Path(tmpdir) / "b.pt").write_bytes(b"b")
            with self.assertRaises(FileNotFoundError):
                eval_dense_onset._find_saved_model_path(tmpdir)

            (pathlib.Path(tmpdir) / "b.pt").unlink()
            resolved = eval_dense_onset._find_saved_model_path(tmpdir)
            self.assertTrue(resolved.endswith("a.pt"))
