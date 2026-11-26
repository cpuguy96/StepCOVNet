"""Tests for the PyTorch AR onset offline evaluation CLI."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("STEPCOVNET_NO_WSL", "1")

REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_ar_onset_offline as eval_cli  # noqa: E402

from stepcovnet.onset_ar import config  # noqa: E402


class EvalArOnsetOfflineTest(unittest.TestCase):
    def test_resolve_model_path_prefers_explicit_then_best_pt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = config.ArExperimentConfig(
                dataset=config.ArDatasetConfig(),
                model=config.ArModelConfig(),
                run=config.ArRunConfig(model_output_dir=tmpdir),
            )
            explicit = eval_cli._resolve_model_path(exp, "/custom/model.pt")
            self.assertEqual(explicit, pathlib.Path("/custom/model.pt"))

            fallback = eval_cli._resolve_model_path(exp, "")
            self.assertEqual(fallback, pathlib.Path(tmpdir) / "ar_onset_model.pt")

            best_pt = pathlib.Path(tmpdir) / "best.pt"
            best_pt.write_bytes(b"pt")
            resolved = eval_cli._resolve_model_path(exp, "")
            self.assertEqual(resolved, best_pt)

    def test_main_returns_nonzero_when_checkpoint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = pathlib.Path(tmpdir) / "cfg.json"
            exp = config.ArExperimentConfig(
                dataset=config.ArDatasetConfig(),
                model=config.ArModelConfig(),
                run=config.ArRunConfig(model_output_dir=tmpdir),
            )
            cfg_path.write_text(
                json.dumps(exp.as_dict(), indent=2),
                encoding="utf-8",
            )
            rc = eval_cli.main(["--config", str(cfg_path), "--device", "cpu"])
            self.assertEqual(rc, 1)

    def test_main_runs_eval_and_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = pathlib.Path(tmpdir) / "cfg.json"
            ckpt_path = pathlib.Path(tmpdir) / "best.pt"
            out_path = pathlib.Path(tmpdir) / "report.json"
            ckpt_path.write_bytes(b"stub")
            exp = config.ArExperimentConfig(
                dataset=config.ArDatasetConfig(),
                model=config.ArModelConfig(),
                run=config.ArRunConfig(model_output_dir=tmpdir),
            )
            cfg_path.write_text(
                json.dumps(exp.as_dict(), indent=2),
                encoding="utf-8",
            )
            fake_report = mock.Mock()
            fake_report.as_dict.return_value = {
                "n_charts": 1,
                "micro_event_f1": 0.95,
                "micro_timing_match": 0.92,
            }
            with (
                mock.patch.object(
                    eval_cli.trainers, "load_ar_onset_model", return_value=mock.Mock()
                ),
                mock.patch.object(
                    eval_cli.datasets, "load_split_samples", return_value=[mock.Mock()]
                ),
                mock.patch.object(
                    eval_cli.inference, "evaluate_ar_samples", return_value=fake_report
                ) as mock_eval,
            ):
                rc = eval_cli.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--split",
                        "overfit",
                        "--ar_decode",
                        "--ar_decode_eos_logit_bias",
                        "-0.5",
                        "--ar_decode_min_onset_tokens",
                        "10",
                        "--device",
                        "cpu",
                        "--output",
                        str(out_path),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.is_file())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["micro_event_f1"], 0.95)
            self.assertEqual(payload["split"], "overfit")
            lc = mock_eval.call_args.kwargs["length_control"]
            self.assertAlmostEqual(lc.eos_logit_bias, -0.5)
            self.assertEqual(lc.min_onset_tokens, 10)


if __name__ == "__main__":
    unittest.main()
