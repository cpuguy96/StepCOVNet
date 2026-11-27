from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet import wsl_gpu
from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants, diagnostics
from stepcovnet.pytorch.itgpt import eval as pt_eval

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

with mock.patch.object(wsl_gpu, "bootstrap_gpu_script", autospec=True):
    import eval_itgpt_placement  # noqa: E402


def _make_chart(
    *,
    song_key: str = "pack_a/song_1",
    difficulty: str = "hard",
    meter: int = 9,
    n_beats: int = 12,
    active_slots: tuple[tuple[int, int], ...] = ((0, 0), (1, 24), (2, 12), (3, 8)),
) -> ddcl_datasets.DdclChart:
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    for beat, slot in active_slots:
        if beat < n_beats:
            slots[beat, slot] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 0] = float(meter)
    stream[:, 1] = 120.0
    gt_times = np.asarray(
        [(beat + slot / constants.N_SLOTS) * 0.5 for beat, slot in active_slots],
        dtype=np.float64,
    )
    return ddcl_datasets.DdclChart(
        song_key=song_key,
        difficulty=difficulty,
        meter=meter,
        beat_audio=np.zeros((n_beats, 24, 4), dtype=np.float32),
        stream=stream,
        slots=slots,
        memlen=0,
        bpm_segments=(prep_models.BpmSegment(start_beat=0.0, bpm=120.0),),
        offset_sec=0.0,
        gt_times_sec=gt_times,
    )


class _ScriptedPredictor:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = outputs
        self.calls = 0

    def predict(self, inputs: object, verbose: int = 0) -> np.ndarray:
        del verbose
        n_beats = int(inputs["audio"].shape[1])  # type: ignore[index]
        arr = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        out = np.zeros((1, n_beats, constants.N_SLOTS), dtype=np.float32)
        out[0, : arr.shape[0]] = arr
        return out


class ItgptDiagnosticsTest(unittest.TestCase):
    def test_error_breakdown_rejects_empty_or_mismatched(self) -> None:
        chart = _make_chart()
        with self.assertRaises(ValueError):
            diagnostics.error_breakdown_from_predictions([], [])
        with self.assertRaises(ValueError):
            diagnostics.error_breakdown_from_predictions([chart], [])
        with self.assertRaises(ValueError):
            diagnostics.evaluate_error_breakdown(_ScriptedPredictor([]), [])

    def test_error_breakdown_metrics_and_offsets_cover_all_buckets(self) -> None:
        chart_a = _make_chart(
            song_key="pack_a/song_1",
            meter=8,
            n_beats=20,
            active_slots=(
                (0, 0),
                (1, 24),
                (2, 12),
                (3, 8),
                (4, 6),
                (5, 1),
                (18, 0),
            ),
        )
        chart_b = _make_chart(
            song_key="pack_b/song_2",
            difficulty="challenge",
            meter=13,
            n_beats=10,
            active_slots=((0, 0), (5, 24)),
        )
        pred_a = np.full((20, constants.N_SLOTS), 0.05, dtype=np.float32)
        pred_a[0, 0] = 0.96  # TP quarter
        pred_a[1, 24] = 0.85  # TP 8th
        pred_a[2, 13] = 0.72  # FP 1 slot from GT (2,12)
        pred_a[3, 10] = 0.62  # FP 2 slots from GT (3,8)
        pred_a[4, 11] = 0.55  # FP 5 slots from GT (4,6)
        pred_a[5, 10] = 0.52  # FP 9 slots from GT (5,1)
        pred_a[0, 30] = 0.58  # FP >12 slots, same beat as GT (0,0)
        pred_a[10, 0] = 0.75  # FP on empty GT beat (other_beat)
        # Missing GT (18,0) -> FN on empty pred beat (other_beat)
        # Also include mid-probability false slots for calibration coverage
        pred_a[6, 0] = 0.15
        pred_a[7, 0] = 0.30
        pred_a[8, 0] = 0.45

        pred_b = np.full((10, constants.N_SLOTS), 0.02, dtype=np.float32)
        pred_b[0, 0] = 0.98
        pred_b[5, 24] = 0.92

        report = diagnostics.error_breakdown_from_predictions(
            [chart_a, chart_b],
            [pred_a, pred_b],
            threshold=0.5,
            exact_meter=False,
        )
        self.assertEqual(report.n_charts, 2)
        self.assertEqual(report.n_beats, 30)
        self.assertEqual(report.overall.tp, 4)
        self.assertEqual(report.overall.fp, 6)
        self.assertEqual(report.overall.fn, 5)
        self.assertEqual(report.total_errors, 11)

        fp_dist = report.fp_timing_offsets
        self.assertEqual(fp_dist.total, 6)
        self.assertEqual(fp_dist.le_1_slot, 1)
        self.assertEqual(fp_dist.slots_2_3, 1)
        self.assertEqual(fp_dist.slots_4_6, 1)
        self.assertEqual(fp_dist.slots_7_12, 1)
        self.assertEqual(fp_dist.other_same_beat, 1)
        self.assertEqual(fp_dist.other_beat, 1)

        fn_dist = report.fn_timing_offsets
        self.assertEqual(fn_dist.total, 5)
        self.assertEqual(fn_dist.other_beat, 1)

        payload = report.as_dict()
        self.assertEqual(payload["metric"], "itgpt_error_breakdown")
        self.assertIn("16th_off", payload["subdivision_f1"])  # type: ignore[operator]
        self.assertIn("intro", payload["section_f1"])  # type: ignore[operator]
        self.assertEqual([m.name for m in report.by_meter], ["01_<=8", "04_13+"])
        self.assertEqual(report.worst_charts[0].song_key, "pack_a/song_1")
        self.assertEqual(report.best_charts[0].song_key, "pack_b/song_2")

        text = report.format_text()
        self.assertIn("OVERALL (tau=0.50", text)
        self.assertIn("16th_off", text)
        self.assertIn("WORST 2 CHARTS", text)
        self.assertIn("BEST 2 CHARTS", text)

    def test_empty_opposite_events_and_meter_buckets(self) -> None:
        chart_empty_gt = _make_chart(
            song_key="pack/empty_gt",
            meter=9,
            n_beats=4,
            active_slots=(),
        )
        chart_empty_pred = _make_chart(
            song_key="pack/empty_pred",
            meter=11,
            n_beats=4,
            active_slots=((0, 0),),
        )
        pred_only_fp = np.zeros((4, constants.N_SLOTS), dtype=np.float32)
        pred_only_fp[0, 0] = 0.9
        pred_empty = np.zeros((4, constants.N_SLOTS), dtype=np.float32)

        report = diagnostics.error_breakdown_from_predictions(
            [chart_empty_gt, chart_empty_pred],
            [pred_only_fp, pred_empty],
            exact_meter=False,
        )
        self.assertEqual(report.fp_timing_offsets.other_beat, 1)
        self.assertEqual(report.fn_timing_offsets.other_beat, 1)
        self.assertEqual([m.name for m in report.by_meter], ["02_9-10", "03_11-12"])

        exact_report = diagnostics.evaluate_error_breakdown(
            _ScriptedPredictor([pred_only_fp]),
            [chart_empty_gt],
            max_beats=constants.CHUNK_ALIGN,
            exact_meter=True,
        )
        self.assertEqual([m.name for m in exact_report.by_meter], ["meter_09"])

    def test_eval_cli_grounding_and_breakdown_reuse_predictions(self) -> None:
        chart = _make_chart(n_beats=8, active_slots=((0, 0), (4, 24)))
        pred = np.full((8, constants.N_SLOTS), 0.05, dtype=np.float32)
        pred[0, 0] = 0.95
        pred[4, 24] = 0.90

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg_path = root / "cfg.json"
            out_dir = root / "run_out"
            model_path = out_dir / "best.pt"
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"stub")

            exp = config.ItgptExperimentConfig(
                run=config.ItgptRunConfig(
                    model_name="test_run",
                    model_output_dir=str(out_dir),
                    seed=7,
                ),
                dataset=config.ItgptDatasetConfig(
                    training_index_path=str(root / "training_index.json"),
                    max_beats=constants.CHUNK_ALIGN,
                ),
            )
            cfg_path.write_text(json.dumps(exp.as_dict()), encoding="utf-8")

            predictor = _ScriptedPredictor([pred])
            grounding_out = out_dir / "eval_val_grounding_best.json"
            with (
                mock.patch.object(
                    eval_itgpt_placement.wsl_gpu, "guard_pytorch_gpu_job", autospec=True
                ),
                mock.patch.object(
                    eval_itgpt_placement.trainers,
                    "load_placement_model",
                    autospec=True,
                    return_value=predictor,
                ),
                mock.patch.object(
                    eval_itgpt_placement.datasets,
                    "load_split_charts",
                    autospec=True,
                    return_value=[chart],
                ),
            ):
                rc = eval_itgpt_placement.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--model_path",
                        str(model_path),
                        "--output",
                        str(grounding_out),
                        "--diagnose_grounding",
                        "--error_breakdown",
                    ]
                )
            self.assertEqual(rc, 0)
            # 1 none + 1 shuffle + 1 zeros = 3 calls (NOT 4!)
            self.assertEqual(predictor.calls, 3)
            self.assertTrue(grounding_out.is_file())
            breakdown_out = out_dir / "eval_val_breakdown.json"
            self.assertTrue(breakdown_out.is_file())
            b_data = json.loads(breakdown_out.read_text(encoding="utf-8"))
            self.assertEqual(b_data["metric"], "itgpt_error_breakdown")
            self.assertEqual(b_data["total_errors"], 0)

            # Also verify standalone --error_breakdown on stepcovnet.pytorch.itgpt.eval
            pt_predictor = _ScriptedPredictor([pred])
            custom_bd = out_dir / "custom_bd.json"
            with (
                mock.patch.object(
                    pt_eval.trainers,
                    "load_placement_model",
                    autospec=True,
                    return_value=pt_predictor,
                ),
                mock.patch.object(
                    pt_eval.datasets,
                    "load_split_charts",
                    autospec=True,
                    return_value=[chart],
                ),
            ):
                rc2 = pt_eval.main(
                    [
                        "--config",
                        str(cfg_path),
                        "--model_path",
                        str(model_path),
                        "--error_breakdown",
                        "--breakdown_output",
                        str(custom_bd),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(rc2, 0)
            self.assertEqual(pt_predictor.calls, 1)
            self.assertTrue(custom_bd.is_file())


if __name__ == "__main__":
    unittest.main()
