"""Tests for DDCL ``M-slot48`` evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import datasets, evaluation, slots


class _StubModel:
    """Keras-like model that returns a queued slot matrix."""

    def __init__(self, pred: np.ndarray) -> None:
        self._pred = np.asarray(pred, dtype=np.float32)

    def predict(self, inputs, verbose=0):
        """Return the queued predictions.

        Args:
            inputs: Unused.
            verbose: Unused Keras flag.

        Returns:
            Slot probabilities.
        """
        del inputs, verbose
        return self._pred


def _tiny_chart(slots: np.ndarray) -> datasets.DdclChart:
    """Return a chart with dummy audio windows matching ``slots``.

    Args:
        slots: Target ``(n_beats, 48)``.

    Returns:
        DdclChart.
    """
    n_beats = slots.shape[0]
    beat_audio = np.zeros((n_beats, 4, 4, 3), dtype=np.float32)
    return datasets.DdclChart(
        song_key="pack/song",
        difficulty="challenge",
        meter=9,
        beat_audio=beat_audio,
        stream=np.zeros((n_beats, 2), dtype=np.float32),
        slots=slots.astype(np.float32),
        memlen=1,
    )


class _AudioReadingModel:
    """Stub model whose slot prediction depends directly on ``audio``."""

    def predict(self, inputs, verbose=0):
        """Predict slot 0 from the current beat's mean forward audio energy.

        Args:
            inputs: Dict with ``audio_fwd`` window ``(batch, memlen+1, 4, 4, 3)``.
            verbose: Unused Keras flag.

        Returns:
            Slot probabilities ``(batch, 48)``.
        """
        del verbose
        audio = np.asarray(inputs["audio_fwd"], dtype=np.float32)
        current_energy = audio[:, -1].mean(axis=(1, 2, 3))
        out = np.zeros((audio.shape[0], 48), dtype=np.float32)
        out[:, 0] = np.clip(current_energy, 0.0, 1.0)
        return out


class DdclEvalTest(unittest.TestCase):
    def test_perfect_prediction_beats_null(self):
        target = np.zeros((8, 48), dtype=np.float32)
        target[0, 0] = 1.0
        target[3, 24] = 1.0
        target[7, 47] = 1.0
        chart = _tiny_chart(target)
        model = _StubModel(target)
        report = evaluation.evaluate_slot48(model, [chart], seed=0)
        self.assertAlmostEqual(report.f1_at_05, 1.0)
        self.assertAlmostEqual(report.f1_max, 1.0)
        self.assertLess(report.null_f1_at_05, 1.0)
        self.assertEqual(report.as_dict()["metric"], "M-slot48")

    def test_slot_centers_match_chart_times(self):
        segments = (prep_models.BpmSegment(start_beat=0.0, bpm=120.0),)
        times = np.array([0.0, 0.25, 0.5], dtype=np.float64)
        matrix = slots.times_to_slot_matrix(times, 0.0, segments)
        chart = _tiny_chart(matrix)
        chart = datasets.DdclChart(
            song_key=chart.song_key,
            difficulty=chart.difficulty,
            meter=chart.meter,
            beat_audio=chart.beat_audio,
            stream=chart.stream,
            slots=chart.slots,
            memlen=chart.memlen,
            offset_sec=0.0,
            bpm_segments=segments,
            gt_times_sec=times,
        )
        report = evaluation.timing_match_from_predictions([chart], [matrix])
        self.assertEqual(report.as_dict()["metric"], "timing_match")
        self.assertAlmostEqual(report.micro_timing_match, 1.0)
        self.assertEqual(report.n_matched, 3)
        with self.assertRaises(ValueError):
            evaluation.timing_match_from_predictions([_tiny_chart(matrix)], [matrix])

    def test_counts_at_threshold(self):
        pred = np.array([[0.9, 0.1, 0.0]])
        target = np.array([[1.0, 0.0, 1.0]])
        counts = evaluation.counts_at_threshold(pred, target, 0.5)
        self.assertEqual(counts.true_positives, 1)
        self.assertEqual(counts.false_positives, 0)
        self.assertEqual(counts.false_negatives, 1)

    def test_diagnose_timing_isolates_fp_and_fn_rank_drift(self):
        segments = (prep_models.BpmSegment(start_beat=0.0, bpm=120.0),)
        # Four quarter-note steps at 0.0, 0.5, 1.0, 1.5 s (beats 0..3, slot 0).
        times = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        matrix = slots.times_to_slot_matrix(times, 0.0, segments)
        base = _tiny_chart(matrix)
        chart = datasets.DdclChart(
            song_key=base.song_key,
            difficulty=base.difficulty,
            meter=base.meter,
            beat_audio=base.beat_audio,
            stream=base.stream,
            slots=base.slots,
            memlen=base.memlen,
            offset_sec=0.0,
            bpm_segments=segments,
            gt_times_sec=times,
        )

        # Prediction: early FP in beat 0 slot 24 (+0.25 s), missed beat 2 slot 0,
        # true positives at beats 0, 1, 3 slot 0 (with higher confidence on TPs).
        pred = np.zeros_like(matrix, dtype=np.float32)
        pred[0, 0] = 0.95
        pred[0, 24] = 0.60  # FP: shifts subsequent ranks forward by +1 until beat 2 FN
        pred[1, 0] = 0.90
        pred[2, 0] = 0.20  # FN at threshold 0.5
        pred[3, 0] = 0.92
        model = _StubModel(pred)

        diag = evaluation.diagnose_decoded_timing(
            model,
            [chart],
            threshold=0.5,
            thresholds=(0.1, 0.5, 0.8),
            seed=7,
        )
        payload = diag.as_dict()
        self.assertEqual(payload["metric"], "timing_match_diagnostic")
        self.assertEqual(diag.n_charts, 1)

        oracle = diag.conditions["oracle"]
        self.assertAlmostEqual(oracle.micro_timing_match, 1.0)
        self.assertAlmostEqual(oracle.micro_event_f1, 1.0)

        raw = diag.conditions["raw"]
        self.assertEqual(raw.slot_tp, 3)
        self.assertEqual(raw.slot_fp, 1)
        self.assertEqual(raw.slot_fn, 1)
        # Beat-1 TP is rank-shifted by beat-0 FP, so only 0.0 and 1.5 match in order.
        self.assertEqual(raw.timing_n_matched, 2)
        self.assertAlmostEqual(raw.micro_timing_match, 0.5)
        # Bipartite event F1 is order-invariant and keeps all 3 TPs (2*3/(4+4) = 0.75).
        self.assertAlmostEqual(raw.micro_event_f1, 0.75)

        fp_removed = diag.conditions["fp_removed"]
        self.assertEqual(fp_removed.slot_fp, 0)
        self.assertEqual(fp_removed.slot_fn, 1)
        # Stripping beat-0 FP fixes beat 1, but beat-2 FN still shifts beat 3.
        self.assertEqual(fp_removed.timing_n_matched, 2)

        fn_filled = diag.conditions["fn_filled"]
        self.assertEqual(fn_filled.slot_fp, 1)
        self.assertEqual(fn_filled.slot_fn, 0)
        # Filling the FN leaves beat-0 FP shifting beats 1..3 out of rank alignment.
        self.assertEqual(fn_filled.timing_n_matched, 1)

        beat_top_k = diag.conditions["beat_top_k"]
        # Top-1 per beat picks the top slot in beats 0..3 -> all 4 match in order.
        self.assertAlmostEqual(beat_top_k.micro_timing_match, 1.0)
        self.assertEqual(len(diag.threshold_sweep), 3)
        # At threshold 0.8, the 0.60 FP is suppressed (TPs 0.95, 0.90, 0.92 remain).
        self.assertEqual(diag.threshold_sweep[2].n_pred, 3)

        with self.assertRaises(ValueError):
            evaluation.diagnose_timing_from_predictions([], [])
        with self.assertRaises(ValueError):
            evaluation.diagnose_timing_from_predictions([chart], [])
        with self.assertRaises(ValueError):
            evaluation.diagnose_timing_from_predictions([base], [pred])

    def test_corrupt_chart_audio_preserves_stream_and_targets(self):
        target = np.zeros((6, 48), dtype=np.float32)
        target[[0, 2, 4], 0] = 1.0
        chart = _tiny_chart(target)
        audio = np.arange(6 * 4 * 4 * 3, dtype=np.float32).reshape(6, 4, 4, 3)
        stream = np.full((6, 2), 9.0, dtype=np.float32)
        chart = datasets.DdclChart(
            song_key=chart.song_key,
            difficulty=chart.difficulty,
            meter=chart.meter,
            beat_audio=audio,
            stream=stream,
            slots=chart.slots,
            memlen=chart.memlen,
        )

        same = evaluation.corrupt_chart_audio(chart, "none", seed=0)
        self.assertIs(same, chart)

        zeroed = evaluation.corrupt_chart_audio(chart, "zeros", seed=0)
        np.testing.assert_array_equal(zeroed.beat_audio, np.zeros_like(audio))
        np.testing.assert_array_equal(zeroed.stream, stream)
        np.testing.assert_array_equal(zeroed.slots, target)

        shuffled = evaluation.corrupt_chart_audio(chart, "shuffle", seed=123)
        self.assertFalse(np.array_equal(shuffled.beat_audio, audio))
        np.testing.assert_array_equal(shuffled.stream, stream)
        np.testing.assert_array_equal(shuffled.slots, target)

        with self.assertRaises(ValueError):
            evaluation.corrupt_chart_audio(chart, "reverse", seed=0)

    def test_evaluate_audio_grounding_detects_audio_dependence(self):
        segments = (prep_models.BpmSegment(start_beat=0.0, bpm=120.0),)
        times = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        matrix = slots.times_to_slot_matrix(times, 0.0, segments)
        n_beats = matrix.shape[0]
        audio = np.zeros((n_beats, 4, 4, 3), dtype=np.float32)
        for beat_idx in range(n_beats):
            if matrix[beat_idx, 0] > 0.5:
                audio[beat_idx] = 0.9
        chart = datasets.DdclChart(
            song_key="pack/song",
            difficulty="challenge",
            meter=9,
            beat_audio=audio,
            stream=np.full((n_beats, 2), 9.0, dtype=np.float32),
            slots=matrix,
            memlen=1,
            offset_sec=0.0,
            bpm_segments=segments,
            gt_times_sec=times,
        )

        report = evaluation.evaluate_audio_grounding(
            _AudioReadingModel(),
            [chart],
            threshold=0.5,
            thresholds=(0.3, 0.5, 0.7),
            seed=42,
        )
        payload = report.as_dict()
        self.assertEqual(payload["metric"], "audio_grounding_ablation")
        self.assertAlmostEqual(report.conditions["none"].slot_f1_at_05, 1.0)
        self.assertAlmostEqual(report.conditions["none"].mean_abs_prob_delta, 0.0)
        self.assertAlmostEqual(report.conditions["none"].slot_flip_rate, 0.0)
        self.assertAlmostEqual(report.conditions["zeros"].slot_f1_at_05, 0.0)
        self.assertAlmostEqual(report.slot_f1_drop_zeros, 1.0)
        self.assertGreater(report.slot_f1_drop_shuffle, 0.0)
        self.assertGreater(report.conditions["zeros"].occupied_jaccard_distance, 0.99)

        with self.assertRaises(ValueError):
            evaluation.grounding_report_from_predictions([], {})
        with self.assertRaises(ValueError):
            evaluation.grounding_report_from_predictions([chart], {"none": []})


if __name__ == "__main__":
    unittest.main()
