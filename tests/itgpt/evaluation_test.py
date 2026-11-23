"""Tests for ITGPT ``M-slot48`` evaluation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import constants, evaluation


class _ConstPredictor:
    """Return a constant slot probability tensor."""

    def __init__(self, value: float, *, squeeze_batch: bool = False) -> None:
        """Store the fill value.

        Args:
            value: Probability written to every slot.
            squeeze_batch: If True, return ``(beats, 48)``.
        """
        self.value = value
        self.squeeze_batch = squeeze_batch

    def predict(self, inputs: object, verbose: int = 0) -> np.ndarray:
        """Return filled slot probabilities.

        Args:
            inputs: Packed chart dict with ``audio``.
            verbose: Unused Keras verbosity.

        Returns:
            Constant probabilities.
        """
        del verbose
        n_beats = int(inputs["audio"].shape[1])  # type: ignore[index]
        pred = np.full((1, n_beats, constants.N_SLOTS), self.value, dtype=np.float32)
        if self.squeeze_batch:
            return pred[0]
        return pred


def _chart(*, n_beats: int = 8) -> ddcl_datasets.DdclChart:
    """Build a tiny labeled chart.

    Args:
        n_beats: Integer-beat length.

    Returns:
        Chart with one occupied 16th slot.
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


class _AudioSensitivePredictor:
    """Predict slot 0 occupied whenever beat audio mean exceeds 0.5."""

    def predict(self, inputs: object, verbose: int = 0) -> np.ndarray:
        """Return audio-conditioned slot probabilities.

        Args:
            inputs: Packed chart dict with ``audio``.
            verbose: Unused Keras verbosity.

        Returns:
            Probabilities ``(1, beats, 48)``.
        """
        del verbose
        audio = np.asarray(inputs["audio"], dtype=np.float32)  # type: ignore[index]
        beat_means = audio.mean(axis=tuple(range(2, audio.ndim)))
        pred = np.full(
            (audio.shape[0], audio.shape[1], constants.N_SLOTS), 0.05, dtype=np.float32
        )
        pred[:, :, 0] = np.where(beat_means > 0.5, 0.95, 0.05)
        return pred


def _mert_grounding_chart(*, n_beats: int = 8) -> ddcl_datasets.DdclChart:
    """Build a small MERT chart with audio correlated to beat-level onsets.

    Args:
        n_beats: Integer-beat length.

    Returns:
        Chart with BPM segments and ``gt_times_sec``.
    """
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    active_beats = [0, 2, 5, 7]
    for beat in active_beats:
        if beat < n_beats:
            slots[beat, 0] = 1.0
    beat_audio = np.zeros((n_beats, 24, 4), dtype=np.float32)
    for beat in active_beats:
        if beat < n_beats:
            beat_audio[beat] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 0] = 8.0
    stream[:, 1] = 120.0
    gt_times = np.asarray([beat * 0.5 for beat in active_beats if beat < n_beats])
    return ddcl_datasets.DdclChart(
        song_key="bundle/mert_grounding",
        difficulty="hard",
        meter=8,
        beat_audio=beat_audio,
        stream=stream,
        slots=slots,
        memlen=0,
        bpm_segments=(prep_models.BpmSegment(start_beat=0.0, bpm=120.0),),
        offset_sec=0.0,
        gt_times_sec=gt_times,
    )


class ItgptEvaluationTest(unittest.TestCase):
    def test_evaluate_slot48_rejects_empty(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate_slot48(_ConstPredictor(0.9), [])

    def test_predict_and_report_cite_table2(self):
        chart = _chart()
        pred = evaluation.predict_chart_slots(
            _ConstPredictor(0.9, squeeze_batch=True), chart
        )
        self.assertEqual(pred.shape, (chart.n_beats, constants.N_SLOTS))
        report = evaluation.evaluate_slot48(
            _ConstPredictor(0.9),
            [chart],
            max_beats=constants.CHUNK_ALIGN,
        )
        self.assertGreater(report.f1_at_05, 0.0)
        mixed = evaluation.evaluate_slot48(
            _ConstPredictor(0.2),
            [chart],
            max_beats=constants.CHUNK_ALIGN,
        )
        self.assertGreater(mixed.f1_max, mixed.f1_at_05)
        payload = evaluation.report_as_dict(report, weights="last")
        self.assertEqual(payload["citation"], "omalley2026itgpt")
        self.assertEqual(payload["weights"], "last")
        self.assertEqual(
            payload["published_f1_at_05_expanded_fraxtil"],
            constants.PUBLISHED_F1_AT_05_EXPANDED,
        )

    def test_evaluate_audio_grounding_detects_audio_drop(self):
        chart = _mert_grounding_chart()
        report = evaluation.evaluate_audio_grounding(
            _AudioSensitivePredictor(),
            [chart],
            max_beats=constants.CHUNK_ALIGN,
            seed=42,
        )
        self.assertAlmostEqual(report.conditions["none"].slot_f1_at_05, 1.0)
        self.assertGreater(report.slot_f1_drop_shuffle, 0.0)
        self.assertAlmostEqual(report.conditions["zeros"].slot_f1_at_05, 0.0)
        self.assertAlmostEqual(report.slot_f1_drop_zeros, 1.0)
        with self.assertRaises(ValueError):
            evaluation.evaluate_audio_grounding(_AudioSensitivePredictor(), [])
        with self.assertRaises(ValueError):
            evaluation.evaluate_audio_grounding(
                _AudioSensitivePredictor(),
                [_mert_grounding_chart(n_beats=70)],
                max_beats=constants.CHUNK_ALIGN,
            )


if __name__ == "__main__":
    unittest.main()
