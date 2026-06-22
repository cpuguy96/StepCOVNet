import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.io.wavfile

from stepcovnet import constants
from stepcovnet.onset_events import audio, config, inference, models


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    int_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, int_samples)


def _mock_keras_input(name: str):
    tensor = mock.Mock()
    tensor.name = f"{name}:0"
    return tensor


class _FixedPredictModel:
    """Minimal stand-in for a Keras model with dict outputs."""

    def __init__(
        self,
        pred_times: np.ndarray,
        pred_confidence: np.ndarray,
        *,
        include_duration: bool = True,
    ) -> None:
        self._pred_times = np.asarray(pred_times, dtype=np.float32)
        self._pred_confidence = np.asarray(pred_confidence, dtype=np.float32)
        if include_duration:
            self.inputs = [
                _mock_keras_input("audio"),
                _mock_keras_input("duration"),
            ]
        else:
            self.inputs = [_mock_keras_input("audio")]
        self.output_names = ["pred_times", "pred_confidence"]

    def predict(self, model_input, verbose=0):
        _ = verbose
        if isinstance(model_input, dict):
            audio_batch = model_input["audio"]
            duration_batch = model_input["duration"]
            self.last_duration = float(duration_batch[0])
        else:
            audio_batch = model_input
            self.last_duration = None
        self.last_audio_batch = audio_batch
        return {
            "pred_times": self._pred_times[np.newaxis, :],
            "pred_confidence": self._pred_confidence[np.newaxis, :],
        }


class InferenceTest(unittest.TestCase):
    def test_predict_onsets_filters_sorts_and_applies_min_gap(self):
        pred_times = np.array(
            [0.10, 0.12, 0.50, 0.80, 0.83, 1.00],
            dtype=np.float32,
        )
        pred_confidence = np.array(
            [0.9, 0.95, 0.2, 0.85, 0.7, 0.6],
            dtype=np.float32,
        )
        model = _FixedPredictModel(pred_times, pred_confidence)
        waveform = np.zeros(4410, dtype=np.float32)

        times, confidences = inference.predict_onsets(
            model,
            waveform,
            confidence_threshold=0.5,
            min_onset_distance_ms=50.0,
            target_sample_rate=constants.TARGET_SR,
            max_audio_seconds=0.25,
        )

        np.testing.assert_array_equal(
            times, np.array([0.10, 0.80, 1.00], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            confidences,
            np.array([0.9, 0.85, 0.6], dtype=np.float32),
        )

    def test_predict_onsets_scales_normalized_times_without_duration_input(self):
        pred_times = np.array([0.25, 0.75], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.8], dtype=np.float32)
        model = _FixedPredictModel(
            pred_times,
            pred_confidence,
            include_duration=False,
        )
        waveform = np.ones(2205, dtype=np.float32)

        times, confidences = inference.predict_onsets(
            model,
            waveform,
            confidence_threshold=0.5,
            min_onset_distance_ms=0.0,
            target_sample_rate=constants.TARGET_SR,
            max_audio_seconds=0.25,
        )

        expected_duration = 2205 / constants.TARGET_SR
        np.testing.assert_allclose(times, pred_times * expected_duration, rtol=1e-5)
        np.testing.assert_array_equal(
            confidences, np.array([0.9, 0.8], dtype=np.float32)
        )

    def test_predict_onsets_from_wav_path(self):
        sr = constants.TARGET_SR
        duration_sec = 0.05
        n = int(duration_sec * sr)
        tone = 0.4 * np.sin(
            2 * np.pi * 440 * np.linspace(0, duration_sec, n, endpoint=False)
        ).astype(np.float32)
        pred_times = np.array([0.01, 0.02], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.85], dtype=np.float32)
        model = _FixedPredictModel(pred_times, pred_confidence)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "tone.wav"
            _write_wav(path, tone, sr)
            times, confidences = inference.predict_onsets(
                model,
                path,
                confidence_threshold=0.5,
                min_onset_distance_ms=0.0,
                max_audio_seconds=0.25,
            )

        self.assertEqual(times.dtype, np.float32)
        self.assertEqual(confidences.dtype, np.float32)
        np.testing.assert_array_equal(times, pred_times)
        np.testing.assert_array_equal(confidences, pred_confidence)
        expected_duration = n / sr
        self.assertAlmostEqual(model.last_duration, expected_duration, places=5)
        self.assertEqual(
            model.last_audio_batch.shape[1], audio.max_samples_for_cap(0.25, sr)
        )

    def test_predict_onsets_truncates_long_waveform(self):
        sr = constants.TARGET_SR
        max_audio_seconds = 0.1
        max_samples = audio.max_samples_for_cap(max_audio_seconds, sr)
        waveform = np.ones(max_samples + 500, dtype=np.float32)
        model = _FixedPredictModel(
            np.array([0.05], dtype=np.float32), np.array([0.9], dtype=np.float32)
        )

        inference.predict_onsets(
            model,
            waveform,
            max_audio_seconds=max_audio_seconds,
            min_onset_distance_ms=0.0,
        )

        expected_duration = max_samples / sr
        self.assertAlmostEqual(model.last_duration, expected_duration, places=5)

    def test_predict_onsets_returns_empty_when_nothing_passes_threshold(self):
        model = _FixedPredictModel(
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
        )
        waveform = np.zeros(100, dtype=np.float32)

        times, confidences = inference.predict_onsets(
            model,
            waveform,
            confidence_threshold=0.5,
            max_audio_seconds=0.25,
        )

        self.assertEqual(times.shape, (0,))
        self.assertEqual(confidences.shape, (0,))
        self.assertEqual(times.dtype, np.float32)
        self.assertEqual(confidences.dtype, np.float32)

    def test_predict_onsets_invalid_waveform_shape_raises(self):
        model = _FixedPredictModel(
            np.array([0.1], dtype=np.float32), np.array([0.9], dtype=np.float32)
        )
        with self.assertRaises(ValueError):
            inference.predict_onsets(
                model, np.zeros((2, 3), dtype=np.float32), max_audio_seconds=0.25
            )

    def test_predict_onsets_invalid_threshold_raises(self):
        model = _FixedPredictModel(
            np.array([0.1], dtype=np.float32), np.array([0.9], dtype=np.float32)
        )
        waveform = np.zeros(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            inference.predict_onsets(
                model, waveform, confidence_threshold=1.5, max_audio_seconds=0.25
            )
        with self.assertRaises(ValueError):
            inference.predict_onsets(
                model, waveform, min_onset_distance_ms=-1.0, max_audio_seconds=0.25
            )

    def test_predict_onsets_list_model_outputs(self):
        pred_times = np.array([0.2], dtype=np.float32)
        pred_confidence = np.array([0.8], dtype=np.float32)
        model = _FixedPredictModel(pred_times, pred_confidence)
        original_predict = model.predict

        def list_predict(model_input, verbose=0):
            outputs = original_predict(model_input, verbose=verbose)
            return [outputs["pred_times"], outputs["pred_confidence"]]

        model.predict = list_predict
        waveform = np.zeros(100, dtype=np.float32)
        times, confidences = inference.predict_onsets(
            model,
            waveform,
            min_onset_distance_ms=0.0,
            max_audio_seconds=0.25,
        )
        np.testing.assert_array_equal(times, pred_times)
        np.testing.assert_array_equal(confidences, pred_confidence)

    def test_predict_onsets_accepts_model_with_single_input_tensor(self):
        model = _FixedPredictModel(
            np.array([0.3], dtype=np.float32),
            np.array([0.9], dtype=np.float32),
            include_duration=False,
        )
        model.inputs = _mock_keras_input("audio")
        times, confidences = inference.predict_onsets(
            model,
            np.array([0.5, -0.5], dtype=np.float32),
            min_onset_distance_ms=0.0,
            max_audio_seconds=0.25,
        )
        self.assertEqual(times.shape, (1,))
        self.assertEqual(confidences.shape, (1,))

    def test_public_api_reexports_predict_onsets(self):
        from stepcovnet.onset_events import predict_onsets

        self.assertIs(predict_onsets, inference.predict_onsets)

    def test_predict_onsets_with_built_small_model(self):
        model_cfg = config.OnsetEventModelConfig(
            max_audio_seconds=0.25,
            num_queries=8,
            embed_dim=16,
            decoder_layers=1,
            base_filters=8,
        )
        model = models.build_onset_event_model(model_cfg)
        waveform = np.random.randn(1102).astype(np.float32)

        times, confidences = inference.predict_onsets(
            model,
            waveform,
            max_audio_seconds=0.25,
            min_onset_distance_ms=0.0,
            confidence_threshold=0.0,
        )

        self.assertEqual(times.ndim, 1)
        self.assertEqual(confidences.ndim, 1)
        self.assertEqual(times.shape, confidences.shape)
        self.assertEqual(times.dtype, np.float32)
        self.assertEqual(confidences.dtype, np.float32)
        if times.size > 1:
            self.assertTrue(np.all(np.diff(times) >= 0.0))
