import json
import tempfile
import unittest

import keras
import numpy as np
import tensorflow as tf

from stepcovnet import constants
from stepcovnet.onset_events import config
from stepcovnet.onset_events import frontend
from stepcovnet.onset_events import models


def _small_model_config(**overrides) -> config.OnsetEventModelConfig:
    defaults = {
        "max_audio_seconds": 0.25,
        "num_queries": 16,
        "embed_dim": 32,
        "decoder_layers": 1,
        "base_filters": 16,
    }
    defaults.update(overrides)
    return config.OnsetEventModelConfig(**defaults)


class ModelsTest(unittest.TestCase):
    def test_onset_event_model_config_plan_defaults(self):
        model_cfg = config.OnsetEventModelConfig()
        self.assertEqual(model_cfg.frontend, "conv1d")
        self.assertEqual(model_cfg.num_queries, 1024)
        self.assertEqual(model_cfg.embed_dim, 256)
        self.assertEqual(model_cfg.decoder_layers, 2)
        self.assertEqual(model_cfg.target_sample_rate, constants.TARGET_SR)
        self.assertEqual(model_cfg.max_audio_seconds, 300.0)
        self.assertEqual(model_cfg.encoder.depth, 2)
        self.assertEqual(model_cfg.encoder.initial_filters, 16)
        self.assertTrue(model_cfg.include_duration_input)

    def test_experiment_config_json_roundtrip(self):
        experiment = config.OnsetEventExperimentConfig(
            dataset=config.OnsetEventDatasetConfig(),
            model=config.OnsetEventModelConfig(),
            run=config.OnsetEventRunConfig(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/onset_event.json"
            experiment.to_json(path)
            with open(path, encoding="utf-8") as config_file:
                raw = json.load(config_file)
            loaded = config.OnsetEventExperimentConfig.from_json(path)
        self.assertEqual(raw["dataset"]["target_sample_rate"], 44100)
        self.assertEqual(raw["model"]["num_queries"], 1024)
        self.assertEqual(raw["run"]["lambda_time"], 5.0)
        self.assertEqual(loaded.model.encoder.dilation_rates, [1, 2, 4, 8])

    def test_build_onset_event_model_returns_keras_model(self):
        model = models.build_onset_event_model(_small_model_config())
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(model.name, "onset_event_model")
        if isinstance(model.output_names, list):
            output_names = set(model.output_names)
        else:
            output_names = set(model.output_names.keys())
        self.assertIn("pred_times", output_names)
        self.assertIn("pred_confidence", output_names)

    def test_forward_with_duration_output_shapes(self):
        model_cfg = _small_model_config()
        model = models.build_onset_event_model(model_cfg)
        num_samples = frontend.max_waveform_samples(
            model_cfg.max_audio_seconds,
            model_cfg.target_sample_rate,
        )
        audio = np.zeros((1, num_samples), dtype=np.float32)
        duration = np.array([0.2], dtype=np.float32)
        outputs = model({"audio": audio, "duration": duration}, training=False)
        self.assertEqual(outputs["pred_times"].shape, (1, model_cfg.num_queries))
        pred_std = float(tf.math.reduce_std(outputs["pred_times"]).numpy())
        self.assertGreater(pred_std, 1e-4)
        self.assertEqual(outputs["pred_confidence"].shape, (1, model_cfg.num_queries))
        self.assertLess(float(tf.reduce_mean(outputs["pred_confidence"]).numpy()), 0.2)
        self.assertEqual(outputs["pred_times"].dtype, tf.float32)
        self.assertLessEqual(float(np.max(outputs["pred_times"].numpy())), 0.2 + 1e-5)
        self.assertGreaterEqual(float(np.min(outputs["pred_times"].numpy())), 0.0)

    def test_forward_without_duration_output_shapes(self):
        model_cfg = _small_model_config(include_duration_input=False)
        model = models.build_onset_event_model(model_cfg)
        num_samples = frontend.max_waveform_samples(
            model_cfg.max_audio_seconds,
            model_cfg.target_sample_rate,
        )
        audio = np.random.randn(1, num_samples).astype(np.float32)
        outputs = model(audio, training=False)
        times = outputs["pred_times"]
        confidence = outputs["pred_confidence"]
        self.assertEqual(times.shape, (1, model_cfg.num_queries))
        self.assertEqual(confidence.shape, (1, model_cfg.num_queries))
        self.assertLessEqual(float(np.max(times.numpy())), 1.0)
        self.assertGreaterEqual(float(np.min(times.numpy())), 0.0)

    def test_build_onset_event_model_with_query_ref(self):
        ref = tuple((i + 0.5) / 16.0 for i in range(16))
        model = models.build_onset_event_model(
            _small_model_config(),
            query_ref_normalized=ref,
        )
        layer = model.get_layer("pred_times_norm")
        self.assertEqual(layer._ref_normalized, ref)

    def test_build_onset_event_model_without_time_delta(self):
        model = models.build_onset_event_model(
            _small_model_config(),
            learn_time_delta=False,
        )
        layer = model.get_layer("pred_times_norm")
        self.assertFalse(layer._learn_time_delta)
        self.assertIsNone(layer._delta_dense)

    def test_build_onset_event_model_mel_frontend(self):
        model_cfg = _small_model_config(frontend="mel")
        model = models.build_onset_event_model(model_cfg)
        max_frames = frontend.target_encoder_frames(model_cfg.max_audio_seconds)
        features = np.zeros((1, max_frames, constants.N_MELS), dtype=np.float32)
        duration = np.array([0.2], dtype=np.float32)
        outputs = model({"features": features, "duration": duration}, training=False)
        self.assertEqual(outputs["pred_times"].shape, (1, model_cfg.num_queries))

    def test_unsupported_frontend_raises(self):
        with self.assertRaises(ValueError):
            models.build_onset_event_model(
                _small_model_config(frontend="stft"),
            )
