import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet import config, constants, datasets, ssl_features


class SslFeaturesTest(unittest.TestCase):
    def test_mert_npy_path_beside_audio(self):
        path = ssl_features.mert_npy_path("/music/song.mp3")
        self.assertEqual(
            pathlib.Path(path),
            pathlib.Path("/music/song.mert.npy"),
        )

    def test_mert_npy_path_with_features_dir_preserves_relative_layout(self):
        path = ssl_features.mert_npy_path(
            "/data/train/sub/song.mp3",
            features_dir="/cache/mert",
            data_root="/data/train",
        )
        self.assertEqual(path, os.path.join("/cache/mert", "sub", "song.mert.npy"))

    def test_resample_features_to_hop_grid_upsamples(self):
        features = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        out = ssl_features.resample_features_to_hop_grid(
            features, audio_duration_sec=0.3
        )
        self.assertEqual(out.shape, (30, 2))
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        self.assertGreater(float(out[-1, 0]), float(out[0, 0]))

    def test_resample_features_to_hop_grid_single_frame(self):
        features = np.array([[1.0, 2.0]], dtype=np.float32)
        out = ssl_features.resample_features_to_hop_grid(
            features, audio_duration_sec=1.0
        )
        self.assertEqual(out.shape, (100, 2))
        np.testing.assert_allclose(out[:, 0], 1.0)
        np.testing.assert_allclose(out[:, 1], 2.0)

    def test_resample_features_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            ssl_features.resample_features_to_hop_grid(
                np.zeros((3, 4, 5), dtype=np.float32),
                audio_duration_sec=1.0,
            )

    def test_save_and_load_mert_features_roundtrip(self):
        features = np.random.randn(50, constants.MERT_HIDDEN_SIZE).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "track.wav")
            out_path = ssl_features.mert_npy_path(audio_path)
            ssl_features.save_mert_features(features, out_path)
            loaded = ssl_features.load_mert_features(audio_path)
            np.testing.assert_allclose(loaded, features)

    def test_load_mert_features_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ssl_features.load_mert_features("/no/such/song.mp3")

    def test_save_mert_features_invalid_shape_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                ssl_features.save_mert_features(
                    np.zeros((2, 3, 4), dtype=np.float32),
                    os.path.join(tmpdir, "bad.mert.npy"),
                )

    def test_load_mert_features_invalid_shape_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "track.wav")
            bad = np.zeros((2, 3, 4), dtype=np.float32)
            np.save(ssl_features.mert_npy_path(audio_path), bad)
            with self.assertRaises(ValueError):
                ssl_features.load_mert_features(audio_path)

    def test_extract_mert_features_from_audio_with_mocks(self):
        fake_hidden = np.random.randn(10, constants.MERT_HIDDEN_SIZE).astype(np.float32)
        fake_waveform = np.random.randn(24000).astype(np.float32)
        mock_model = mock.Mock()
        mock_processor = mock.Mock()
        with (
            mock.patch("librosa.load", return_value=(fake_waveform, 24000)),
            mock.patch.object(
                ssl_features,
                "_load_mert_model",
                return_value=(mock_model, mock_processor),
            ),
            mock.patch.object(
                ssl_features,
                "_mert_hidden_states_for_chunk",
                return_value=fake_hidden,
            ),
            mock.patch(
                "stepcovnet.datasets.onset_frame_count",
                return_value=100,
            ),
        ):
            out = ssl_features.extract_mert_features_from_audio("/fake/song.mp3")
        self.assertEqual(out.shape[1], constants.MERT_HIDDEN_SIZE)
        self.assertEqual(out.shape[0], 100)

    def test_extract_and_save_mert_features_with_mocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "song.mert.npy")
            with mock.patch.object(
                ssl_features,
                "extract_mert_features_from_audio",
                return_value=np.ones((5, constants.MERT_HIDDEN_SIZE), dtype=np.float32),
            ):
                written = ssl_features.extract_and_save_mert_features(
                    "/fake/song.mp3",
                    out_path,
                )
            self.assertEqual(written, out_path)
            loaded = np.load(out_path)
            self.assertEqual(loaded.shape, (5, constants.MERT_HIDDEN_SIZE))

    def test_resample_features_to_hop_grid_same_length(self):
        features = np.random.randn(100, 4).astype(np.float32)
        out = ssl_features.resample_features_to_hop_grid(
            features, audio_duration_sec=1.0
        )
        np.testing.assert_allclose(out, features)

    def test_resample_features_to_frame_count_upsamples(self):
        features = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        out = ssl_features.resample_features_to_frame_count(features, 4)
        self.assertEqual(out.shape, (4, 2))
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        self.assertGreater(float(out[-1, 0]), float(out[0, 0]))

    def test_resample_features_to_frame_count_same_length(self):
        features = np.random.randn(50, 8).astype(np.float32)
        out = ssl_features.resample_features_to_frame_count(features, 50)
        np.testing.assert_allclose(out, features)

    def test_resample_features_to_frame_count_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            ssl_features.resample_features_to_frame_count(
                np.zeros((2, 3, 4), dtype=np.float32),
                10,
            )

    def test_extract_mert_features_concatenates_multiple_chunks(self):
        fake_waveform = np.random.randn(48000).astype(np.float32)
        chunk_a = np.ones((5, constants.MERT_HIDDEN_SIZE), dtype=np.float32)
        chunk_b = np.ones((5, constants.MERT_HIDDEN_SIZE), dtype=np.float32) * 2.0
        with (
            mock.patch("librosa.load", return_value=(fake_waveform, 24000)),
            mock.patch.object(
                ssl_features,
                "_load_mert_model",
                return_value=(mock.Mock(), mock.Mock()),
            ),
            mock.patch.object(
                ssl_features,
                "_mert_hidden_states_for_chunk",
                side_effect=[chunk_a, chunk_b],
            ),
            mock.patch(
                "stepcovnet.datasets.onset_frame_count",
                return_value=200,
            ),
        ):
            out = ssl_features.extract_mert_features_from_audio("/fake/song.mp3")
        self.assertEqual(out.shape[0], 200)
        self.assertEqual(out.shape[1], constants.MERT_HIDDEN_SIZE)

    def test_require_ssl_deps_import_error(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                ssl_features._require_ssl_deps()

    def test_extract_mert_features_from_audio_empty_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with mock.patch("librosa.load", return_value=(np.array([]), 24000)):
                with self.assertRaises(ValueError):
                    ssl_features.extract_mert_features_from_audio(tmp.name)

    def test_mert_hidden_states_for_empty_chunk(self):
        out = ssl_features._mert_hidden_states_for_chunk(
            np.array([], dtype=np.float32),
            model=mock.Mock(),
            processor=mock.Mock(),
            layer=0,
            device="cpu",
        )
        self.assertEqual(out.shape, (0, constants.MERT_HIDDEN_SIZE))


class MertDatasetIntegrationTest(unittest.TestCase):
    def test_create_dataset_with_precomputed_mert_features(self):
        test_data_dir = os.path.join(os.path.dirname(__file__), "testdata")
        audio_path = None
        for root, _, files in os.walk(test_data_dir):
            for name in files:
                if name.endswith((".mp3", ".ogg", ".wav")):
                    audio_path = os.path.join(root, name)
                    break
            if audio_path:
                break
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        mel_features = datasets.load_onset_features(
            audio_path,
            config.FeatureSource.MEL,
        )
        n_steps = mel_features.shape[0]
        mert_array = np.random.randn(n_steps, constants.MERT_HIDDEN_SIZE).astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = ssl_features.mert_npy_path(
                audio_path,
                features_dir=tmpdir,
                data_root=test_data_dir,
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, mert_array)
            ds = datasets.create_dataset(
                test_data_dir,
                feature_source=config.FeatureSource.MERT,
                mert_features_dir=tmpdir,
                n_features=constants.MERT_HIDDEN_SIZE,
            )
            features, targets = next(iter(ds.take(1)))  # type: ignore[misc]
            self.assertEqual(features.shape[0], 1)
            self.assertEqual(features.shape[2], constants.MERT_HIDDEN_SIZE)
            self.assertEqual(features.shape[1], targets.shape[1])


class ResolveOnsetInputFeaturesTest(unittest.TestCase):
    def test_defaults_mel_to_n_mels(self):
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="train",
            val_data_dir="val",
        )
        model_cfg = config.OnsetModelConfig()
        self.assertEqual(
            config.resolve_onset_input_features(dataset_cfg, model_cfg),
            constants.N_MELS,
        )

    def test_defaults_mert_to_mert_hidden_size(self):
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="train",
            val_data_dir="val",
            feature_source=config.FeatureSource.MERT,
        )
        model_cfg = config.OnsetModelConfig()
        self.assertEqual(
            config.resolve_onset_input_features(dataset_cfg, model_cfg),
            constants.MERT_HIDDEN_SIZE,
        )

    def test_explicit_input_features_override(self):
        dataset_cfg = config.OnsetDatasetConfig(
            data_dir="train",
            val_data_dir="val",
            feature_source=config.FeatureSource.MERT,
        )
        model_cfg = config.OnsetModelConfig(input_features=512)
        self.assertEqual(
            config.resolve_onset_input_features(dataset_cfg, model_cfg),
            512,
        )


class ListAudioChartPairsTest(unittest.TestCase):
    def test_list_audio_chart_pairs_finds_files(self):
        test_data_dir = os.path.join(os.path.dirname(__file__), "testdata")
        pairs = datasets.list_audio_chart_pairs(test_data_dir)
        self.assertGreater(len(pairs), 0)
        for audio_path, chart_path in pairs:
            self.assertTrue(os.path.isfile(audio_path))
            self.assertTrue(os.path.isfile(chart_path))
            self.assertEqual(
                pathlib.Path(audio_path).stem,
                pathlib.Path(chart_path).stem.split(".")[0],
            )
