import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from stepcovnet import constants, datasets

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _first_batch(ds):
    return next(iter(ds.take(1)))  # type: ignore


def _get_one_audio_chart_pair(data_dir):
    """Return (audio_path, chart_path) for one pair in data_dir. Uses no private API."""
    for root, _, files in os.walk(data_dir):
        audio_ext = (".mp3", ".ogg", ".wav")
        chart_ext = ".txt"
        audio_files = [f for f in files if f.endswith(audio_ext)]
        chart_files = [f for f in files if f.endswith(chart_ext)]
        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching = [f for f in chart_files if f.startswith(stem)]
            if matching:
                return (
                    os.path.join(root, audio_file),
                    os.path.join(root, matching[0]),
                )
    return None, None


class DatasetsTest(unittest.TestCase):
    def test_create_dataset(self):
        ds = datasets.create_dataset(TEST_DATA_DIR)
        features, targets = _first_batch(ds)

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 128)  # Mel bins
        self.assertEqual(features.shape[1], 11726)  # Time steps

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[2], 1)  # Channels
        self.assertEqual(targets.shape[1], 11726)  # Time steps

        self.assertEqual(int(np.sum(targets[0])), 384)

    def test_create_dataset_with_empty_directory_raises_error(self):
        for create_fn in (datasets.create_dataset, datasets.create_arrow_dataset):
            with self.subTest(create_fn=create_fn.__name__):
                with self.assertRaises(ValueError):
                    create_fn("")

    def test_create_arrow_dataset(self):
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR)
        features, targets = _first_batch(ds)

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[1], 384)  # Timings

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[1], 384)  # Arrows

        self.assertTrue(np.all(targets[0] > 0))

    def _assert_snippet_batch_structure(
        self, features, targets, min_batch_size=1, max_batch_size=1
    ):
        self.assertIn("timing_input", features)
        self.assertIn("snippet_input", features)
        times = features["timing_input"]
        snippets = features["snippet_input"]
        batch_dim = times.shape[0]
        self.assertGreaterEqual(batch_dim, min_batch_size)
        self.assertLessEqual(batch_dim, max_batch_size)
        self.assertEqual(snippets.shape[0], batch_dim)
        self.assertEqual(targets.shape[0], batch_dim)
        self.assertEqual(times.shape[2], 1)
        self.assertEqual(snippets.shape[2], 11)
        self.assertEqual(snippets.shape[3], 128)
        self.assertEqual(times.shape[1], snippets.shape[1])
        self.assertEqual(times.shape[1], targets.shape[1])

    def test_create_arrow_dataset_batch_size_two_timing_only(self):
        """Multi-sample batching without snippets uses (times [None,1], cols [None])."""
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR, batch_size=2)
        times, cols = _first_batch(ds)
        self.assertGreaterEqual(times.shape[0], 1)
        self.assertLessEqual(times.shape[0], 2)
        self.assertEqual(times.shape[0], cols.shape[0])
        self.assertEqual(times.shape[2], 1)
        self.assertEqual(times.shape[1], cols.shape[1])

    def test_create_arrow_dataset_with_audio_snippets(self):
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            snippet_half_frames=5,
        )
        features, targets = _first_batch(ds)
        self._assert_snippet_batch_structure(features, targets)
        times = features["timing_input"]
        seq_len = times.shape[1]
        self.assertGreater(int(seq_len), 0)
        self.assertLessEqual(seq_len, constants.MAX_STEPS)
        self.assertGreater(int(np.sum(targets[0].numpy() > 0)), 0)

    def test_create_arrow_dataset_with_audio_snippets_batch_size_two(self):
        """Multi-sample batching with snippets uses correct padded_shapes (dict + cols)."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            batch_size=2,
            snippet_half_frames=5,
        )
        features, targets = _first_batch(ds)
        self._assert_snippet_batch_structure(
            features, targets, min_batch_size=1, max_batch_size=2
        )

    def test_extract_arrow_snippets_returns_correct_shapes(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        times_norm, snippets, cols = datasets.extract_arrow_snippets(
            audio_path, chart_path, half_frames=5
        )
        self.assertEqual(times_norm.ndim, 1)
        self.assertEqual(snippets.shape[0], len(times_norm))
        self.assertEqual(snippets.shape[1], 11)
        self.assertEqual(snippets.shape[2], 128)
        self.assertEqual(cols.shape[0], len(times_norm))

    def test_normalize_onset_spectrogram(self):
        spec = np.random.randn(100, constants.N_MELS).astype(np.float64)
        out = datasets.normalize_onset_spectrogram(spec)
        self.assertEqual(out.shape, spec.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.mean(out, axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(np.std(out, axis=0), 1.0, atol=1e-5)

    def test_audio_to_spectrogram(self):
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        spec = datasets.audio_to_spectrogram(audio_path)
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[0], constants.N_MELS)
        self.assertGreater(int(spec.shape[1]), 0)

    def test_extract_snippets_from_spec(self):
        n_times = 5
        half_frames = 2
        n_frames = 2 * half_frames + 1
        time_steps = 500
        spec = np.random.randn(time_steps, constants.N_MELS).astype(np.float32)
        times_seconds = np.linspace(0.1, 2.0, n_times)
        snippets = datasets.extract_snippets_from_spec(
            spec, times_seconds, half_frames=half_frames
        )
        self.assertEqual(snippets.shape, (n_times, n_frames, constants.N_MELS))
        self.assertEqual(snippets.dtype, np.float32)

    def test_extract_arrow_snippets_empty_chart(self):
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = os.path.join(tmpdir, "empty_chart.txt")
            with open(chart_path, "w") as f:
                f.write("TITLE Empty\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
            times_norm, snippets, cols = datasets.extract_arrow_snippets(
                audio_path, chart_path, half_frames=5
            )
            self.assertEqual(len(times_norm), 0)
            self.assertEqual(snippets.shape[0], 0)
            self.assertEqual(snippets.shape[1], 11)
            self.assertEqual(snippets.shape[2], 128)
            self.assertEqual(len(cols), 0)

    def test_extract_arrow_snippets_invalid_chart_raises(self):
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = os.path.join(tmpdir, "bad_chart.txt")
            with open(chart_path, "w") as f:
                f.write("TITLE Bad\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                f.write("4012 7.5\n")  # invalid base-4 digit '4'
            with self.assertRaises(ValueError):
                datasets.extract_arrow_snippets(audio_path, chart_path, half_frames=5)

    def test_extract_arrow_snippets_empty_arrows_line_raises(self):
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = os.path.join(tmpdir, "bad_chart_empty_arrows.txt")
            with open(chart_path, "w") as f:
                f.write("TITLE Bad\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                f.write(" 7.5\n")  # empty arrows -> ValueError
            with self.assertRaises(ValueError):
                datasets.extract_arrow_snippets(audio_path, chart_path, half_frames=5)

    def test_create_dataset_use_gaussian_target(self):
        ds = datasets.create_dataset(
            TEST_DATA_DIR, use_gaussian_target=True, gaussian_sigma=1.5
        )
        features, targets = _first_batch(ds)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(targets.shape[0], 1)
        self.assertGreater(int(np.sum(targets[0].numpy() > 0)), 0)

    def test_create_dataset_use_gaussian_target_empty_chart(self):
        """Covers create_dataset with use_gaussian_target when chart has 0 steps."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_chart = os.path.join(tmpdir, pathlib.Path(audio_path).stem + ".txt")
            with open(empty_chart, "w") as f:
                f.write("TITLE Empty\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
            shutil.copy2(
                audio_path,
                os.path.join(tmpdir, os.path.basename(audio_path)),
            )
            ds = datasets.create_dataset(
                tmpdir, use_gaussian_target=True, gaussian_sigma=1.0
            )
            features, targets = _first_batch(ds)
            self.assertEqual(features.shape[0], 1)
            self.assertEqual(targets.shape[0], 1)
            self.assertEqual(int(np.sum(targets[0].numpy())), 0)

    def test_create_dataset_batch_size_two(self):
        ds = datasets.create_dataset(TEST_DATA_DIR, batch_size=2)
        features, targets = _first_batch(ds)
        self.assertGreaterEqual(features.shape[0], 1)
        self.assertLessEqual(features.shape[0], 2)
        self.assertEqual(features.shape[0], targets.shape[0])
        self.assertEqual(features.shape[2], 128)
        self.assertEqual(targets.shape[2], 1)

    def test_base4_to_int_valid(self):
        self.assertEqual(datasets._base4_to_int("0"), 0)
        self.assertEqual(datasets._base4_to_int("123"), 27)

    def test_base4_to_int_empty_raises(self):
        with self.assertRaises(ValueError):
            datasets._base4_to_int("")

    def test_base4_to_int_invalid_char_raises(self):
        with self.assertRaises(ValueError):
            datasets._base4_to_int("4012")

    def test_create_target(self):
        # frame_idx < spec_length (in range) and frame_idx >= spec_length (skipped)
        spec_length = 100
        times = np.array([0.5, 0.99, 2.0])  # frames 50, 99, 200 (200 out of range)
        cols = np.array([0, 0, 0], dtype=np.int32)
        target = datasets._create_target(times, cols, spec_length)
        self.assertEqual(target.shape, (spec_length, 1))
        self.assertEqual(target[50, 0], 1.0)
        self.assertEqual(target[99, 0], 1.0)
        self.assertEqual(int(np.sum(target)), 2)

    def test_create_target_gaussian_empty_times(self):
        out = datasets._create_target_gaussian(
            np.array([]), np.array([], dtype=np.int32), 100, sigma=1.0
        )
        self.assertEqual(out.shape, (100, 1))
        self.assertEqual(int(np.sum(out)), 0)

    def test_create_target_gaussian_skips_col_ge_n_target(self):
        # cols with value >= _N_TARGET (1) are skipped
        spec_length = 50
        times = np.array([0.0, 0.5])
        cols = np.array([0, 1], dtype=np.int32)  # col 1 >= _N_TARGET=1, skipped
        target = datasets._create_target_gaussian(times, cols, spec_length, sigma=1.0)
        self.assertEqual(target.shape, (spec_length, 1))
        self.assertGreater(float(target[0, 0]), 0)
        # Second onset (col 1) is skipped; only one peak from first onset
        self.assertLessEqual(int(np.sum(target > 0)), 7)  # kernel width ~3*sigma

    def test_temporal_augment_scipy_warp_longer(self):
        spec = np.random.randn(constants.N_MELS, 100).astype(np.float32)
        labels = np.zeros((100, 1), dtype=np.float32)
        with mock.patch("numpy.random.uniform", return_value=1.1):
            out_spec, out_labels = datasets._temporal_augment_scipy(spec, labels)
        self.assertEqual(out_spec.shape, spec.shape)
        self.assertEqual(out_labels.shape, labels.shape)

    def test_temporal_augment_scipy_warp_shorter(self):
        spec = np.random.randn(constants.N_MELS, 100).astype(np.float32)
        labels = np.zeros((100, 1), dtype=np.float32)
        with mock.patch("numpy.random.uniform", return_value=0.9):
            out_spec, out_labels = datasets._temporal_augment_scipy(spec, labels)
        self.assertEqual(out_spec.shape, spec.shape)
        self.assertEqual(out_labels.shape, labels.shape)

    def test_apply_spec_augment(self):
        spec = np.random.randn(constants.N_MELS, 200).astype(np.float32)
        out = datasets._apply_spec_augment(spec, F=10, T=20)
        self.assertEqual(out.shape, spec.shape)
        self.assertFalse(np.array_equal(out, spec))
        self.assertGreater(int(np.sum(out == 0)), 0)

    def test_load_and_preprocess_paths(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, target = datasets._load_and_preprocess_paths(
            audio_path, chart_path, use_gaussian_target=False, gaussian_sigma=1.0
        )
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertEqual(target.ndim, 2)
        self.assertEqual(target.shape[1], 1)
        self.assertGreater(int(np.sum(target > 0)), 0)

    def test_load_and_preprocess_paths_gaussian(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, target = datasets._load_and_preprocess_paths(
            audio_path, chart_path, use_gaussian_target=True, gaussian_sigma=1.5
        )
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertGreater(int(np.sum(target > 0)), 0)

    def test_augment_features_numpy_no_aug(self):
        features = np.random.randn(200, constants.N_MELS).astype(np.float32)
        target = np.zeros((200, 1), dtype=np.float32)
        out_f, out_t = datasets._augment_features_numpy(features, target, False, False)
        self.assertEqual(out_f.shape, features.shape)
        self.assertEqual(out_t.shape, target.shape)

    def test_augment_features_numpy_temporal_and_spec(self):
        features = np.random.randn(200, constants.N_MELS).astype(np.float32)
        target = np.zeros((200, 1), dtype=np.float32)
        with mock.patch("numpy.random.uniform", return_value=1.0):
            out_f, out_t = datasets._augment_features_numpy(
                features, target, True, True
            )
        self.assertEqual(out_f.shape, features.shape)
        self.assertEqual(out_t.shape, target.shape)

    def test_load_and_preprocess_py_callback(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        ap = tf.constant(audio_path)
        cp = tf.constant(chart_path)
        features, target = datasets._load_and_preprocess_py_callback(
            ap,  # type: ignore[arg-type]
            cp,  # type: ignore[arg-type]
            False,
            1.0,
        )
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertEqual(target.shape[1], 1)

    def test_augment_py_callback(self):
        features = np.random.randn(100, constants.N_MELS).astype(np.float32)
        target = np.zeros((100, 1), dtype=np.float32)
        f_t = tf.constant(features)
        t_t = tf.constant(target)
        out_f, out_t = datasets._augment_py_callback(f_t, t_t, False, False)  # type: ignore[arg-type]
        self.assertEqual(out_f.shape, features.shape)
        self.assertEqual(out_t.shape, target.shape)

    def test_arrow_snippets_py_callback(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        ap = tf.constant(audio_path)
        cp = tf.constant(chart_path)
        times, snippets, cols = datasets._arrow_snippets_py_callback(ap, cp, 5)  # type: ignore[arg-type]
        self.assertEqual(times.ndim, 1)
        self.assertEqual(snippets.shape[1], 11)
        self.assertEqual(snippets.shape[2], constants.N_MELS)
        self.assertEqual(len(cols), len(times))

    def test_parse_step_chart_py_callback(self):
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        cp = tf.constant(chart_path)
        times, cols = datasets._parse_step_chart_py_callback(cp)  # type: ignore[arg-type]
        self.assertEqual(times.ndim, 1)
        self.assertEqual(cols.ndim, 1)
        self.assertEqual(len(times), len(cols))

    def test_parse_step_chart_difficulty_break(self):
        """Chart with second DIFFICULTY line stops parsing (break branch)."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi_diff.txt")
            with open(path, "w") as f:
                f.write("TITLE X\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                f.write("0000 0.5\n")
                f.write("DIFFICULTY Easy\n")
                f.write("0000 1.0\n")
            times, _ = datasets._parse_step_chart(path, binary_timings=False)
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], 0.5)

    def test_audio_to_spectrogram_resample_branch(self):
        """Cover the sr != _TARGET_SR resample path in audio_to_spectrogram."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with mock.patch("stepcovnet.datasets.librosa") as mock_librosa:
            y_orig = np.random.randn(44100 * 2).astype(np.float32)
            mock_librosa.load.return_value = (y_orig, 22050)
            mock_librosa.resample.return_value = np.random.randn(44100 * 2).astype(
                np.float32
            )
            mock_librosa.power_to_db.side_effect = lambda x, **kw: x.astype(np.float64)
            mock_librosa.feature.melspectrogram.return_value = np.random.rand(
                constants.N_MELS, 200
            ).astype(np.float64)
            _ = datasets.audio_to_spectrogram(audio_path)
            mock_librosa.resample.assert_called_once()

    def test_load_and_preprocess_tf_map(self):
        """Cover _load_and_preprocess_tf_map (tf.data map helper)."""
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, target = datasets._load_and_preprocess_tf_map(
            tf.constant(audio_path),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            use_gaussian_target=False,
            gaussian_sigma=1.0,
        )
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertEqual(target.shape[1], 1)

    def test_apply_augmentations_tf_map(self):
        """Cover _apply_augmentations_tf_map (tf.data map helper)."""
        features = np.random.randn(100, constants.N_MELS).astype(np.float32)
        target = np.zeros((100, 1), dtype=np.float32)
        aug_f, aug_t = datasets._apply_augmentations_tf_map(
            tf.constant(features),  # type: ignore[arg-type]
            tf.constant(target),  # type: ignore[arg-type]
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
        )
        self.assertEqual(aug_f.shape[1], constants.N_MELS)
        self.assertEqual(aug_t.shape[1], 1)

    def test_process_pair_with_snippets_tf_map(self):
        """Cover _process_pair_with_snippets_tf_map (tf.data map helper)."""
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, _ = datasets._process_pair_with_snippets_tf_map(
            tf.constant(audio_path),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            snippet_half_frames=5,
        )
        self.assertIn("timing_input", features)
        self.assertIn("snippet_input", features)
        self.assertEqual(features["timing_input"].shape[-1], 1)
        self.assertEqual(features["snippet_input"].shape[1], 11)
        self.assertEqual(features["snippet_input"].shape[2], constants.N_MELS)

    def test_process_pair_tf_map(self):
        """Cover _process_pair_tf_map (tf.data map helper)."""
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        times, cols = datasets._process_pair_tf_map(tf.constant(chart_path))  # type: ignore[arg-type]
        self.assertEqual(times.shape[-1], 1)
        self.assertEqual(len(times.shape), 2)
        self.assertEqual(len(cols.shape), 1)


if __name__ == "__main__":
    unittest.main()
