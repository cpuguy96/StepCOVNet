import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

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

        self.assertEqual(np.sum(targets[0]), 384)

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
        self.assertGreater(seq_len, 0)
        self.assertLessEqual(seq_len, constants.MAX_STEPS)
        self.assertGreater(np.sum(targets[0].numpy() > 0), 0)

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
        self.assertGreater(spec.shape[1], 0)

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
        self.assertGreater(np.sum(targets[0].numpy() > 0), 0)

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
            self.assertEqual(np.sum(targets[0].numpy()), 0)

    def test_create_dataset_apply_temporal_augment(self):
        ds = datasets.create_dataset(TEST_DATA_DIR, apply_temporal_augment=True)
        features, targets = _first_batch(ds)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 128)
        self.assertEqual(targets.shape[0], 1)

    def test_create_dataset_should_apply_spec_augment(self):
        ds = datasets.create_dataset(TEST_DATA_DIR, should_apply_spec_augment=True)
        features, targets = _first_batch(ds)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 128)
        self.assertEqual(targets.shape[0], 1)

    def test_create_dataset_batch_size_two(self):
        ds = datasets.create_dataset(TEST_DATA_DIR, batch_size=2)
        features, targets = _first_batch(ds)
        self.assertGreaterEqual(features.shape[0], 1)
        self.assertLessEqual(features.shape[0], 2)
        self.assertEqual(features.shape[0], targets.shape[0])
        self.assertEqual(features.shape[2], 128)
        self.assertEqual(targets.shape[2], 1)


if __name__ == "__main__":
    unittest.main()
