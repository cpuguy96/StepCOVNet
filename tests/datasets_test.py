import os
import unittest

import numpy as np

from stepcovnet import constants
from stepcovnet import datasets

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class DatasetsTest(unittest.TestCase):

    def test_create_dataset(self):
        ds = datasets.create_dataset(TEST_DATA_DIR)

        features, targets = next(iter(ds.take(1)))  # type: ignore

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 128)  # Mel bins
        self.assertEqual(features.shape[1], 11726)  # Time steps

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[2], 1)  # Channels
        self.assertEqual(targets.shape[1], 11726)  # Time steps

        self.assertEqual(np.sum(targets[0]), 384)

    def test_create_dataset_with_empty_directory_raises_error(self):
        with self.assertRaises(ValueError):
            datasets.create_dataset("")

    def test_create_arrow_dataset(self):
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR)
        features, targets = next(iter(ds.take(1)))  # type: ignore

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[1], 384)  # Timings

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[1], 384)  # Arrows

        self.assertTrue(np.all(targets[0] > 0))

    def test_create_arrow_dataset_with_empty_directory_raises_error(self):
        with self.assertRaises(ValueError):
            datasets.create_arrow_dataset("")

    def test_create_arrow_dataset_batch_size_two_timing_only(self):
        """Multi-sample batching without snippets uses (times [None,1], cols [None])."""
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR, batch_size=2)
        batch = next(iter(ds.take(1)))
        times, cols = batch
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
        batch = next(iter(ds.take(1)))
        features, targets = batch
        self.assertIn("timing_input", features)
        self.assertIn("snippet_input", features)
        times = features["timing_input"]
        snippets = features["snippet_input"]
        self.assertEqual(times.shape[0], 1)
        seq_len = times.shape[1]
        self.assertGreater(seq_len, 0)
        self.assertLessEqual(seq_len, constants.MAX_STEPS)
        self.assertEqual(times.shape[2], 1)
        self.assertEqual(snippets.shape[0], 1)
        self.assertEqual(snippets.shape[1], seq_len)
        self.assertEqual(snippets.shape[2], 11)
        self.assertEqual(snippets.shape[3], 128)
        self.assertEqual(times.shape[1], snippets.shape[1])
        self.assertEqual(times.shape[1], targets.shape[1])
        self.assertGreater(np.sum(targets[0].numpy() > 0), 0)

    def test_create_arrow_dataset_with_audio_snippets_batch_size_two(self):
        """Multi-sample batching with snippets uses correct padded_shapes (dict + cols)."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            batch_size=2,
            snippet_half_frames=5,
        )
        batch = next(iter(ds.take(1)))
        features, targets = batch
        self.assertIn("timing_input", features)
        self.assertIn("snippet_input", features)
        times = features["timing_input"]
        snippets = features["snippet_input"]
        batch_dim = times.shape[0]
        self.assertGreaterEqual(batch_dim, 1)
        self.assertLessEqual(batch_dim, 2)
        self.assertEqual(snippets.shape[0], batch_dim)
        self.assertEqual(targets.shape[0], batch_dim)
        self.assertEqual(times.shape[2], 1)
        self.assertEqual(snippets.shape[2], 11)
        self.assertEqual(snippets.shape[3], 128)
        self.assertEqual(times.shape[1], snippets.shape[1])
        self.assertEqual(times.shape[1], targets.shape[1])

    def test_extract_arrow_snippets_returns_correct_shapes(self):
        pairs = datasets._load_and_pair_files(TEST_DATA_DIR)
        self.assertGreater(len(pairs), 0)
        audio_path, chart_path = pairs[0]
        times_norm, snippets, cols = datasets.extract_arrow_snippets(
            audio_path, chart_path, half_frames=5
        )
        self.assertEqual(times_norm.ndim, 1)
        self.assertEqual(snippets.shape[0], len(times_norm))
        self.assertEqual(snippets.shape[1], 11)
        self.assertEqual(snippets.shape[2], 128)
        self.assertEqual(cols.shape[0], len(times_norm))


if __name__ == "__main__":
    unittest.main()
