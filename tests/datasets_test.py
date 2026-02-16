import os
import unittest

import numpy as np

from stepcovnet import datasets

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')


class DatasetsTest(unittest.TestCase):

    def test_create_dataset(self):
        ds = datasets.create_dataset(TEST_DATA_DIR)

        features, targets = next(iter(ds.take(1))) # type: ignore

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 128)  # Mel bins
        self.assertEqual(features.shape[1], 12852)  # Time steps

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[2], 1)  # Channels
        self.assertEqual(targets.shape[1], 12852)  # Time steps

        self.assertTrue(np.any(targets > 0))

    def test_create_dataset_with_empty_directory_raises_error(self):
        with self.assertRaises(ValueError):
            datasets.create_dataset("")

    def test_create_arrow_dataset(self):
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR)
        features, targets = next(iter(ds.take(1)))  # type: ignore

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[1], 634)  # Timings

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[1], 634)  # Arrows

        self.assertTrue(np.any(targets > 0))

    def test_create_arrow_dataset_with_empty_directory_raises_error(self):
        with self.assertRaises(ValueError):
            datasets.create_arrow_dataset("")


if __name__ == '__main__':
    unittest.main()
