import os
import unittest

import numpy as np

from src.stepcovnet import data_collection

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')

class TestDataCollection(unittest.TestCase):
    def test_create_dataset_with_empty_directory_raises_error(self):
        with self.assertRaises(ValueError):
            data_collection.create_dataset("")

    def test_create_dataset(self):
        ds = data_collection.create_dataset(TEST_DATA_DIR)

        features, targets = next(iter(ds.take(1)))

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 128)  # Mel bins
        self.assertEqual(features.shape[1], 12852)  # Time steps

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[2], 1)  # Channels
        self.assertEqual(targets.shape[1], 12852)  # Time steps

        self.assertTrue(np.any(targets > 0))

if __name__ == '__main__':
    unittest.main()
