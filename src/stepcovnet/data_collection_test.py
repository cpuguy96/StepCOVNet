import unittest

from src.stepcovnet import data_collection


class TestDataCollection(unittest.TestCase):

    def test_create_dataset_with_empty_directory_raises_error(self):
        with self.assertRaises(ValueError):
            data_collection.create_dataset("")


if __name__ == '__main__':
    unittest.main()
