import os
import tempfile
import unittest

from stepcovnet import trainers

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class TrainersTest(unittest.TestCase):
    def test_run_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            history, model = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                normalize=True,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=1.0,
                model_params={},
                take_count=1,
                epoch=1,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )

        self.assertIsNotNone(history)
        self.assertIsNotNone(model)

    def test_run_arrow_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            history, model = trainers.run_arrow_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                normalize=True,
                model_params={},
                take_count=1,
                epoch=1,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )

        self.assertIsNotNone(history)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
