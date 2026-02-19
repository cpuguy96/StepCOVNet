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
            model, history = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                normalize=True,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
                model_params={
                    "initial_filters": 8,
                    "depth": 1,
                    "dilation_rates": [1, 2],
                    "dropout_rate": 0.0,
                },
                take_count=1,
                epoch=3,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)

    @unittest.skip("Test takes too long.")
    def test_run_train_overfits_single_song(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            callback_root_dir = os.path.join(temp_dir, "callbacks")
            model_output_dir = os.path.join(temp_dir, "models")
            model, history = trainers.run_train(
                data_dir=TEST_DATA_DIR,
                val_data_dir=TEST_DATA_DIR,
                batch_size=1,
                normalize=True,
                apply_temporal_augment=False,
                should_apply_spec_augment=False,
                use_gaussian_target=False,
                gaussian_sigma=0.0,
                model_params={
                    "initial_filters": 16,
                    "depth": 2,
                    "dilation_rates": [1, 2, 4, 8],
                    "dropout_rate": 0.0,
                },
                take_count=1,
                epoch=300,
                callback_root_dir=callback_root_dir,
                model_output_dir=model_output_dir,
            )
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
        self.assertTrue("val_onset_f1_score" in history.history)
        self.assertAlmostEqual(history.history["val_onset_f1_score"][-1], 1.0, places=2)

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
