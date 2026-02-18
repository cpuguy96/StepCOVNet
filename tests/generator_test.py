import os
import unittest
from unittest import mock

import keras
import numpy as np

from stepcovnet import generator
from stepcovnet import models

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')


class GeneratorTest(unittest.TestCase):
    def test_generate_output_data_with_mock_models(self):

        def _onset_pred_mock(x):
            self.assertEqual(x.shape, (1, 12852, 128))
            return np.random.random((1, 12852, 1)).astype(np.float32)

        mock_onset_model = mock.MagicMock()
        mock_onset_model.predict.side_effect = _onset_pred_mock

        def _arrow_pred_mock(x):
            num_arrows = x.shape[1]
            self.assertEqual(x.shape, (1, num_arrows, 1))
            return np.random.random((1, num_arrows, 256)).astype(np.float32)

        mock_arrow_model = mock.MagicMock()
        mock_arrow_model.predict.side_effect = _arrow_pred_mock

        for use_post_processing in [True, False]:
            with self.subTest(f"use_post_processing={use_post_processing}"):
                output_data = generator.generate_output_data(
                    audio_path=os.path.join(TEST_DATA_DIR, "tide.ogg"), song_title="Test Song", bpm=120,
                    onset_model=mock_onset_model, arrow_model=mock_arrow_model, use_post_processing=use_post_processing
                )
                self.assertEqual(output_data.title, "Test Song")
                self.assertEqual(output_data.bpm, 120)
                self.assertTrue("Challenge" in output_data.notes)
                self.assertLessEqual(len(output_data.notes["Challenge"]), 12852)
                for onset, arrow in output_data.notes["Challenge"]:
                    self.assertNotIn("4", arrow)
                    # 0 is used as padding for training datasets. So there should be none present.
                    self.assertNotEqual(arrow, "0000")

    def test_generate_output_data(self):
        onset_model = keras.models.load_model(
            os.path.join(TEST_DATA_DIR, "stepcovnet_ONSET-tide_overfit.keras"),
            custom_objects={"_crop_to_match": models._crop_to_match},
            compile=False)
        arrow_model = keras.models.load_model(os.path.join(TEST_DATA_DIR, "arrow_model.keras"),
                                              compile=False,
                                              custom_objects={"PositionalEncoding": models.PositionalEncoding})


        for use_post_processing in [True, False]:
            with self.subTest(f"use_post_processing={use_post_processing}"):
                output_data = generator.generate_output_data(
                    audio_path=os.path.join(TEST_DATA_DIR, "tide.ogg"),
                    song_title="Tide", bpm=175,
                    onset_model=onset_model, arrow_model=arrow_model
                )
                self.assertEqual(output_data.title, "Tide")
                self.assertEqual(output_data.bpm, 175)
                self.assertTrue("Challenge" in output_data.notes)
                self.assertEqual(len(output_data.notes["Challenge"]), 606)
                for onset, arrow in output_data.notes["Challenge"]:
                    self.assertNotIn("4", arrow)
                    # 0 is used as padding for training datasets. So there
                    # should be none present.
                    self.assertNotEqual(arrow, "0000")

    def test_output_data_generate_txt_output(self):
        output_data = generator.OutputData(
            title="Test Song",
            bpm=120,
            notes={
                "Challenge": [
                    ("3103", "0.04")
                ]
            }
        )
        expected_output = 'TITLE Test Song\nBPM 120\nNOTES\nDIFFICULTY Challenge\n0.04 3103\n'
        self.assertEqual(output_data.generate_txt_output(), expected_output)


if __name__ == '__main__':
    unittest.main()
