import os
import unittest
from unittest import mock

import numpy as np

from stepcovnet import generator

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')


class GeneratorTest(unittest.TestCase):
    def test_generate_output_data(self):
        mock_onset_model = mock.MagicMock()
        mock_onset_model.predict.return_value = np.random.random((1, 100, 1)).astype(np.float32)

        mock_arrow_model = mock.MagicMock()

        def _arrow_pred_mock(x):
            num_arrows = x.shape[1]
            return np.random.random((1, num_arrows, 256)).astype(np.float32)

        mock_arrow_model.predict.side_effect = _arrow_pred_mock

        txt_output = generator.generate_output_data(
            os.path.join(TEST_DATA_DIR, "tide.ogg"), song_title="Test Song", bpm=120,
            onset_model=mock_onset_model, arrow_model=mock_arrow_model
        )

        self.assertEqual(txt_output.title, "Test Song")
        self.assertEqual(txt_output.bpm, 120)
        self.assertTrue("Challenge" in txt_output.notes)
        # Post-processing can remove some of the onsets predicted. So the total number of arrows chosen will be less
        # than or equal to the total number of onsets.
        self.assertLessEqual(len(txt_output.notes["Challenge"]), 100)
        for onset, arrow in txt_output.notes["Challenge"]:
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
