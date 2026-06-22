"""Tests for dataset_prep.constants."""

import unittest

from stepcovnet.dataset_prep import constants
from stepcovnet.onset_events import charts


class DatasetPrepConstantsTest(unittest.TestCase):
    def test_schema_version_is_one(self):
        self.assertEqual(constants.SCHEMA_VERSION, 1)

    def test_max_steps_matches_onset_charts(self):
        self.assertEqual(constants.MAX_STEPS_PER_CHART, charts.MAX_STEPS_PER_CHART)

    def test_difficulty_rank_order(self):
        self.assertGreater(
            constants.DIFFICULTY_RANK["challenge"],
            constants.DIFFICULTY_RANK["beginner"],
        )
        self.assertEqual(constants.DIFFICULTY_RANK["custom"], 0)


if __name__ == "__main__":
    unittest.main()
