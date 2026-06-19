"""Tests for dataset_prep.pack_results."""

import unittest

from stepcovnet.dataset_prep import pack_results


class PackResultsTest(unittest.TestCase):
    def test_pack_result_exported_when_reason_none(self):
        self.assertEqual(
            pack_results.pack_result(None), pack_results.PACK_RESULT_EXPORTED
        )

    def test_pack_result_skipped_for_policy_reasons(self):
        self.assertEqual(
            pack_results.pack_result(pack_results.REASON_NO_DANCE_SINGLE),
            pack_results.PACK_RESULT_SKIPPED,
        )

    def test_pack_result_error_for_unexpected_reasons(self):
        self.assertEqual(
            pack_results.pack_result(pack_results.REASON_IO_ERROR),
            pack_results.PACK_RESULT_ERROR,
        )


if __name__ == "__main__":
    unittest.main()
