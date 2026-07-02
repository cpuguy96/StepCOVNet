"""Tests for canonical onset metric names and legacy aliases."""

import unittest

from stepcovnet import onset_metric_names as mn


class OnsetMetricNamesTest(unittest.TestCase):
    def test_resolve_checkpoint_legacy_gate(self) -> None:
        self.assertEqual(
            mn.resolve_checkpoint_metric("val_overfit_gate"),
            "val_overfit_gate",
        )

    def test_resolve_checkpoint_canonical_gate(self) -> None:
        self.assertEqual(
            mn.resolve_checkpoint_metric("val_gate_teacher"),
            "val_overfit_gate",
        )

    def test_resolve_checkpoint_canonical_timing(self) -> None:
        self.assertEqual(
            mn.resolve_checkpoint_metric("val_timing_match_teacher"),
            "val_ordered_onset_match",
        )

    def test_publish_legacy_val_aliases(self) -> None:
        logs = {"val_timing_match_teacher": 0.99, "val_aux_f1_hungarian": 0.98}
        mn.publish_legacy_val_aliases(logs)
        self.assertEqual(logs["val_ordered_onset_match"], 0.99)
        self.assertEqual(logs["val_event_onset_f1"], 0.98)

    def test_publish_legacy_from_legacy_keys(self) -> None:
        logs = {"val_ordered_onset_match": 1.0}
        mn.publish_legacy_val_aliases(logs)
        self.assertEqual(logs["val_timing_match_teacher"], 1.0)


if __name__ == "__main__":
    unittest.main()
