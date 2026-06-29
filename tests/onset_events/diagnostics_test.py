import unittest

import numpy as np

from stepcovnet.onset_events import config, diagnostics


class OnsetEventDiagnosticsTest(unittest.TestCase):
    def test_confidence_stats_counts_thresholds(self):
        stats = diagnostics.confidence_stats(np.array([[0.0, 0.49, 0.51, 0.9]]))

        self.assertEqual(stats["count_ge_0.5"], 2)
        self.assertEqual(stats["count_ge_0.1"], 3)
        self.assertAlmostEqual(stats["max"], 0.9)

    def test_assignment_summary_reports_pair_counts(self):
        pred_times = np.array([[1.0, 5.0]])
        gt_times = np.array([[1.01, 5.01, 0.0]])
        gt_mask = np.array([[1.0, 1.0, 0.0]])

        summary = diagnostics.assignment_summary(
            pred_times,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
        )

        self.assertEqual(summary["num_gt"], 2)
        self.assertEqual(summary["hungarian_l1_pairs"], 2)
        self.assertEqual(summary["hungarian_eval_pairs"], 2)

    def test_diagnose_overfit_outputs_zero_f1_when_all_conf_low(self):
        experiment = config.OnsetEventExperimentConfig(
            dataset=config.OnsetEventDatasetConfig(),
            model=config.OnsetEventModelConfig(num_queries=4),
            run=config.OnsetEventRunConfig(),
        )
        report = diagnostics.diagnose_overfit_outputs(
            model_path="model.keras",
            experiment=experiment,
            pred_times=np.array([[1.0, 2.0, 3.0, 4.0]]),
            pred_confidence=np.array([[0.01, 0.02, 0.03, 0.04]]),
            gt_times=np.array([[1.0, 2.0, 0.0, 0.0]]),
            gt_mask=np.array([[1.0, 1.0, 0.0, 0.0]]),
            duration_sec=10.0,
        )

        self.assertEqual(report.eval_f1, 0.0)
        self.assertEqual(report.eval_tp, 0.0)
        self.assertEqual(report.confidence["count_ge_0.5"], 0)
        self.assertEqual(report.timing_match_n_ref, 2)
        self.assertEqual(report.timing_match_n_matched, 0)

    def test_sweep_confidence_thresholds_finds_best(self):
        pred_times = np.array([[1.0, 5.0]])
        pred_confidence = np.array([[0.9, 0.2]])
        gt_times = np.array([[1.0, 0.0]])
        gt_mask = np.array([[1.0, 0.0]])

        sweep = diagnostics.sweep_confidence_thresholds(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
            thresholds=(0.5, 0.8),
        )

        self.assertEqual(len(sweep), 2)
        self.assertGreater(sweep[0]["f1"], 0.0)
        self.assertGreaterEqual(sweep[0]["f1"], sweep[1]["f1"])

    def test_oracle_uniform_grid_coverage_counts_nearby_slots(self):
        duration_sec = 10.0
        num_queries = 4
        grid_times = diagnostics.uniform_grid_ref_times_sec(num_queries, duration_sec)
        gt_times = np.array([grid_times[0], grid_times[2], 9.99, 0.0])
        gt_mask = np.array([1.0, 1.0, 1.0, 0.0])

        oracle = diagnostics.oracle_uniform_grid_coverage(
            gt_times,
            gt_mask,
            duration_sec,
            num_queries,
            tolerance_sec=0.02,
        )

        self.assertEqual(oracle["num_gt"], 3)
        self.assertEqual(oracle["grid_matchable"], 2)
        self.assertAlmostEqual(oracle["grid_matchable_fraction"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
