import json
import pathlib
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class ArTideIterTrainingLogTest(unittest.TestCase):
    def _import_training_log(self):
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import training_log  # noqa: PLC0415

        return training_log

    def test_run_kind_fresh_vs_retry(self) -> None:
        training_log = self._import_training_log()
        self.assertEqual(
            training_log.run_kind(attempt=1, retry_reason=""),
            "fresh",
        )
        self.assertEqual(
            training_log.run_kind(attempt=2, retry_reason=""),
            "retry",
        )
        self.assertEqual(
            training_log.run_kind(
                attempt=2,
                retry_reason="removed in-loop AR decode",
            ),
            "retry — removed in-loop AR decode",
        )

    def test_format_log_heading_shows_attempt(self) -> None:
        training_log = self._import_training_log()
        self.assertEqual(
            training_log.format_log_heading("iter31", 1, "2026-06-29T01:00:00"),
            "### iter31 (2026-06-29T01:00:00)",
        )
        self.assertEqual(
            training_log.format_log_heading("iter31", 2, "2026-06-29T02:00:00"),
            "### iter31 · attempt 2 (2026-06-29T02:00:00)",
        )

    def test_count_logged_attempts_reads_results_jsonl(self) -> None:
        training_log = self._import_training_log()
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = pathlib.Path(tmp)
            training_log.ITER_DIR = iter_dir
            training_log.RESULTS_JSONL = iter_dir / "results.jsonl"
            training_log.RESULTS_JSONL.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "iter31", "attempt": 1}),
                        json.dumps({"id": "iter32", "attempt": 1}),
                        json.dumps({"id": "iter31", "attempt": 2}),
                    ],
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(training_log.count_logged_attempts("iter31"), 2)
            self.assertEqual(training_log.count_logged_attempts("iter32"), 1)

    def test_teacher_metrics_perfect_requires_all_val_keys(self) -> None:
        training_log = self._import_training_log()
        self.assertFalse(
            training_log.teacher_metrics_perfect(
                {
                    "val_token_accuracy": 1.0,
                    "val_ordered_onset_match": 0.9984,
                    "val_event_onset_f1": 1.0,
                    "val_overfit_gate": 0.9984,
                },
            ),
        )
        self.assertTrue(
            training_log.teacher_metrics_perfect(
                {
                    "val_token_accuracy": 1.0,
                    "val_ordered_onset_match": 1.0,
                    "val_event_onset_f1": 1.0,
                    "val_overfit_gate": 1.0,
                },
            ),
        )

    def test_teacher_report_perfect_requires_full_ordered_match(self) -> None:
        training_log = self._import_training_log()
        self.assertFalse(
            training_log.teacher_report_perfect(
                {
                    "ordered_onset_match": {
                        "n_matched": 633,
                        "n_denom": 634,
                        "rate": 633 / 634,
                    },
                    "event_f1": 1.0,
                },
            ),
        )
        self.assertTrue(
            training_log.teacher_report_perfect(
                {
                    "ordered_onset_match": {
                        "n_matched": 634,
                        "n_denom": 634,
                        "rate": 1.0,
                    },
                    "event_f1": 1.0,
                },
            ),
        )
        self.assertTrue(
            training_log.teacher_report_perfect(
                {
                    "ordered_onset_match": {
                        "n_matched": 634,
                        "n_denom": 634,
                        "rate": 1.0,
                    },
                    "event_f1": 0.9984,
                },
            ),
        )

    def test_train_log_path_versions_attempts(self) -> None:
        training_log = self._import_training_log()
        with tempfile.TemporaryDirectory() as tmp:
            iter_dir = pathlib.Path(tmp)
            training_log.ITER_DIR = iter_dir
            self.assertEqual(
                training_log.train_log_path("iter31", 1).name,
                "iter31.log",
            )
            self.assertEqual(
                training_log.train_log_path("iter31", 2).name,
                "iter31.attempt2.log",
            )


if __name__ == "__main__":
    unittest.main()
