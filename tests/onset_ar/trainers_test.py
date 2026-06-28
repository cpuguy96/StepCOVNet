import unittest
from unittest import mock

from stepcovnet.onset_ar import trainers


class TrainersTest(unittest.TestCase):
    def test_lambda_time_ramp_starts_at_zero(self) -> None:
        self.assertEqual(
            trainers.lambda_time_for_epoch(
                0,
                lambda_time_final=1.0,
                ramp_epochs=100,
            ),
            0.01,
        )

    def test_lambda_time_ramp_reaches_final(self) -> None:
        self.assertEqual(
            trainers.lambda_time_for_epoch(
                99,
                lambda_time_final=1.0,
                ramp_epochs=100,
            ),
            1.0,
        )

    def test_lambda_time_no_ramp_uses_final(self) -> None:
        self.assertEqual(
            trainers.lambda_time_for_epoch(
                0,
                lambda_time_final=1.0,
                ramp_epochs=0,
            ),
            1.0,
        )

    def test_scheduled_sampling_ramp_starts_at_zero(self) -> None:
        self.assertEqual(
            trainers.scheduled_sampling_for_epoch(
                0,
                max_p=0.5,
                ramp_epochs=150,
            ),
            0.5 / 150.0,
        )

    def test_scheduled_sampling_ramp_reaches_max(self) -> None:
        self.assertEqual(
            trainers.scheduled_sampling_for_epoch(
                149,
                max_p=0.5,
                ramp_epochs=150,
            ),
            0.5,
        )

    def test_scheduled_sampling_no_ramp_uses_max(self) -> None:
        self.assertEqual(
            trainers.scheduled_sampling_for_epoch(
                0,
                max_p=0.5,
                ramp_epochs=0,
            ),
            0.5,
        )

    def test_scheduled_sampling_warmup_holds_zero(self) -> None:
        self.assertEqual(
            trainers.scheduled_sampling_for_epoch(
                5,
                max_p=1.0,
                ramp_epochs=100,
                warmup_epochs=10,
            ),
            0.0,
        )

    def test_scheduled_sampling_warmup_then_ramps(self) -> None:
        self.assertEqual(
            trainers.scheduled_sampling_for_epoch(
                10,
                max_p=1.0,
                ramp_epochs=100,
                warmup_epochs=10,
            ),
            0.01,
        )

    def test_overfit_gate_score_teacher_fed_only(self) -> None:
        self.assertEqual(
            trainers.overfit_gate_score(
                token_accuracy=0.9,
                event_f1=1.0,
            ),
            0.9,
        )

    def test_overfit_gate_score_includes_ar_decode(self) -> None:
        self.assertEqual(
            trainers.overfit_gate_score(
                token_accuracy=1.0,
                event_f1=1.0,
                ar_decode_f1=0.8,
            ),
            0.8,
        )

    def test_overfit_gate_callback_publishes_metrics(self) -> None:
        callback = trainers.OverfitGateCallback(include_ar_decode=False)
        logs = {"val_token_accuracy": 0.95, "val_event_onset_f1": 1.0}
        callback.on_epoch_end(0, logs)
        self.assertEqual(logs["val_overfit_gate"], 0.95)

    def test_ar_decode_val_schedule_every_n(self) -> None:
        self.assertTrue(
            trainers.should_run_ar_decode_validation(0, every_n_epochs=10),
        )
        self.assertFalse(
            trainers.should_run_ar_decode_validation(3, every_n_epochs=10),
        )
        self.assertTrue(
            trainers.should_run_ar_decode_validation(10, every_n_epochs=10),
        )
        self.assertTrue(
            trainers.should_run_ar_decode_validation(0, every_n_epochs=1),
        )
        self.assertFalse(
            trainers.should_run_ar_decode_validation(0, every_n_epochs=0),
        )

    def test_ar_decode_validation_callback_skips_eager_decode(self) -> None:
        class _StubTrainingModel:
            decode_calls = 0

            def run_ar_decode_eval_eager(self, *_args, **_kwargs):
                type(self).decode_calls += 1
                return 1.0, 0.0, 0.0

            def set_ar_decode_f1_counts(self, tp, fp, fn):
                self.last = (tp, fp, fn)

            @property
            def ar_decode_f1_metric(self):
                return _StubMetric(0.5)

        class _StubMetric:
            def __init__(self, value: float) -> None:
                self._value = value

            def result(self):
                return self._value

        stub = _StubTrainingModel()
        val_batch = {
            "mert_patches": None,
            "patch_mask": None,
            "gt_times": None,
            "gt_mask": None,
        }
        with (
            mock.patch.object(
                trainers.datasets,
                "load_overfit_sample",
                return_value=mock.Mock(),
            ),
            mock.patch.object(
                trainers.datasets,
                "sample_to_training_batch",
                return_value=val_batch,
            ),
        ):
            callback = trainers.ArDecodeValidationCallback(
                stub,  # type: ignore[arg-type]
                experiment_config=mock.Mock(),  # type: ignore[arg-type]
                every_n_epochs=10,
            )

        logs: dict[str, float] = {}
        callback.on_epoch_end(3, logs)
        self.assertEqual(_StubTrainingModel.decode_calls, 0)
        self.assertNotIn("val_ar_decode_event_f1", logs)

        callback.on_epoch_end(10, logs)
        self.assertEqual(_StubTrainingModel.decode_calls, 1)
        self.assertEqual(stub.last, (1.0, 0.0, 0.0))
        self.assertEqual(logs["val_ar_decode_event_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
