import unittest
import unittest.mock

import keras

from stepcovnet import onset_metric_names as mn
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
                ordered_onset_match=633 / 634,
            ),
            0.9,
        )

    def test_overfit_gate_callback_publishes_metrics(self) -> None:
        callback = trainers.OverfitGateCallback()
        logs = {
            mn.val_name(mn.TOKEN_ACCURACY): 0.95,
            mn.val_name(mn.TIMING_MATCH_TEACHER): 633 / 634,
        }
        callback.on_epoch_end(0, logs)
        self.assertEqual(logs[mn.val_name(mn.GATE_TEACHER)], 0.95)
        self.assertEqual(logs[mn.val_name("overfit_gate")], 0.95)
        self.assertEqual(logs[mn.val_name("ordered_onset_match")], 633 / 634)

    def test_overfit_gate_callback_early_stops_on_primary_monitor(self) -> None:
        class _EarlyStopModel:
            stop_training = False

        callback = trainers.OverfitGateCallback(
            early_stop=True,
            early_stop_monitor=mn.val_name("ordered_onset_match"),
            min_score=1.0,
            patience=2,
        )
        fit_model = _EarlyStopModel()
        callback.set_model(fit_model)
        perfect_logs = {
            mn.val_name(mn.TOKEN_ACCURACY): 0.95,
            mn.val_name(mn.TIMING_MATCH_TEACHER): 1.0,
            mn.val_name("ordered_onset_match"): 1.0,
        }
        callback.on_epoch_end(0, perfect_logs)
        self.assertFalse(fit_model.stop_training)
        callback.on_epoch_end(1, perfect_logs)
        self.assertTrue(fit_model.stop_training)


class BuildArOptimizerTest(unittest.TestCase):
    def test_build_ar_optimizer_returns_adam(self) -> None:
        from stepcovnet.onset_ar import config

        run_config = config.ArRunConfig(mixed_precision=True, learning_rate=1e-4)
        optimizer = trainers.build_ar_optimizer(run_config)
        self.assertIsInstance(optimizer, keras.optimizers.Adam)


if __name__ == "__main__":
    unittest.main()
