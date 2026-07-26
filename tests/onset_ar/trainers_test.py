import unittest

import keras

from stepcovnet import onset_metric_names as mn
from stepcovnet.onset_ar import config as ar_config
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

    def test_should_attach_overfit_gate_only_for_overfit_runs(self) -> None:
        overfit = ar_config.ArRunConfig(overfit_one_song=True)
        multi_song = ar_config.ArRunConfig(overfit_one_song=False)
        self.assertTrue(trainers.should_attach_overfit_gate_callback(overfit))
        self.assertFalse(trainers.should_attach_overfit_gate_callback(multi_song))

    def test_early_stopping_patience_defaults_disabled(self) -> None:
        run = ar_config.ArRunConfig()
        self.assertEqual(run.early_stopping_patience, 0)

    def test_experiment_name_includes_split_and_epochs(self) -> None:
        experiment = ar_config.ArExperimentConfig(
            dataset=ar_config.ArDatasetConfig(),
            model=ar_config.ArModelConfig(
                patch_frames=8,
                d_model=384,
                n_enc_layers=4,
                n_dec_layers=4,
            ),
            run=ar_config.ArRunConfig(
                epochs=500,
                early_stopping_patience=25,
                overfit_one_song=False,
            ),
        )
        name = trainers._get_experiment_name(
            experiment,
            n_train_samples=200,
            n_val_samples=50,
        )
        self.assertEqual(
            name,
            "AR_ONSET-P8-d384-enc4-dec4-200t50v-ep500-es25",
        )

    def test_experiment_name_overfit_tag(self) -> None:
        experiment = ar_config.ArExperimentConfig(
            dataset=ar_config.ArDatasetConfig(),
            model=ar_config.ArModelConfig(
                patch_frames=8,
                d_model=384,
                n_enc_layers=4,
                n_dec_layers=4,
            ),
            run=ar_config.ArRunConfig(epochs=400, overfit_one_song=True),
        )
        name = trainers._get_experiment_name(experiment)
        self.assertEqual(name, "AR_ONSET-P8-d384-enc4-dec4-overfit-ep400")

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

    def test_metric_alias_callback_publishes_monitor_key(self) -> None:
        callback = trainers.MetricAliasCallback()
        logs = {
            mn.val_name(mn.AUX_F1_HUNGARIAN): 0.178,
            mn.AUX_F1_HUNGARIAN: 0.42,
        }
        callback.on_epoch_end(0, logs)
        monitor = mn.resolve_checkpoint_metric("val_aux_f1_hungarian")
        self.assertIn(monitor, logs)
        self.assertEqual(logs[monitor], 0.178)
        self.assertEqual(logs["event_onset_f1"], 0.42)

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
