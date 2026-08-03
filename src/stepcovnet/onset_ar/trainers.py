"""Training loop for AR onset models (``gate-tide-overfit``)."""

from __future__ import annotations

import logging
import pathlib
import re
import time

import keras
import numpy as np
import tensorflow as tf

from stepcovnet import onset_metric_names as mn
from stepcovnet import reproducibility, timing_match, wsl_gpu
from stepcovnet.onset_ar import config, datasets, inference, losses, models
from stepcovnet.onset_events import matching
from stepcovnet.onset_events import trainers as event_trainers

ordered_onset_match_counts_numpy = timing_match.timing_match_counts_numpy
ordered_onset_match_rate_numpy = timing_match.timing_match_rate_numpy


def configure_ar_gpu_training(run_config: config.ArRunConfig) -> None:
    """Enable optional mixed precision and XLA when a GPU is present."""
    if not tf.config.list_physical_devices("GPU"):
        if run_config.mixed_precision or run_config.enable_xla:
            logging.warning(
                "AR GPU options requested (mixed_precision=%s, enable_xla=%s) "
                "but no GPU is visible; skipping.",
                run_config.mixed_precision,
                run_config.enable_xla,
            )
        return
    if run_config.mixed_precision:
        keras.mixed_precision.set_global_policy(
            keras.mixed_precision.Policy("mixed_float16"),
        )
        logging.info("AR training: mixed_float16 policy enabled")
    if run_config.enable_xla:
        tf.config.optimizer.set_jit("autoclustering")
        logging.info("AR training: XLA autoclustering enabled")


def build_ar_optimizer(run_config: config.ArRunConfig) -> keras.optimizers.Optimizer:
    """Build Adam for AR training."""
    return keras.optimizers.Adam(  # type: ignore[return-value]
        learning_rate=run_config.learning_rate,
        clipnorm=5.0,
    )


def _ar_event_onset_counts_numpy(
    pred_times: np.ndarray,
    pred_mask: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
) -> tuple[int, int, int]:
    """Count TP/FP/FN for Hungarian event F1 (auxiliary on overfit)."""
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    pred_mask_arr = np.asarray(pred_mask, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_times, dtype=np.float64).reshape(-1)
    gt_mask_arr = np.asarray(gt_mask, dtype=np.float64).reshape(-1)
    pred_kept = pred[pred_mask_arr > 0.5]
    gt_kept = gt[gt_mask_arr > 0.5]
    if pred_kept.size == 0 and gt_kept.size == 0:
        return 0, 0, 0
    if pred_kept.size == 0:
        return 0, 0, int(gt_kept.size)
    gt_mask_ones = np.ones((1, gt_kept.size), dtype=np.float32)
    result = matching.match_onsets_numpy(
        pred_kept.reshape(1, -1),
        gt_kept.reshape(1, -1),
        gt_mask_ones,
        tolerance_sec=tolerance_sec,
    )
    num_matches = int(result.num_matches[0])
    false_positives = int(pred_kept.size) - num_matches
    false_negatives = int(result.gt_unmatched_mask[0].sum())
    return num_matches, false_positives, false_negatives


def _ar_event_onset_counts_numpy_wrapper(
    pred_times: np.ndarray,
    pred_mask: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp, fp, fn = _ar_event_onset_counts_numpy(
        pred_times,
        pred_mask,
        gt_times,
        gt_mask,
        tolerance_sec=float(tolerance_sec.reshape(-1)[0]),
    )
    return (
        np.array(tp, dtype=np.float64),
        np.array(fp, dtype=np.float64),
        np.array(fn, dtype=np.float64),
    )


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ArOrderedOnsetMatchMetric(keras.metrics.Metric):
    """Primary tide overfit metric: ordered match @ tolerance / max(n_pred, n_gt)."""

    def __init__(
        self,
        tolerance_sec: float = matching.DEFAULT_TOLERANCE_SEC,
        name: str = "ordered_onset_match",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.tolerance_sec = tolerance_sec
        self.n_matched = self.add_weight(name="n_matched", initializer="zeros")
        self.n_gt = self.add_weight(name="n_gt", initializer="zeros")
        self.n_pred = self.add_weight(name="n_pred", initializer="zeros")

    def update_state(
        self,
        pred_times: tf.Tensor,
        target_times: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        _ = sample_weight
        n_matched, n_gt, n_pred = tf.numpy_function(
            timing_match.timing_match_wrapper,
            [
                pred_times,
                target_times,
                np.array([self.tolerance_sec], dtype=np.float64),
            ],
            [tf.float64, tf.float64, tf.float64],
        )
        self.n_matched.assign_add(tf.cast(tf.reshape(n_matched, []), self.dtype))
        self.n_gt.assign_add(tf.cast(tf.reshape(n_gt, []), self.dtype))
        self.n_pred.assign_add(tf.cast(tf.reshape(n_pred, []), self.dtype))

    def result(self) -> tf.Tensor:
        denom = tf.maximum(self.n_pred, self.n_gt)
        return self.n_matched / (denom + 1e-9)

    def reset_state(self) -> None:
        self.n_matched.assign(0.0)
        self.n_gt.assign(0.0)
        self.n_pred.assign(0.0)

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"tolerance_sec": self.tolerance_sec})
        return config_dict


@keras.saving.register_keras_serializable(package="stepcovnet.onset_ar")
class ArEventOnsetF1Metric(keras.metrics.Metric):
    """Micro event F1 for teacher-fed pointer+residual decode."""

    def __init__(
        self,
        tolerance_sec: float = matching.DEFAULT_TOLERANCE_SEC,
        name: str = "event_onset_f1",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.tolerance_sec = tolerance_sec
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(
        self,
        pred_times: tf.Tensor,
        pred_mask: tf.Tensor,
        gt_times: tf.Tensor,
        gt_mask: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        _ = sample_weight
        tp, fp, fn = tf.numpy_function(
            _ar_event_onset_counts_numpy_wrapper,
            [
                pred_times,
                pred_mask,
                gt_times,
                gt_mask,
                np.array([self.tolerance_sec], dtype=np.float64),
            ],
            [tf.float64, tf.float64, tf.float64],
        )
        tp = tf.reshape(tp, [])
        fp = tf.reshape(fp, [])
        fn = tf.reshape(fn, [])
        self.true_positives.assign_add(tf.cast(tp, self.dtype))
        self.false_positives.assign_add(tf.cast(fp, self.dtype))
        self.false_negatives.assign_add(tf.cast(fn, self.dtype))

    def result(self) -> tf.Tensor:
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return 2.0 * precision * recall / (precision + recall + 1e-9)

    def reset_state(self) -> None:
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"tolerance_sec": self.tolerance_sec})
        return config_dict


class ArOnsetTrainingModel(keras.Model):
    """Custom train/eval loop for teacher-forced AR onset training."""

    def __init__(
        self,
        base_model: keras.Model,
        *,
        experiment_config: config.ArExperimentConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.experiment_config = experiment_config
        run_config = experiment_config.run
        model_config = experiment_config.model
        self.patch_frames = model_config.patch_frames
        self.hop_sec = experiment_config.dataset.hop_sec
        self.lambda_time_final = run_config.lambda_time
        self.lambda_time_ramp_epochs = run_config.lambda_time_ramp_epochs
        self.lambda_time = lambda_time_for_epoch(
            -1,
            lambda_time_final=self.lambda_time_final,
            ramp_epochs=self.lambda_time_ramp_epochs,
        )
        self.length_normalize_ce = run_config.length_normalize_ce
        self.use_soft_pointer_time = run_config.use_soft_pointer_time
        self.lambda_residual = run_config.lambda_residual
        self.lambda_incremental_consistency = run_config.lambda_incremental_consistency
        self.incremental_consistency_max_steps = (
            run_config.incremental_consistency_max_steps
        )
        self.pointer_loss_weight = run_config.pointer_loss_weight
        self.scheduled_sampling_max_p = run_config.scheduled_sampling_max_p
        self.scheduled_sampling_ramp_epochs = run_config.scheduled_sampling_ramp_epochs
        self.scheduled_sampling_warmup_epochs = (
            run_config.scheduled_sampling_warmup_epochs
        )
        # A tf.Variable, not a float: train_step is traced once (during warmup,
        # while p is still 0), so a Python value would freeze the sampling
        # branch out of the graph for the whole run.
        self.scheduled_sampling_p = tf.Variable(
            scheduled_sampling_for_epoch(
                -1,
                max_p=self.scheduled_sampling_max_p,
                ramp_epochs=self.scheduled_sampling_ramp_epochs,
                warmup_epochs=self.scheduled_sampling_warmup_epochs,
            ),
            trainable=False,
            dtype=tf.float32,
            name="scheduled_sampling_p",
        )
        self.max_decoder_len = experiment_config.max_decoder_len()
        self.tolerance_sec = run_config.tolerance_sec
        self.experiment_config = experiment_config
        self.token_class_weights = self._build_token_class_weights(experiment_config)
        self._infer_encoder: keras.Model | None = None
        self._infer_decoder: keras.Model | None = None
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.token_loss_tracker = keras.metrics.Mean(name="token_loss")
        self.pointer_loss_tracker = keras.metrics.Mean(name="pointer_loss")
        self.time_loss_tracker = keras.metrics.Mean(name="time_loss")
        self.residual_loss_tracker = keras.metrics.Mean(name="residual_loss")
        self.incremental_consistency_loss_tracker: keras.metrics.Mean | None
        if self.lambda_incremental_consistency > 0.0:
            self.incremental_consistency_loss_tracker = keras.metrics.Mean(
                name="incremental_consistency_loss",
            )
        else:
            self.incremental_consistency_loss_tracker = None
        self.token_accuracy = keras.metrics.Mean(name="token_accuracy")
        self.use_ordered_onset_gate = run_config.overfit_one_song
        self.event_f1_metric = ArEventOnsetF1Metric(
            tolerance_sec=run_config.tolerance_sec,
            name=mn.AUX_F1_HUNGARIAN,
        )
        self.ordered_match_metric: ArOrderedOnsetMatchMetric | None
        if self.use_ordered_onset_gate:
            self.ordered_match_metric = ArOrderedOnsetMatchMetric(
                tolerance_sec=run_config.tolerance_sec,
                name=mn.TIMING_MATCH_TEACHER,
            )
        else:
            self.ordered_match_metric = None

    @property
    def metrics(self):
        """Metrics updated each train/val step."""
        return self._batch_metrics()

    def _batch_metrics(self) -> list[keras.metrics.Metric]:
        """Metrics updated each train/val step."""
        tracked: list[keras.metrics.Metric] = [
            self.loss_tracker,
            self.token_loss_tracker,
            self.pointer_loss_tracker,
            self.time_loss_tracker,
            self.residual_loss_tracker,
        ]
        if self.incremental_consistency_loss_tracker is not None:
            tracked.append(self.incremental_consistency_loss_tracker)
        tracked.extend(
            [
                self.token_accuracy,
                self.event_f1_metric,
            ],
        )
        if self.ordered_match_metric is not None:
            tracked.append(self.ordered_match_metric)
        return tracked

    def _metric_results(
        self, metrics: list[keras.metrics.Metric]
    ) -> dict[str, tf.Tensor]:
        return {metric.name: metric.result() for metric in metrics}

    @staticmethod
    def _build_token_class_weights(
        experiment_config: config.ArExperimentConfig,
    ) -> tf.Tensor | None:
        """Precompute tide/overfit token CE weights from the single training batch."""
        scheme = experiment_config.run.token_class_weight
        if scheme == "none":
            return None
        batch_np = datasets.sample_to_training_batch(
            datasets.load_overfit_sample(experiment_config),
            experiment_config,
        )
        weights = losses.build_token_class_weights_numpy(
            batch_np["decoder_target_ids"][0],
            batch_np["decoder_mask"][0],
            vocab_size=experiment_config.build_vocab().vocab_size,
            scheme=scheme,
            eos_token_weight_scale=experiment_config.run.eos_token_weight_scale,
        )
        if weights is None:
            return None
        return tf.constant(weights, dtype=tf.float32)

    @staticmethod
    def _unpack_batch(data) -> dict[str, tf.Tensor]:
        if isinstance(data, tuple):
            batch = data[0]
            if isinstance(batch, tuple):
                batch = batch[0]
        else:
            batch = data
        return batch

    def _model_inputs(self, batch: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        return {
            "mert_patches": batch["mert_patches"],
            "patch_mask": batch["patch_mask"],
            "decoder_input_ids": batch["decoder_input_ids"],
            "decoder_mask": batch["decoder_mask"],
        }

    def _ensure_infer_models(self) -> tuple[keras.Model, keras.Model]:
        if self._infer_encoder is None or self._infer_decoder is None:
            encoder, decoder = models.build_ar_onset_inference_models(
                self.base_model,
                self.experiment_config,
            )
            self._infer_encoder = encoder
            self._infer_decoder = decoder
        return self._infer_encoder, self._infer_decoder

    def _forward_parallel_infer(
        self,
        batch: dict[str, tf.Tensor],
        *,
        training: bool,
        decoder_input_ids: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        encoder, decoder = self._ensure_infer_models()
        memory = encoder(
            {
                "mert_patches": batch["mert_patches"],
                "patch_mask": batch["patch_mask"],
            },
            training=training,
        )
        dec_in = (
            decoder_input_ids
            if decoder_input_ids is not None
            else batch["decoder_input_ids"]
        )
        outputs = decoder(
            {
                "encoder_memory": memory,
                "patch_mask": batch["patch_mask"],
                "decoder_input_ids": dec_in,
                "decoder_mask": batch["decoder_mask"],
            },
            training=training,
        )
        return memory, outputs

    def _incremental_consistency_term(
        self,
        parallel_outputs: dict[str, tf.Tensor],
        batch: dict[str, tf.Tensor],
        *,
        encoder_memory: tf.Tensor,
    ) -> tf.Tensor:
        _, decoder = self._ensure_infer_models()
        parallel_times = losses.predicted_times_from_outputs(
            parallel_outputs["pointer_logits"],
            parallel_outputs["residual_sec"],
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            use_soft_expected=self.use_soft_pointer_time,
        )
        n_samples = self.incremental_consistency_max_steps
        if n_samples <= 0:
            n_samples = 1
        return losses.sampled_incremental_consistency_loss_tf(
            decoder,
            encoder_memory,
            batch["patch_mask"],
            batch["decoder_input_ids"],
            batch["decoder_mask"],
            parallel_times,
            batch["onset_step_mask"],
            max_decoder_len=self.max_decoder_len,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            use_soft_pointer_time=self.use_soft_pointer_time,
            n_samples=n_samples,
        )

    def _update_incremental_consistency_metric(
        self,
        batch: dict[str, tf.Tensor],
        *,
        training: bool,
    ) -> None:
        """Log incremental-consistency loss without affecting teacher-fed val metrics."""
        tracker = self.incremental_consistency_loss_tracker
        if tracker is None:
            return
        memory, parallel_outputs = self._forward_parallel_infer(
            batch,
            training=training,
        )
        inc_loss = self._incremental_consistency_term(
            parallel_outputs,
            batch,
            encoder_memory=memory,
        )
        tracker.update_state(inc_loss)

    def _forward_and_loss(
        self,
        batch: dict[str, tf.Tensor],
        *,
        training: bool,
        decoder_input_ids: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        # Incremental consistency needs split encoder/decoder only while training.
        # Validation and offline debug use the full ``base_model`` forward path.
        use_incremental = self.lambda_incremental_consistency > 0.0 and training
        if use_incremental:
            memory, outputs = self._forward_parallel_infer(
                batch,
                training=training,
                decoder_input_ids=decoder_input_ids,
            )
        else:
            model_inputs = self._model_inputs(batch)
            if decoder_input_ids is not None:
                model_inputs = {
                    **model_inputs,
                    "decoder_input_ids": decoder_input_ids,
                }
            outputs = self.base_model(model_inputs, training=training)
            memory = None
        total_loss, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            lambda_time=self.lambda_time,
            lambda_residual=self.lambda_residual,
            pointer_loss_weight=self.pointer_loss_weight,
            length_normalize_ce=self.length_normalize_ce,
            token_class_weights=self.token_class_weights,
            use_soft_pointer_time=self.use_soft_pointer_time,
        )
        if use_incremental:
            assert memory is not None
            inc_loss = self._incremental_consistency_term(
                outputs,
                batch,
                encoder_memory=memory,
            )
            total_loss = (
                total_loss
                + tf.cast(
                    self.lambda_incremental_consistency,
                    tf.float32,
                )
                * inc_loss
            )
            parts = {**parts, "incremental_consistency_loss": inc_loss}
        return total_loss, parts, outputs

    def _update_teacher_fed_f1(
        self,
        outputs: dict[str, tf.Tensor],
        batch: dict[str, tf.Tensor],
    ) -> None:
        pred_times, pred_mask = inference.decode_teacher_fed_times_tf(
            outputs,
            batch,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
            use_soft_expected=self.use_soft_pointer_time,
        )
        self.event_f1_metric.update_state(
            pred_times,
            pred_mask,
            batch["gt_times"],
            batch["gt_mask"],
        )
        if self.ordered_match_metric is None:
            return
        n_matched, n_gt, n_pred = tf.numpy_function(
            timing_match.timing_match_teacher_wrapper,
            [
                pred_times,
                batch["target_times"],
                batch["onset_step_mask"],
                np.array([self.tolerance_sec], dtype=np.float64),
            ],
            [tf.float64, tf.float64, tf.float64],
        )
        self.ordered_match_metric.n_matched.assign_add(
            tf.cast(tf.reshape(n_matched, []), self.ordered_match_metric.dtype),
        )
        self.ordered_match_metric.n_gt.assign_add(
            tf.cast(tf.reshape(n_gt, []), self.ordered_match_metric.dtype),
        )
        self.ordered_match_metric.n_pred.assign_add(
            tf.cast(tf.reshape(n_pred, []), self.ordered_match_metric.dtype),
        )

    def _reset_metrics(self) -> None:
        for metric in self._batch_metrics():
            metric.reset_state()

    def _scheduled_sampled_decoder_inputs(
        self,
        batch: dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Replace teacher tokens with the model's own predictions at rate ``p``."""
        probe_outputs = self.base_model(self._model_inputs(batch), training=True)
        mixed_inputs = inference.build_scheduled_decoder_inputs(
            batch["decoder_input_ids"],
            tf.stop_gradient(probe_outputs["token_logits"]),
            batch["decoder_mask"],
            self.scheduled_sampling_p,
        )
        return tf.cast(mixed_inputs, batch["decoder_input_ids"].dtype)

    def _decoder_inputs_for_step(self, batch: dict[str, tf.Tensor]) -> tf.Tensor:
        """Choose teacher-forced or scheduled-sampled decoder inputs at run time.

        The branch must stay in the graph. A Python ``if`` on
        ``scheduled_sampling_p`` is resolved when Keras traces ``train_step``,
        which happens while the ramp is still in warmup at ``p = 0``, and the
        sampling path is then dropped for every later epoch.
        """
        return tf.cond(
            self.scheduled_sampling_p > 0.0,
            lambda: self._scheduled_sampled_decoder_inputs(batch),
            lambda: batch["decoder_input_ids"],
        )

    def train_step(self, data):
        batch = self._unpack_batch(data)
        decoder_input_ids = self._decoder_inputs_for_step(batch)
        with tf.GradientTape() as tape:
            total_loss, parts, outputs = self._forward_and_loss(
                batch,
                training=True,
                decoder_input_ids=decoder_input_ids,
            )
        trainable_vars = self.base_model.trainable_variables
        if hasattr(self.optimizer, "scale_loss"):
            scaled_loss = self.optimizer.scale_loss(total_loss)
            grads = tape.gradient(scaled_loss, trainable_vars)
        else:
            grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars, strict=False))
        self.loss_tracker.update_state(total_loss)
        self.token_loss_tracker.update_state(parts["token_loss"])
        self.pointer_loss_tracker.update_state(parts["pointer_loss"])
        self.time_loss_tracker.update_state(parts["time_loss"])
        self.residual_loss_tracker.update_state(parts["residual_loss"])
        tracker = self.incremental_consistency_loss_tracker
        if tracker is not None and "incremental_consistency_loss" in parts:
            tracker.update_state(parts["incremental_consistency_loss"])
        self.token_accuracy.update_state(
            losses.masked_token_accuracy(
                outputs["token_logits"],
                batch["decoder_target_ids"],
                batch["decoder_mask"],
            ),
        )
        self._update_teacher_fed_f1(outputs, batch)
        return self._metric_results(self._batch_metrics())

    def test_step(self, data):
        batch = self._unpack_batch(data)
        total_loss, parts, outputs = self._forward_and_loss(batch, training=False)
        self.loss_tracker.update_state(total_loss)
        self.token_loss_tracker.update_state(parts["token_loss"])
        self.pointer_loss_tracker.update_state(parts["pointer_loss"])
        self.time_loss_tracker.update_state(parts["time_loss"])
        self.residual_loss_tracker.update_state(parts["residual_loss"])
        tracker = self.incremental_consistency_loss_tracker
        if tracker is not None:
            if "incremental_consistency_loss" in parts:
                tracker.update_state(parts["incremental_consistency_loss"])
            else:
                self._update_incremental_consistency_metric(batch, training=False)
        self.token_accuracy.update_state(
            losses.masked_token_accuracy(
                outputs["token_logits"],
                batch["decoder_target_ids"],
                batch["decoder_mask"],
            ),
        )
        self._update_teacher_fed_f1(outputs, batch)
        return self._metric_results(self._batch_metrics())


def _save_config(
    experiment_config: config.ArExperimentConfig,
    callback_root_dir: str,
    callback_name: str,
) -> None:
    logdir = pathlib.Path(callback_root_dir) / "logs" / callback_name
    logdir.mkdir(parents=True, exist_ok=True)
    experiment_config.to_json(str(logdir / "config.json"))


def _get_experiment_name(
    experiment_config: config.ArExperimentConfig,
    *,
    n_train_samples: int | None = None,
    n_val_samples: int | None = None,
) -> str:
    """Build TensorBoard / callback run suffix from model + data + schedule.

    Rungs of one ladder share a ``callback_root_dir``, so ``run.run_label``
    carries the rung identity into the timestamped run folder name.
    """
    parts = ["AR_ONSET"]
    run_label = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        experiment_config.run.run_label,
    ).strip("_")
    if run_label:
        parts.append(run_label)
    parts += [
        f"P{experiment_config.model.patch_frames}",
        f"d{experiment_config.model.d_model}",
        f"enc{experiment_config.model.n_enc_layers}",
        f"dec{experiment_config.model.n_dec_layers}",
    ]
    if experiment_config.run.overfit_one_song:
        parts.append("overfit")
    elif n_train_samples is not None and n_val_samples is not None:
        parts.append(f"{n_train_samples}t{n_val_samples}v")
    parts.append(f"ep{experiment_config.run.epochs}")
    if experiment_config.run.early_stopping_patience > 0:
        parts.append(f"es{experiment_config.run.early_stopping_patience}")
    return "-".join(parts)


def lambda_time_for_epoch(
    epoch_index: int,
    *,
    lambda_time_final: float,
    ramp_epochs: int,
) -> float:
    """Linear ramp of ``lambda_time`` from 0 to ``lambda_time_final``."""
    if ramp_epochs <= 0:
        return float(lambda_time_final)
    if lambda_time_final <= 0.0:
        return 0.0
    progress = min(1.0, float(epoch_index + 1) / float(ramp_epochs))
    return float(lambda_time_final) * progress


def should_attach_overfit_gate_callback(
    run_config: config.ArRunConfig,
) -> bool:
    """Overfit gate metrics / early-stop apply only to single-song overfit runs."""
    return bool(run_config.overfit_one_song)


def overfit_gate_score(
    *,
    token_accuracy: float,
    ordered_onset_match: float,
) -> float:
    """Min teacher-fed metrics for checkpointing and early stop."""
    return float(min(token_accuracy, ordered_onset_match))


class EpochTimingCallback(keras.callbacks.Callback):
    """Log per-epoch wall time for throughput ablations."""

    def __init__(self) -> None:
        super().__init__()
        self._epoch_start: float | None = None
        self._epoch_times: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        del epoch, logs
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        del logs
        if self._epoch_start is None:
            return
        elapsed = time.perf_counter() - self._epoch_start
        self._epoch_times.append(elapsed)
        avg = sum(self._epoch_times) / len(self._epoch_times)
        logging.info(
            "Epoch %d wall time: %.2fs (running avg %.2fs over %d epochs)",
            epoch + 1,
            elapsed,
            avg,
            len(self._epoch_times),
        )


class MetricAliasCallback(keras.callbacks.Callback):
    """Publish canonical/legacy metric aliases before monitors read ``logs``.

    ``ModelCheckpoint`` and ``EarlyStopping`` look up one exact key, while
    :func:`onset_metric_names.resolve_checkpoint_metric` may map a config value
    to the other spelling of the same metric. Without both keys present a
    monitor silently no-ops. :class:`OverfitGateCallback` publishes these
    aliases too, but only attaches to single-song overfit runs.
    """

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Mirror canonical and legacy names for every metric in ``logs``.

        Args:
            epoch: Zero-based epoch index (unused).
            logs: Metric mapping Keras passes between callbacks.
        """
        del epoch
        if logs is None:
            return
        mn.publish_legacy_val_aliases(logs)
        mn.publish_legacy_val_aliases(logs, val_prefix=False)


class OverfitGateCallback(keras.callbacks.Callback):
    """Publish teacher-fed overfit gate metrics for checkpointing and early stop.

    Attach only when :func:`should_attach_overfit_gate_callback` is true.
    """

    def __init__(
        self,
        *,
        early_stop: bool = False,
        early_stop_monitor: str | None = None,
        min_score: float = 0.9999,
        patience: int = 3,
    ) -> None:
        super().__init__()
        self.early_stop = early_stop
        self.early_stop_monitor = early_stop_monitor
        self.min_score = float(min_score)
        self.patience = int(patience)
        self._perfect_epochs = 0

    @staticmethod
    def _metric_from_logs(logs: dict, monitor_key: str) -> float:
        """Read a Keras ``val_*`` monitor from logs with canonical/legacy aliases."""
        if monitor_key in logs:
            return float(logs[monitor_key])
        canonical = mn.canonical_metric_name(monitor_key.removeprefix("val_"))
        canonical_key = mn.val_name(canonical)
        if canonical_key in logs:
            return float(logs[canonical_key])
        legacy = mn.CANONICAL_TO_LEGACY_METRIC.get(canonical)
        if legacy is not None:
            legacy_key = mn.val_name(legacy)
            if legacy_key in logs:
                return float(logs[legacy_key])
        return 0.0

    @staticmethod
    def _val_timing_match_teacher(logs: dict) -> float:
        canonical = mn.val_name(mn.TIMING_MATCH_TEACHER)
        legacy = mn.val_name(mn.CANONICAL_TO_LEGACY_METRIC[mn.TIMING_MATCH_TEACHER])
        if canonical in logs:
            return float(logs[canonical])
        if legacy in logs:
            return float(logs[legacy])
        return float(
            logs.get(
                mn.val_name(mn.AUX_F1_HUNGARIAN),
                logs.get(mn.val_name("event_onset_f1"), 0.0),
            ),
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            logs = {}
        token_acc = float(logs.get(mn.val_name(mn.TOKEN_ACCURACY), 0.0))
        ordered_match = self._val_timing_match_teacher(logs)
        gate = overfit_gate_score(
            token_accuracy=token_acc,
            ordered_onset_match=ordered_match,
        )
        logs[mn.val_name(mn.GATE_TEACHER)] = gate
        logs[mn.val_name(mn.CANONICAL_TO_LEGACY_METRIC[mn.GATE_TEACHER])] = gate
        mn.publish_legacy_val_aliases(logs)
        if not self.early_stop:
            return
        monitor_key = self.early_stop_monitor or mn.val_name(
            mn.CANONICAL_TO_LEGACY_METRIC[mn.GATE_TEACHER],
        )
        monitor = self._metric_from_logs(logs, monitor_key)
        if monitor >= self.min_score:
            self._perfect_epochs += 1
        else:
            self._perfect_epochs = 0
        if self._perfect_epochs >= self.patience:
            logging.info(
                "Perfect overfit on %s reached (%.4f >= %.4f for %d epochs); stopping.",
                monitor_key,
                monitor,
                self.min_score,
                self.patience,
            )
            self.model.stop_training = True


def scheduled_sampling_for_epoch(
    epoch_index: int,
    *,
    max_p: float,
    ramp_epochs: int,
    warmup_epochs: int = 0,
) -> float:
    """Linear ramp of scheduled sampling probability from 0 to ``max_p``."""
    if max_p <= 0.0:
        return 0.0
    if epoch_index < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(max_p)
    progress = min(
        1.0,
        float(epoch_index - warmup_epochs + 1) / float(ramp_epochs),
    )
    return float(max_p) * progress


class ScheduledSamplingRampCallback(keras.callbacks.Callback):
    """Update ``ArOnsetTrainingModel.scheduled_sampling_p`` each epoch."""

    def __init__(
        self,
        training_model: ArOnsetTrainingModel,
        *,
        max_p: float,
        ramp_epochs: int,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.training_model = training_model
        self.max_p = float(max_p)
        self.ramp_epochs = int(ramp_epochs)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        _ = logs
        self.training_model.scheduled_sampling_p.assign(
            scheduled_sampling_for_epoch(
                epoch,
                max_p=self.max_p,
                ramp_epochs=self.ramp_epochs,
                warmup_epochs=self.warmup_epochs,
            ),
        )


class LambdaTimeRampCallback(keras.callbacks.Callback):
    """Update ``ArOnsetTrainingModel.lambda_time`` each epoch."""

    def __init__(
        self,
        training_model: ArOnsetTrainingModel,
        *,
        lambda_time_final: float,
        ramp_epochs: int,
    ) -> None:
        super().__init__()
        self.training_model = training_model
        self.lambda_time_final = float(lambda_time_final)
        self.ramp_epochs = int(ramp_epochs)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        _ = logs
        self.training_model.lambda_time = lambda_time_for_epoch(
            epoch,
            lambda_time_final=self.lambda_time_final,
            ramp_epochs=self.ramp_epochs,
        )


def train_ar_onset(
    experiment_config: config.ArExperimentConfig,
    *,
    take_count: int = -1,
    val_take_count: int = -1,
) -> tuple[ArOnsetTrainingModel, keras.callbacks.History]:
    """Train AR onset from overfit or manifest-backed datasets."""
    wsl_gpu.guard_tensorflow_gpu_job()
    run_config = experiment_config.run
    if not run_config.model_output_dir:
        raise ValueError("run.model_output_dir is required")

    configure_ar_gpu_training(run_config)
    reproducibility.apply_training_seed(run_config.seed)
    base_model = models.build_ar_onset_model(experiment_config)
    if run_config.init_model_path:
        init_path = pathlib.Path(run_config.init_model_path)
        if not init_path.is_file():
            raise FileNotFoundError(f"init_model_path not found: {init_path}")
        init_model = keras.models.load_model(str(init_path), compile=False)
        base_model.set_weights(init_model.get_weights())
        logging.info("Loaded AR onset weights from %s", init_path)
    training_model = ArOnsetTrainingModel(
        base_model,
        experiment_config=experiment_config,
    )
    training_model.compile(
        optimizer=build_ar_optimizer(run_config),
        auto_scale_loss=False,
    )

    train_ds, val_ds, n_train_samples, n_val_samples = (
        datasets.create_ar_training_datasets(experiment_config)
    )
    logging.info(
        "AR dataset: %d train samples, %d val samples",
        n_train_samples,
        n_val_samples,
    )
    if take_count != -1:
        train_ds = train_ds.take(take_count)
    if val_take_count != -1:
        val_ds = val_ds.take(val_take_count)
    elif run_config.overfit_one_song:
        val_ds = val_ds.take(1)

    callbacks: list[keras.callbacks.Callback] = []
    monitor_metric = mn.resolve_checkpoint_metric(run_config.checkpoint_metric)
    monitor_mode = "min" if "loss" in monitor_metric else "max"
    if run_config.callback_root_dir:
        experiment_name = _get_experiment_name(
            experiment_config,
            n_train_samples=n_train_samples,
            n_val_samples=n_val_samples,
        )
        tb_callbacks, callback_name = event_trainers._get_callbacks(  # noqa: SLF001
            root_dir=run_config.callback_root_dir,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            experiment_name=experiment_name,
        )
        _save_config(experiment_config, run_config.callback_root_dir, callback_name)
        callbacks.extend(tb_callbacks)

    if run_config.lambda_time_ramp_epochs > 0 and run_config.lambda_time > 0.0:
        callbacks.append(
            LambdaTimeRampCallback(
                training_model,
                lambda_time_final=run_config.lambda_time,
                ramp_epochs=run_config.lambda_time_ramp_epochs,
            ),
        )

    if (
        run_config.scheduled_sampling_ramp_epochs > 0
        and run_config.scheduled_sampling_max_p > 0.0
    ):
        callbacks.append(
            ScheduledSamplingRampCallback(
                training_model,
                max_p=run_config.scheduled_sampling_max_p,
                ramp_epochs=run_config.scheduled_sampling_ramp_epochs,
                warmup_epochs=run_config.scheduled_sampling_warmup_epochs,
            ),
        )

    callbacks.insert(0, EpochTimingCallback())
    callbacks.insert(0, MetricAliasCallback())
    if should_attach_overfit_gate_callback(run_config):
        callbacks.insert(
            2,
            OverfitGateCallback(
                early_stop=run_config.perfect_overfit_early_stop,
                early_stop_monitor=(
                    monitor_metric if run_config.perfect_overfit_early_stop else None
                ),
                min_score=run_config.perfect_overfit_min_score,
                patience=run_config.perfect_overfit_patience,
            ),
        )

    if run_config.early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode=monitor_mode,
                patience=run_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        )

    history = training_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=run_config.epochs,
        callbacks=callbacks,
    )

    event_trainers._write_model(  # noqa: SLF001
        training_model,
        run_config.model_output_dir,
        callback_root_dir=run_config.callback_root_dir,
    )
    logging.info("Finished AR onset training; saved to %s", run_config.model_output_dir)
    return training_model, history
