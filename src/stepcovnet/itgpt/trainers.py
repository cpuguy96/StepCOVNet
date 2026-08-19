"""Train an ITGPT hierarchical placement model (`omalley2026itgpt`)."""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil
from typing import cast

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from stepcovnet.ddcl import evaluation as ddcl_eval
from stepcovnet.itgpt import config, datasets, evaluation, models


def set_seed(seed: int) -> None:
    """Seed NumPy and TensorFlow RNGs.

    Args:
        seed: Reproducibility seed.
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


@keras.saving.register_keras_serializable(package="stepcovnet.itgpt")
def placement_binary_crossentropy(y_true: object, y_pred: object) -> object:
    """Return per-beat mean of ITGPT grid-weighted BCE.

    Keras reduces this ``(batch, beats)`` tensor against the beat mask
    ``sample_weight``. Slot grid weights (16th=2, 24th=1, 32nd=1, micro=0.5)
    are applied inside the loss so they are not collapsed by the default
    last-axis mean.

    Args:
        y_true: Slot labels ``(batch, beats, 48)``.
        y_pred: Slot probabilities ``(batch, beats, 48)``.

    Returns:
        Weighted BCE ``(batch, beats)``.
    """
    epsilon = keras.backend.epsilon()
    y_pred = ops.clip(ops.cast(y_pred, "float32"), epsilon, 1.0 - epsilon)
    y_true = ops.cast(y_true, "float32")
    per_slot = -(y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred))
    weighted = per_slot * models.grid_importance_weights()
    return ops.mean(weighted, axis=-1)


def load_placement_model(path: str | pathlib.Path) -> keras.Model:
    """Load a saved ITGPT placement model, including the custom loss.

    Args:
        path: Path to a ``.keras`` checkpoint.

    Returns:
        Loaded Keras model.
    """
    return keras.models.load_model(
        path,
        custom_objects={
            "placement_binary_crossentropy": placement_binary_crossentropy,
            "ItgptPlacementModel": models.ItgptPlacementModel,
        },
    )


def compile_placement_model(
    model: keras.Model,
    run_config: config.ItgptRunConfig,
) -> keras.Model:
    """Compile with ITGPT AdamW + slot-weighted binary cross-entropy.

    Upstream uses ``BCEWithLogitsLoss`` times grid weights. The Keras port
    emits probabilities; grid weights live in the loss and the beat mask is
    ``sample_weight``.

    Args:
        model: Uncompiled placement model.
        run_config: Optimizer settings.

    Returns:
        The same model, compiled in place.
    """
    optimizer = keras.optimizers.AdamW(
        learning_rate=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
        clipnorm=run_config.clipnorm,
    )
    model.compile(
        optimizer=optimizer,
        loss=placement_binary_crossentropy,
        jit_compile=run_config.jit_compile,
    )
    return model


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by ITGPT placement runs.
        model_name: Run label.

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "itgpt_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


BEST_CHECKPOINT_FILENAME = "best.keras"
LAST_CHECKPOINT_FILENAME = "last.keras"
BACKUP_DIRNAME = "backup"
LAST_EVAL_FILENAME = "eval_val_slot48.json"
BEST_EVAL_FILENAME = "eval_val_slot48_best.json"


def _write_eval_report(
    report: ddcl_eval.Slot48EvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write ``M-slot48`` JSON.

    Args:
        report: Pooled scores.
        path: Destination file.
        weights: ``last`` or ``best``.
    """
    path.write_text(
        json.dumps(evaluation.report_as_dict(report, weights=weights), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation.report_as_dict(report, weights=weights), indent=2))


def train_placement(
    experiment: config.ItgptExperimentConfig,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
) -> keras.Model:
    """Load Dataset B charts, train the transformer, and save weights.

    Args:
        experiment: Placement experiment config.
        extra_callbacks: Optional Keras callbacks (tests).

    Returns:
        Trained Keras model.

    Raises:
        ValueError: If the train or val split is empty.
    """
    set_seed(experiment.run.seed)
    if experiment.run.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("ITGPT training: mixed_float16 policy enabled")
    train_charts = datasets.load_split_charts(
        experiment.dataset,
        "train",
    )
    val_charts = datasets.load_split_charts(
        experiment.dataset,
        "val",
    )
    if not train_charts:
        raise ValueError("train split is empty")
    if not val_charts:
        raise ValueError("val split is empty")
    model = build_and_compile(experiment)
    dummy_inputs, _, _ = datasets.pack_chart(
        train_charts[0], max_beats=experiment.dataset.max_beats
    )
    model(dummy_inputs, training=False)
    output_dir = pathlib.Path(experiment.run.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = output_dir / BACKUP_DIRNAME
    if not experiment.run.resume and backup_dir.exists():
        shutil.rmtree(backup_dir)
    best_path = output_dir / BEST_CHECKPOINT_FILENAME
    last_path = output_dir / LAST_CHECKPOINT_FILENAME
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir),
            save_freq="epoch",
            double_checkpoint=True,
            delete_checkpoint=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(last_path),
            save_best_only=False,
            verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]
    if experiment.run.callback_root_dir:
        log_dir = tensorboard_run_log_dir(
            experiment.run.callback_root_dir,
            experiment.run.model_name,
        )
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=0,
                write_graph=False,
                write_images=False,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    model.fit(
        datasets.batch_generator(
            train_charts,
            max_beats=experiment.dataset.max_beats,
            seed=experiment.run.seed,
        ),
        epochs=experiment.run.epoch,
        steps_per_epoch=max(1, experiment.run.steps_per_epoch),
        validation_data=datasets.batch_generator(
            val_charts,
            max_beats=experiment.dataset.max_beats,
            seed=experiment.run.seed + 1,
        ),
        validation_steps=max(1, experiment.run.validation_steps),
        callbacks=callbacks,
        verbose="auto",
        shuffle=False,
    )
    save_path = output_dir / f"{experiment.run.model_name}.keras"
    model.save(save_path)
    print(f"saved {save_path}")
    last_report = evaluation.evaluate_slot48(
        cast(ddcl_eval.SlotPredictor, model),
        val_charts,
        seed=experiment.run.seed,
        max_beats=experiment.dataset.max_beats,
    )
    _write_eval_report(last_report, output_dir / LAST_EVAL_FILENAME, weights="last")
    if best_path.is_file():
        best_model = load_placement_model(best_path)
        best_report = evaluation.evaluate_slot48(
            cast(ddcl_eval.SlotPredictor, best_model),
            val_charts,
            seed=experiment.run.seed,
            max_beats=experiment.dataset.max_beats,
        )
        _write_eval_report(best_report, output_dir / BEST_EVAL_FILENAME, weights="best")
        print(f"saved {best_path}")
    return model


def build_and_compile(experiment: config.ItgptExperimentConfig) -> keras.Model:
    """Build and compile the placement model.

    Args:
        experiment: Experiment config.

    Returns:
        Compiled Keras model.
    """
    model = models.build_itgpt_placement_model(
        d_model=experiment.model.d_model,
        n_heads=experiment.model.n_heads,
        n_enc_layers=experiment.model.n_enc_layers,
        cnn_hidden=experiment.model.cnn_hidden,
        dropout_rate=experiment.model.dropout_rate,
        max_beats=experiment.dataset.max_beats,
        model_name=experiment.run.model_name,
    )
    return compile_placement_model(model, experiment.run)
