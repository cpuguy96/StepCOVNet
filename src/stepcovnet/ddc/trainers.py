"""Train a DDC C-LSTM placement model."""

from __future__ import annotations

import datetime
import json
import pathlib
from collections.abc import Iterator

import keras
import numpy as np
import tensorflow as tf

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddc import config, datasets, evaluation, models


def set_seed(seed: int) -> None:
    """Seed NumPy, Python, and TensorFlow RNGs.

    Args:
        seed: Reproducibility seed.
    """
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def compile_placement_model(
    model: keras.Model,
    run_config: config.PlacementRunConfig,
) -> keras.Model:
    """Compile with paper SGD + binary cross-entropy.

    Args:
        model: Uncompiled C-LSTM.
        run_config: Optimizer settings.

    Returns:
        The same model, compiled in place.
    """
    optimizer = keras.optimizers.SGD(
        learning_rate=run_config.learning_rate,
        clipnorm=run_config.clipnorm,
    )
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model


def _batch_generator(
    charts: list[datasets.PlacementChart],
    *,
    batch_size: int,
    nunroll: int,
    seed: int,
) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
    """Infinite generator of truncated-BPTT batches.

    Args:
        charts: Loaded train charts.
        batch_size: Sequences per batch.
        nunroll: Window length in frames.
        seed: RNG seed.

    Yields:
        tuple[dict[str, np.ndarray], np.ndarray]: ``(inputs, target)`` pairs for
        ``model.fit``.
    """
    rng = np.random.default_rng(seed)
    while True:
        yield datasets.sample_train_batch(
            charts,
            batch_size=batch_size,
            nunroll=nunroll,
            rng=rng,
        )


def tensorboard_run_log_dir(callback_root_dir: str, model_name: str) -> pathlib.Path:
    """Return a per-run TensorBoard directory under the stage ``logs/`` tree.

    Args:
        callback_root_dir: Stage root shared by DDC placement runs.
        model_name: Run label (appended after a timestamp).

    Returns:
        ``{callback_root_dir}/logs/{YYYYMMDD-HHMMSS}-{model_name}``.
    """
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = str(model_name).strip() or "ddc_placement"
    log_dir = pathlib.Path(callback_root_dir) / "logs" / f"{stamp}-{safe_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


BEST_CHECKPOINT_FILENAME = "best.keras"
LAST_EVAL_FILENAME = "eval_val_ddc.json"
BEST_EVAL_FILENAME = "eval_val_ddc_best.json"


def _write_eval_report(
    report: evaluation.PlacementEvalReport,
    path: pathlib.Path,
    *,
    weights: str,
) -> None:
    """Write a placement eval JSON tagged with which weights were scored.

    Args:
        report: Val F-score report.
        path: Output JSON path.
        weights: ``last`` or ``best``.
    """
    payload = report.as_dict()
    payload["weights"] = weights
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def train_placement(
    experiment: config.PlacementExperimentConfig,
) -> keras.Model:
    """Load Dataset A charts, train the C-LSTM, and save weights.

    Args:
        experiment: Placement experiment config.

    Returns:
        Trained Keras model.
    """
    set_seed(experiment.run.seed)
    train_charts = datasets.load_split_charts(
        experiment.dataset,
        training_index.SPLIT_TRAIN,
    )
    val_charts = datasets.load_split_charts(
        experiment.dataset,
        training_index.SPLIT_VAL,
    )
    model = compile_placement_model(
        models.build_clstm_placement_model(
            lstm_units=experiment.model.lstm_units,
            lstm_layers=experiment.model.lstm_layers,
            dropout_rate=experiment.model.dropout_rate,
            dnn_sizes=tuple(experiment.model.dnn_sizes),
            model_name=experiment.run.model_name,
        ),
        experiment.run,
    )
    output_dir = pathlib.Path(experiment.run.model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / BEST_CHECKPOINT_FILENAME
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
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
                write_images=False,
            )
        )
    steps = experiment.run.steps_per_epoch
    val_steps = experiment.run.validation_steps
    model.fit(
        _batch_generator(
            train_charts,
            batch_size=experiment.dataset.batch_size,
            nunroll=experiment.dataset.nunroll,
            seed=experiment.run.seed,
        ),
        epochs=experiment.run.epoch,
        steps_per_epoch=max(1, steps),
        validation_data=_batch_generator(
            val_charts,
            batch_size=experiment.dataset.batch_size,
            nunroll=experiment.dataset.nunroll,
            seed=experiment.run.seed + 1,
        ),
        validation_steps=max(1, val_steps),
        callbacks=callbacks,
        verbose="auto",
        shuffle=False,
    )
    save_path = output_dir / f"{experiment.run.model_name}.keras"
    model.save(save_path)
    print(f"saved {save_path}")
    last_report = evaluation.evaluate_placement(
        model, val_charts, seed=experiment.run.seed
    )
    _write_eval_report(
        last_report,
        output_dir / LAST_EVAL_FILENAME,
        weights="last",
    )
    if best_path.is_file():
        best_model = keras.models.load_model(best_path)
        best_report = evaluation.evaluate_placement(
            best_model, val_charts, seed=experiment.run.seed
        )
        _write_eval_report(
            best_report,
            output_dir / BEST_EVAL_FILENAME,
            weights="best",
        )
        print(f"saved {best_path}")
    return model
