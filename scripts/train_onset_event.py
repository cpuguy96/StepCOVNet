r"""Script for training the event-based onset detection model.

Usage:
    python scripts/train_onset_event.py --config=configs/event/audio_baseline.json

    # Smoke run (script-only take_count / epoch overrides):
    python scripts/train_onset_event.py --config=configs/event/audio_baseline.json \
        --take_count=1 --epochs=2 --model_output_dir=models/onset_event_smoke

    # Overfit tide smoke (no training shortcuts; compares pre-processing frontends):
    python scripts/run_overfit_tide_suite.py --epochs=300 \
        --model_root=models_wsl/overfit_tide

    # Override data and output directories:
    python scripts/train_onset_event.py --config=configs/event/audio_baseline.json \
        --train_data_dir=data/v2/train --val_data_dir=data/v2/val \
        --model_output_dir=models/onset_event --callback_root_dir=callbacks/onset_event
"""

import argparse

import tensorflow as tf

from stepcovnet.onset_events import config, trainers

PARSER = argparse.ArgumentParser(description="Train event-based onset detection model.")
PARSER.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to JSON config file.",
)
PARSER.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    help="Override dataset.data_dir.",
)
PARSER.add_argument(
    "--val_data_dir",
    type=str,
    default=None,
    help="Override dataset.val_data_dir.",
)
PARSER.add_argument(
    "--training_index_path",
    type=str,
    default=None,
    help="Override dataset.training_index_path (manifest JSON).",
)
PARSER.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override run.epochs.",
)
PARSER.add_argument(
    "--take_count",
    type=int,
    default=None,
    help="Training batches per epoch (-1 for all). Script-only; not stored in config.",
)
PARSER.add_argument(
    "--val_take_count",
    type=int,
    default=None,
    help="Validation batches per epoch (-1 for all). Script-only; not stored in config.",
)
PARSER.add_argument(
    "--callback_root_dir",
    type=str,
    default=None,
    help="Override run.callback_root_dir.",
)
PARSER.add_argument(
    "--model_output_dir",
    type=str,
    default=None,
    help="Override run.model_output_dir.",
)
PARSER.add_argument(
    "--overfit_one_song",
    action="store_true",
    help="Train and validate on the first valid pair under dataset.data_dir only.",
)
PARSER.add_argument(
    "--overfit_audio_path",
    type=str,
    default=None,
    help="Override dataset.overfit_audio_path (requires --overfit_chart_path).",
)
PARSER.add_argument(
    "--overfit_chart_path",
    type=str,
    default=None,
    help="Override dataset.overfit_chart_path (requires --overfit_audio_path).",
)
ARGS = PARSER.parse_args()

if tf.config.list_physical_devices("GPU"):
    print(
        "Training with GPU (float32). Event onset uses a custom train_step; "
        "mixed_float16 is disabled until loss scaling is wired for it."
    )
    tf.config.optimizer.set_jit("autoclustering")


def main() -> None:
    experiment_config = config.OnsetEventExperimentConfig.from_json(ARGS.config)
    dataset_config = experiment_config.dataset
    run_config = experiment_config.run

    if ARGS.train_data_dir:
        dataset_config.data_dir = ARGS.train_data_dir
    if ARGS.val_data_dir:
        dataset_config.val_data_dir = ARGS.val_data_dir
    if ARGS.training_index_path:
        dataset_config.training_index_path = ARGS.training_index_path
    if ARGS.epochs is not None:
        run_config.epochs = ARGS.epochs
    if ARGS.model_output_dir:
        run_config.model_output_dir = ARGS.model_output_dir
    if ARGS.callback_root_dir is not None:
        run_config.callback_root_dir = ARGS.callback_root_dir
    if ARGS.overfit_one_song:
        run_config.overfit_one_song = True
    if ARGS.overfit_audio_path is not None:
        dataset_config.overfit_audio_path = ARGS.overfit_audio_path
    if ARGS.overfit_chart_path is not None:
        dataset_config.overfit_chart_path = ARGS.overfit_chart_path

    take_count = ARGS.take_count if ARGS.take_count is not None else -1
    val_take_count = ARGS.val_take_count if ARGS.val_take_count is not None else -1

    if not dataset_config.training_index_path and (
        not dataset_config.data_dir or not dataset_config.val_data_dir
    ):
        PARSER.error(
            "dataset.training_index_path or both dataset.data_dir and "
            "dataset.val_data_dir are required"
        )
    if not run_config.model_output_dir:
        PARSER.error("--model_output_dir is required (config or CLI)")

    trainers.train_onset_event(
        experiment_config,
        take_count=take_count,
        val_take_count=val_take_count,
    )


if __name__ == "__main__":
    main()
