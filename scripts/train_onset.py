r"""Script for training the onset detection model.

Usage:
    # Using config file:
    python scripts/train_onset.py --config=configs/onset_baseline.json

    # Using command-line arguments (backward compatible):
    python scripts/train_onset.py --train_data_dir=/path/to/train/data --val_data_dir=/path/to/val/data --epochs=20 --callback_root_dir=/path/to/callbacks --model_output_dir=/path/to/save/model

    # Override config file values:
    python scripts/train_onset.py --config=configs/onset_baseline.json --epochs=30 --batch_size=4
"""

import argparse
import json

import tensorflow as tf

from stepcovnet import config
from stepcovnet import trainers

PARSER = argparse.ArgumentParser(description="Train onset detection model.")
PARSER.add_argument(
    "--config",
    type=str,
    help="Path to JSON config file. If provided, other arguments override config values.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--train_data_dir",
    type=str,
    help="Directory containing training data.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--val_data_dir",
    type=str,
    help="Directory containing validation data.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs to train for.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for training.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--normalize",
    action="store_true",
    help="Normalize spectrograms.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--no_normalize",
    action="store_false",
    dest="normalize",
    help="Don't normalize spectrograms.",
    required=False,
)
PARSER.add_argument(
    "--apply_temporal_augment",
    action="store_true",
    help="Apply temporal augmentation.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--apply_spec_augment",
    action="store_true",
    help="Apply spectrogram augmentation.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--use_gaussian_target",
    action="store_true",
    help="Use Gaussian targets instead of binary.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--gaussian_sigma",
    type=float,
    help="Standard deviation for Gaussian targets.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--initial_filters",
    type=int,
    help="Number of initial filters in U-Net.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--depth",
    type=int,
    help="Depth of U-Net (number of levels).",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--dropout_rate",
    type=float,
    help="Dropout rate.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--callback_root_dir",
    type=str,
    help="Root directory for storing training callbacks (checkpoints, logs).",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--model_output_dir",
    type=str,
    help="Directory where the trained model will be saved.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--take_count",
    type=int,
    help="Number of batches to use from the training dataset.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--model_name",
    type=str,
    default=None,
    required=False,
)
ARGS = PARSER.parse_args()

if tf.config.list_physical_devices("GPU"):
    import keras

    print("Training with GPU.")

    keras.mixed_precision.set_global_policy(
        keras.mixed_precision.Policy("mixed_float16")
    )

    # Enable XLA (Accelerated Linear Algebra) for TensorFlow, which can improve
    # performance by compiling TensorFlow graphs into highly optimized
    # machine code.
    tf.config.optimizer.set_jit("autoclustering")


def main():
    # Load config from file if provided
    if ARGS.config:
        experiment_config = config.OnsetExperimentConfig.from_json(ARGS.config)
        dataset_config = experiment_config.dataset
        model_config = experiment_config.model
        run_config = experiment_config.run
    else:
        # Use defaults if no config file
        if (
            not ARGS.train_data_dir
            or not ARGS.val_data_dir
            or not ARGS.model_output_dir
        ):
            PARSER.error(
                "Either --config must be provided, or --train_data_dir, --val_data_dir, "
                "and --model_output_dir must be provided."
            )
        dataset_config = config.OnsetDatasetConfig(
            data_dir=ARGS.train_data_dir,
            val_data_dir=ARGS.val_data_dir,
            batch_size=1,
            normalize=True,
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
            use_gaussian_target=False,
            gaussian_sigma=1.0,
        )
        model_config = config.OnsetModelConfig()
        run_config = config.RunConfig(
            epoch=10,
            take_count=1,
            model_output_dir=ARGS.model_output_dir,
            callback_root_dir="",
            model_name="",
        )

    # Override with command-line arguments
    if ARGS.train_data_dir:
        dataset_config.data_dir = ARGS.train_data_dir
    if ARGS.val_data_dir:
        dataset_config.val_data_dir = ARGS.val_data_dir
    if ARGS.batch_size is not None:
        dataset_config.batch_size = ARGS.batch_size
    if ARGS.normalize is not None:
        dataset_config.normalize = ARGS.normalize
    if ARGS.apply_temporal_augment is not None:
        dataset_config.apply_temporal_augment = ARGS.apply_temporal_augment
    if ARGS.apply_spec_augment is not None:
        dataset_config.should_apply_spec_augment = ARGS.apply_spec_augment
    if ARGS.use_gaussian_target is not None:
        dataset_config.use_gaussian_target = ARGS.use_gaussian_target
    if ARGS.gaussian_sigma is not None:
        dataset_config.gaussian_sigma = ARGS.gaussian_sigma

    if ARGS.initial_filters is not None:
        model_config.initial_filters = ARGS.initial_filters
    if ARGS.depth is not None:
        model_config.depth = ARGS.depth
    if ARGS.dropout_rate is not None:
        model_config.dropout_rate = ARGS.dropout_rate

    if ARGS.epochs is not None:
        run_config.epoch = ARGS.epochs
    if ARGS.take_count is not None:
        run_config.take_count = ARGS.take_count
    if ARGS.model_output_dir:
        run_config.model_output_dir = ARGS.model_output_dir
    if ARGS.callback_root_dir is not None:
        run_config.callback_root_dir = ARGS.callback_root_dir
    if ARGS.model_name is not None:
        run_config.model_name = ARGS.model_name

    # Validate required fields
    if not dataset_config.data_dir or not dataset_config.val_data_dir:
        PARSER.error("--train_data_dir and --val_data_dir are required")
    if not run_config.model_output_dir:
        PARSER.error("--model_output_dir is required")

    trainers.run_train_from_config(dataset_config, model_config, run_config)


if __name__ == "__main__":
    main()
