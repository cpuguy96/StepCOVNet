r"""Script for training the arrow detection model.

Usage:
    # Using config file:
    python scripts/train_arrow.py --config=configs/arrow_baseline.json

    # Using command-line arguments (backward compatible):
    python scripts/train_arrow.py --train_data_dir=/path/to/train/data --val_data_dir=/path/to/val/data --epochs=20 --callback_root_dir=/path/to/callbacks --model_output_dir=/path/to/save/model

    # Override config file values:
    python scripts/train_arrow.py --config=configs/arrow_baseline.json --epochs=30 --batch_size=4
"""

import argparse
import json

import tensorflow as tf

from stepcovnet import config
from stepcovnet import trainers

PARSER = argparse.ArgumentParser(description="Train arrow detection model.")
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
    "--num_layers",
    type=int,
    help="Number of Transformer encoder layers.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--d_model",
    type=int,
    help="Model dimensionality.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--num_heads",
    type=int,
    help="Number of attention heads.",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--ff_dim",
    type=int,
    help="Feed-forward network dimension.",
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
    help="Number of batches to use from the training dataset (-1 for entire dataset).",
    default=None,
    required=False,
)
PARSER.add_argument(
    "--val_take_count",
    type=int,
    help="Number of batches to use from the validation dataset (-1 for entire dataset).",
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
        experiment_config = config.ArrowExperimentConfig.from_json(ARGS.config)
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
        dataset_config = config.ArrowDatasetConfig(
            data_dir=ARGS.train_data_dir,
            val_data_dir=ARGS.val_data_dir,
            batch_size=1,
        )
        model_config = config.ArrowModelConfig()
        # Default to a single batch for quick, backward-compatible testing.
        # Users can still override this via --take_count or in the config file.
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

    if ARGS.num_layers is not None:
        model_config.num_layers = ARGS.num_layers
    if ARGS.d_model is not None:
        model_config.d_model = ARGS.d_model
    if ARGS.num_heads is not None:
        model_config.num_heads = ARGS.num_heads
    if ARGS.ff_dim is not None:
        model_config.ff_dim = ARGS.ff_dim
    if ARGS.dropout_rate is not None:
        model_config.dropout_rate = ARGS.dropout_rate

    if ARGS.epochs is not None:
        run_config.epoch = ARGS.epochs
    if ARGS.take_count is not None:
        run_config.take_count = ARGS.take_count
    if ARGS.val_take_count is not None:
        run_config.val_take_count = ARGS.val_take_count
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

    trainers.run_arrow_train_from_config(dataset_config, model_config, run_config)


if __name__ == "__main__":
    main()
