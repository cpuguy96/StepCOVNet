r"""Script for training the arrow detection model.

Usage:
    python scripts/train_arrow.py --train_data_dir=/path/to/train/data --val_data_dir=/path/to/val/data --epochs=20 --callback_root_dir=/path/to/callbacks --model_output_file=/path/to/save/model.keras
"""

import argparse

import tensorflow as tf

from stepcovnet import trainers

PARSER = argparse.ArgumentParser(description="Train arrow detection model.")
PARSER.add_argument(
    "--train_data_dir",
    type=str,
    help="Directory containing training data.",
    default=None,
    required=True,
)
PARSER.add_argument(
    "--val_data_dir",
    type=str,
    help="Directory containing validation data.",
    default=None,
    required=True,
)
PARSER.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs to train for.",
    default=10,
    required=False,
)
PARSER.add_argument(
    "--callback_root_dir",
    type=str,
    help="Root directory for storing training callbacks (checkpoints, logs).",
    default="",
    required=False,
)
PARSER.add_argument(
    "--model_output_dir",
    type=str,
    help="Directory where the trained model will be saved.",
    default=None,
    required=True,
)
PARSER.add_argument(
    "--take_count",
    type=int,
    help="Number of batches to use from the training dataset.",
    default=1,
    required=False,
)
PARSER.add_argument(
    "--model_name",
    type=str,
    default="",
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
    batch_size = 1
    normalize = True

    trainers.run_arrow_train(
        data_dir=ARGS.train_data_dir,
        val_data_dir=ARGS.val_data_dir,
        batch_size=batch_size,
        normalize=normalize,
        model_params={},
        take_count=ARGS.take_count,
        epoch=ARGS.epochs,
        callback_root_dir=ARGS.callback_root_dir,
        model_output_dir=ARGS.model_output_dir,
        model_name=ARGS.model_name,
    )


if __name__ == "__main__":
    main()
