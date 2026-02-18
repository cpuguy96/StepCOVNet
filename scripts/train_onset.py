r"""Script for training the onset detection model.

Usage:
    python scripts/train_onset.py --train_data_dir=/path/to/train/data --val_data_dir=/path/to/val/data --epochs=20 --callback_root_dir=/path/to/callbacks --model_output_file=/path/to/save/model.keras
"""

import argparse

from stepcovnet import trainers

PARSER = argparse.ArgumentParser(description="Train onset detection model.")
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
    default=None,
    required=True,
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

ARGS = PARSER.parse_args()


def main():
    apply_temporal_augment = True
    should_apply_spec_augment = True
    use_gaussian_target = True
    gaussian_sigma = 0.1
    batch_size = 1
    normalize = True

    trainers.run_train(
        data_dir=ARGS.train_data_dir,
        val_data_dir=ARGS.val_data_dir,
        batch_size=batch_size,
        normalize=normalize,
        apply_temporal_augment=apply_temporal_augment,
        should_apply_spec_augment=should_apply_spec_augment,
        use_gaussian_target=use_gaussian_target,
        gaussian_sigma=gaussian_sigma,
        model_params={},
        take_count=ARGS.take_count,
        epoch=ARGS.epochs,
        callback_root_dir=ARGS.callback_root_dir,
        model_output_dir=ARGS.model_output_dir,
    )


if __name__ == "__main__":
    main()
