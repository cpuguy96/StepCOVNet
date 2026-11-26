r"""Train autoregressive onset detection (``gate-tide-overfit`` and follow-on gates).

Usage:
    python scripts/train_onset_ar.py --config configs/ar/tide_overfit.json --verify-only

    python scripts/train_onset_ar.py --config configs/ar/tide_overfit.json
"""

from __future__ import annotations

import argparse
import json

from stepcovnet import wsl_gpu

SCRIPT_REL = "scripts/train_onset_ar.py"

wsl_gpu.bootstrap_gpu_script(SCRIPT_REL)

from stepcovnet.onset_ar import config, datasets, trainers

PARSER = argparse.ArgumentParser(description="Train AR onset detection model.")
PARSER.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to JSON config file.",
)
PARSER.add_argument(
    "--verify-only",
    action="store_true",
    help="Check assets and build train/val batches; exit without training.",
)
PARSER.add_argument(
    "--training_index_path",
    type=str,
    default=None,
    help="Override dataset.training_index_path (manifest JSON).",
)
PARSER.add_argument(
    "--overfit_audio_path",
    type=str,
    default=None,
    help="Override dataset.overfit_audio_path.",
)
PARSER.add_argument(
    "--overfit_chart_path",
    type=str,
    default=None,
    help="Override dataset.overfit_chart_path.",
)
PARSER.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override run.epochs.",
)
PARSER.add_argument(
    "--model_output_dir",
    type=str,
    default=None,
    help="Override run.model_output_dir.",
)
PARSER.add_argument(
    "--callback_root_dir",
    type=str,
    default=None,
    help="Override run.callback_root_dir.",
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


def _apply_overrides(
    experiment_config: config.ArExperimentConfig,
    args: argparse.Namespace,
) -> None:
    dataset_config = experiment_config.dataset
    run_config = experiment_config.run
    if args.training_index_path:
        dataset_config.training_index_path = args.training_index_path
    if args.overfit_audio_path is not None:
        dataset_config.overfit_audio_path = args.overfit_audio_path
    if args.overfit_chart_path is not None:
        dataset_config.overfit_chart_path = args.overfit_chart_path
    if args.epochs is not None:
        run_config.epochs = args.epochs
    if args.model_output_dir:
        run_config.model_output_dir = args.model_output_dir
    if args.callback_root_dir is not None:
        run_config.callback_root_dir = args.callback_root_dir


def main() -> None:
    args = PARSER.parse_args()
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    _apply_overrides(experiment_config, args)

    if args.verify_only:
        samples = datasets.load_split_samples(experiment_config, "train")
        batch = datasets.pack_ar_sample(samples[0])
        summary = {
            "num_train_samples": len(samples),
            "first_chart_path": samples[0].chart_path,
            "mert_patches_shape": list(batch["mert_patches"].shape),
            "decoder_input_ids_shape": list(batch["decoder_input_ids"].shape),
        }
        print(json.dumps(summary, indent=2))
        return

    if not experiment_config.run.model_output_dir:
        PARSER.error("--model_output_dir is required (config or CLI)")

    trainers.train_ar_onset(experiment_config)


if __name__ == "__main__":
    main()
