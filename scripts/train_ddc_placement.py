#!/usr/bin/env python
"""Train DDC C-LSTM step placement (`donahue2017ddc`)."""

from __future__ import annotations

import argparse

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/train_ddc_placement.py")

from stepcovnet.ddc import config, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DDC-faithful C-LSTM step placement.",
    )
    parser.add_argument("--config", required=True, help="Path to placement JSON config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--model_output_dir", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run DDC placement training.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    wsl_gpu.guard_tensorflow_gpu_job(__file__)
    experiment = config.PlacementExperimentConfig.from_json(args.config)
    if args.epochs is not None:
        experiment.run.epoch = args.epochs
    if args.batch_size is not None:
        experiment.dataset.batch_size = args.batch_size
    if args.steps_per_epoch is not None:
        experiment.run.steps_per_epoch = args.steps_per_epoch
    if args.model_output_dir:
        experiment.run.model_output_dir = args.model_output_dir
    trainers.train_placement(experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
