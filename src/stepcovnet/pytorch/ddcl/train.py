"""CLI entrypoint: ``python -m stepcovnet.pytorch.ddcl.train``."""

from __future__ import annotations

import argparse

from stepcovnet.ddcl import config
from stepcovnet.pytorch.ddcl import trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PyTorch DDCL ConvLSTM/MERT placement (M-slot48)."
    )
    parser.add_argument("--config", required=True, help="Path to DDCL JSON config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=("fp32", "bf16", "fp16", "auto"),
    )
    parser.add_argument("--fresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PyTorch DDCL placement training.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    experiment = config.DdclExperimentConfig.from_json(args.config)
    if args.epochs is not None:
        experiment.run.epoch = args.epochs
    if args.steps_per_epoch is not None:
        experiment.run.steps_per_epoch = args.steps_per_epoch
    if args.model_output_dir:
        experiment.run.model_output_dir = args.model_output_dir
    if args.fresh:
        experiment.run.resume = False
    trainers.train_placement(
        experiment,
        device=args.device,
        precision=args.precision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
