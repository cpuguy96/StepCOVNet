"""CLI entrypoint: ``python -m stepcovnet.pytorch.onset_ar.train``."""

from __future__ import annotations

import argparse

from stepcovnet.pytorch.onset_ar import config, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PyTorch AR onset model.")
    parser.add_argument("--config", required=True, help="Path to AR config JSON.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default=None,
        help="Override checkpoint output directory.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=("fp32", "bf16", "fp16", "auto"),
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--no_eval_val",
        action="store_true",
        help="Skip free-run validation decode after training.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch PyTorch AR onset training.

    Args:
        argv: Optional argument list.
    """
    args = _build_parser().parse_args(argv)
    exp = config.ArExperimentConfig.from_json(args.config)
    if args.epochs is not None:
        exp.run.epochs = int(args.epochs)
    if args.model_output_dir is not None:
        exp.run.model_output_dir = str(args.model_output_dir)
    trainers.train_ar_onset(
        exp,
        device=args.device,
        precision=args.precision,
        eval_val=not args.no_eval_val,
    )


if __name__ == "__main__":
    main()
