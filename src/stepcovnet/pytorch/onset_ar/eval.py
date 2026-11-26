"""CLI entrypoint: ``python -m stepcovnet.pytorch.onset_ar.eval``."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.onset_ar import config, datasets, inference, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch AR onset model.")
    parser.add_argument(
        "--model",
        dest="model_path",
        required=True,
        help="Path to saved .pt checkpoint.",
    )
    parser.add_argument("--config", required=True, help="Path to AR config JSON.")
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON evaluation report.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and evaluate a PyTorch AR onset checkpoint.

    Args:
        argv: Optional argument list.
    """
    args = _build_parser().parse_args(argv)
    device = runtime.resolve_device(args.device)
    exp = config.ArExperimentConfig.from_json(args.config)
    model = trainers.load_ar_onset_model(args.model_path, device=device)
    samples = datasets.load_split_samples(exp, args.split)
    report = inference.evaluate_ar_samples(model, samples, device=device)
    text = json.dumps(report.as_dict(), indent=2)
    print(text)
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
