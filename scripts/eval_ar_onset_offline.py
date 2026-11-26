"""Offline PyTorch evaluation for AR onset checkpoints (free-run KV decode + metrics)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from stepcovnet import wsl_gpu

SCRIPT_REL = "scripts/eval_ar_onset_offline.py"
wsl_gpu.bootstrap_gpu_script(SCRIPT_REL)

from stepcovnet.onset_ar import config, datasets, inference, trainers
from stepcovnet.pytorch import runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline PyTorch AR onset checkpoint evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ar/tide_overfit.json",
        help="AR experiment config JSON.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="PyTorch checkpoint path (default: run.model_output_dir/best.pt or ar_onset_model.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("overfit", "train", "val"),
        default="overfit",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of songs to evaluate (-1 = all).",
    )
    parser.add_argument(
        "--ar_decode",
        action="store_true",
        help="Included for CLI compatibility (PyTorch eval always runs KV-cached AR decode).",
    )
    parser.add_argument(
        "--ar_decode_eos_logit_bias",
        type=float,
        default=0.0,
        help="Additive bias applied to the <EOS> logit during free-run decode.",
    )
    parser.add_argument(
        "--ar_decode_min_onset_tokens",
        type=int,
        default=0,
        help="Suppress <EOS> until at least this many onset tokens have been emitted.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device.")
    parser.add_argument(
        "--output", type=str, default="", help="Optional output JSON path."
    )
    return parser


def _resolve_model_path(
    experiment_config: config.ArExperimentConfig,
    model_path: str,
) -> pathlib.Path:
    if model_path:
        return pathlib.Path(model_path)
    out_dir = pathlib.Path(experiment_config.run.model_output_dir)
    for name in ("best.pt", "ar_onset_model.pt", "last.pt"):
        cand = out_dir / name
        if cand.is_file():
            return cand
    return out_dir / "ar_onset_model.pt"


def main(argv: list[str] | None = None) -> int:
    """Run PyTorch AR onset offline evaluation and print JSON report."""
    args = _build_parser().parse_args(argv)
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    device = runtime.resolve_device(args.device)
    exp = config.ArExperimentConfig.from_json(args.config)
    ckpt_path = _resolve_model_path(exp, args.model_path)
    if not ckpt_path.is_file():
        print(f"model not found: {ckpt_path}", file=sys.stderr)
        return 1

    model = trainers.load_ar_onset_model(ckpt_path, device=device)
    split = "train" if args.split == "overfit" else args.split
    samples = datasets.load_split_samples(exp, split)
    if args.split == "overfit":
        samples = samples[:1]
    elif args.limit > 0:
        samples = samples[: args.limit]

    length_control = inference.ArLengthControl(
        eos_logit_bias=args.ar_decode_eos_logit_bias,
        min_onset_tokens=args.ar_decode_min_onset_tokens,
    )
    report = inference.evaluate_ar_samples(
        model,
        samples,
        length_control=length_control,
        device=device,
    )
    payload = report.as_dict()
    payload["model_path"] = str(ckpt_path)
    payload["config_path"] = args.config
    payload["split"] = args.split
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
