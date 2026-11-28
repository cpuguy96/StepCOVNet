#!/usr/bin/env python
"""Evaluate PyTorch DDCL placement ``M-slot48`` F1 at 0.5 and max-F1."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/eval_ddcl_placement.py")

from stepcovnet import runtime
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, datasets, evaluation, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DDCL ConvLSTM/MERT placement: M-slot48 F1.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--split",
        default=training_index.SPLIT_VAL,
        help="Manifest split to score (default: val)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--diagnose_timing",
        action="store_true",
        help=(
            "Score counterfactual slot masks and bipartite event F1 alongside "
            "timing_match."
        ),
    )
    mode.add_argument(
        "--diagnose_grounding",
        action="store_true",
        help=(
            "Run none/shuffle/zeros beat-audio corruption grounding ablation "
            "while keeping [meter, bpm] intact."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Slot occupancy cutoff for diagnostic modes (default: 0.5).",
    )
    parser.add_argument("--device", default="auto")
    return parser


def _resolve_model_path(
    experiment: config.DdclExperimentConfig, model_path: str
) -> str:
    """Return an explicit checkpoint path or the saved PyTorch artifact."""
    if model_path:
        return model_path
    model_dir = pathlib.Path(experiment.run.model_output_dir)
    named = model_dir / f"{experiment.run.model_name}.pt"
    preferred = (
        model_dir / trainers.BEST_SLOT_F1_CHECKPOINT_FILENAME,
        model_dir / trainers.BEST_CHECKPOINT_FILENAME,
        named,
        model_dir / trainers.LAST_CHECKPOINT_FILENAME,
    )
    for candidate in preferred:
        if candidate.is_file():
            return str(candidate)
    pt_files = sorted(path.name for path in model_dir.glob("*.pt"))
    if len(pt_files) != 1:
        raise FileNotFoundError(
            f"expected best.pt, {named.name}, last.pt, or one .pt in "
            f"{model_dir}, found {pt_files!r}"
        )
    return str(model_dir / pt_files[0])


def main(argv: list[str] | None = None) -> int:
    """Run DDCL placement evaluation."""
    args = _build_parser().parse_args(argv)
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    device = runtime.resolve_device(args.device)
    experiment = config.DdclExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment, args.model_path)
    if args.diagnose_grounding:
        default_filename = "eval_val_grounding.json"
    elif args.diagnose_timing:
        default_filename = "eval_val_timing_diag.json"
    else:
        default_filename = "eval_val_slot48.json"
    output_path = args.output or str(
        pathlib.Path(experiment.run.model_output_dir) / default_filename
    )
    model = trainers.load_placement_model(model_path, device=device)
    charts = datasets.load_split_charts(experiment.dataset, args.split)
    if args.diagnose_grounding:
        grounding_report = evaluation.evaluate_audio_grounding(
            model,
            charts,
            threshold=args.threshold,
            seed=experiment.run.seed,
        )
        payload = grounding_report.as_dict()
        payload["model_path"] = model_path
        payload["config_path"] = args.config
    elif args.diagnose_timing:
        diag_report = evaluation.diagnose_decoded_timing(
            model,
            charts,
            threshold=args.threshold,
            seed=experiment.run.seed,
        )
        payload = diag_report.as_dict()
        payload["model_path"] = model_path
        payload["config_path"] = args.config
    else:
        report = evaluation.evaluate_slot48(model, charts, seed=experiment.run.seed)
        payload = report.as_dict()
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
