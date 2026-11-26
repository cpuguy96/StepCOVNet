"""CLI entrypoint: ``python -m stepcovnet.pytorch.ddcl.eval``."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, datasets
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.ddcl import evaluation, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch DDCL placement (M-slot48)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--split", default=training_index.SPLIT_VAL)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PyTorch DDCL placement evaluation.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    target_device = runtime.resolve_device(args.device)
    experiment = config.DdclExperimentConfig.from_json(args.config)
    model_dir = pathlib.Path(experiment.run.model_output_dir)
    if args.model_path:
        model_path = args.model_path
    else:
        preferred = (
            model_dir / trainers.BEST_SLOT_F1_CHECKPOINT_FILENAME,
            model_dir / trainers.BEST_CHECKPOINT_FILENAME,
            model_dir / f"{experiment.run.model_name}.pt",
            model_dir / trainers.LAST_CHECKPOINT_FILENAME,
        )
        model_path = next(
            (str(p) for p in preferred if p.is_file()),
            str(model_dir / trainers.BEST_CHECKPOINT_FILENAME),
        )
    model = trainers.load_placement_model(model_path, device=target_device)
    charts = datasets.load_split_charts(experiment.dataset, args.split)
    report = evaluation.evaluate_slot48(
        model, charts, seed=experiment.run.seed, device=target_device
    )
    payload = evaluation.report_as_dict(report, weights="eval")
    output = pathlib.Path(args.output or str(model_dir / trainers.LAST_EVAL_FILENAME))
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
