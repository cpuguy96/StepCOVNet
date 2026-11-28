"""CLI entrypoint: ``python -m stepcovnet.ddc.eval``."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet import runtime
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddc import config, datasets, evaluation, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch DDC C-LSTM placement (M-ddc-20ms)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--split", default=training_index.SPLIT_VAL)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run PyTorch DDC placement evaluation.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    target_device = runtime.resolve_device(args.device)
    experiment = config.PlacementExperimentConfig.from_json(args.config)
    model_dir = pathlib.Path(experiment.run.model_output_dir)
    model_path = args.model_path or str(
        model_dir / trainers.BEST_CHECKPOINT_FILENAME
        if (model_dir / trainers.BEST_CHECKPOINT_FILENAME).is_file()
        else model_dir / f"{experiment.run.model_name}.pt"
    )
    model = trainers.load_placement_model(model_path, device=target_device)
    charts = datasets.load_split_charts(experiment.dataset, args.split)
    report = evaluation.evaluate_placement(
        model, charts, seed=experiment.run.seed, device=target_device
    )
    payload = report.as_dict()
    payload["weights"] = "eval"
    output = pathlib.Path(args.output or str(model_dir / trainers.LAST_EVAL_FILENAME))
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
