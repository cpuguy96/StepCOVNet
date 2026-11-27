"""CLI entrypoint: ``python -m stepcovnet.pytorch.itgpt.eval``."""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet.dataset_prep import training_index
from stepcovnet.itgpt import config, diagnostics
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import datasets, evaluation, trainers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PyTorch ITGPT placement: M-slot48 F1, grounding, "
            "and error breakdown."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--split",
        default=training_index.SPLIT_VAL,
        help="Manifest split to score (default: val)",
    )
    parser.add_argument(
        "--diagnose_grounding",
        action="store_true",
        help=(
            "Run none/shuffle/zeros beat-audio corruption grounding ablation "
            "while keeping scalar bpm/difficulty intact."
        ),
    )
    parser.add_argument(
        "--error_breakdown",
        action="store_true",
        help=(
            "Compute structured error breakdown (subdivisions, timing offsets, "
            "packs, meters, beat phase, song sections, calibration, worst charts)."
        ),
    )
    parser.add_argument(
        "--breakdown_output",
        default="",
        help=(
            "Optional JSON path for --error_breakdown (defaults to "
            "eval_val_breakdown.json next to --output or in model_output_dir)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Slot occupancy cutoff for diagnostic mode (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device ('auto', 'cuda', 'mps', 'cpu').",
    )
    return parser


def _resolve_model_path(
    experiment: config.ItgptExperimentConfig,
    model_path: str,
) -> str:
    """Return an explicit checkpoint path or the saved PyTorch artifact.

    Args:
        experiment: Placement experiment config.
        model_path: Optional override path.

    Returns:
        Path to a ``.pt`` file.

    Raises:
        FileNotFoundError: If no unique checkpoint exists.
    """
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
            f"expected best_slot_f1.pt, best.pt, {named.name}, last.pt, or one "
            f".pt in {model_dir}, found {pt_files!r}"
        )
    return str(model_dir / pt_files[0])


def main(argv: list[str] | None = None) -> int:
    """Run PyTorch ITGPT placement evaluation.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    target_device = runtime.resolve_device(args.device)
    experiment = config.ItgptExperimentConfig.from_json(args.config)
    model_path = _resolve_model_path(experiment, args.model_path)
    model = trainers.load_placement_model(model_path, device=target_device)
    charts = datasets.load_split_charts(experiment.dataset, args.split)
    if args.diagnose_grounding or args.error_breakdown:
        none_preds = [
            evaluation.predict_chart_slots(
                model,
                chart,
                max_beats=experiment.dataset.max_beats,
                device=target_device,
            )
            for chart in charts
        ]
    else:
        none_preds = None

    if args.diagnose_grounding:
        grounding_report = evaluation.evaluate_audio_grounding(
            model,
            charts,
            threshold=args.threshold,
            max_beats=experiment.dataset.max_beats,
            seed=experiment.run.seed,
            device=target_device,
            none_predictions=none_preds,
        )
        payload = grounding_report.as_dict()
        payload["model_path"] = model_path
        payload["config_path"] = args.config
        payload["framework"] = "pytorch"
        default_filename = "eval_val_grounding.json"
        output = args.output or str(
            pathlib.Path(experiment.run.model_output_dir) / default_filename
        )
        output_file = pathlib.Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2)
        output_file.write_text(text + "\n", encoding="utf-8")
        print(text)
    elif not args.error_breakdown:
        report = evaluation.evaluate_slot48(
            model,
            charts,
            seed=experiment.run.seed,
            max_beats=experiment.dataset.max_beats,
            device=target_device,
        )
        payload = evaluation.report_as_dict(report, weights="eval")
        default_filename = "eval_val_slot48.json"
        output = args.output or str(
            pathlib.Path(experiment.run.model_output_dir) / default_filename
        )
        output_file = pathlib.Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2)
        output_file.write_text(text + "\n", encoding="utf-8")
        print(text)

    if args.error_breakdown:
        assert none_preds is not None
        breakdown_report = diagnostics.error_breakdown_from_predictions(
            charts,
            none_preds,
            threshold=args.threshold,
        )
        b_payload = breakdown_report.as_dict()
        b_payload["model_path"] = model_path
        b_payload["config_path"] = args.config
        b_payload["framework"] = "pytorch"
        if args.breakdown_output:
            breakdown_path = pathlib.Path(args.breakdown_output)
        elif args.output and not args.diagnose_grounding:
            breakdown_path = pathlib.Path(args.output)
        elif args.output and args.diagnose_grounding:
            breakdown_path = (
                pathlib.Path(args.output).parent / "eval_val_breakdown.json"
            )
        else:
            breakdown_path = (
                pathlib.Path(experiment.run.model_output_dir)
                / "eval_val_breakdown.json"
            )
        breakdown_path.parent.mkdir(parents=True, exist_ok=True)
        breakdown_path.write_text(
            json.dumps(b_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(breakdown_report.format_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
