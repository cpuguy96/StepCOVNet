r"""Train autoregressive onset detection (``gate-tide-overfit`` and follow-on gates).

Phase 0: ``--verify-only`` checks tide assets and loads one AR sample batch.
Phase 1+: full encoder-decoder training (not implemented yet).

Usage:
    python scripts/train_onset_ar.py --config configs/onset_ar_tide.json --verify-only

    # Future gate-tide-overfit training (Phase 1):
    python scripts/train_onset_ar.py --config configs/onset_ar_tide.json \
        --model_output_dir models/ar_tide_overfit
"""

from __future__ import annotations

import argparse
import json
import sys

from stepcovnet.onset_ar import config, datasets

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
    help="Check tide assets and load one AR sample; exit without training.",
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
    "--model_output_dir",
    type=str,
    default=None,
    help="Override run.model_output_dir.",
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
    if args.model_output_dir:
        run_config.model_output_dir = args.model_output_dir


def main() -> None:
    args = PARSER.parse_args()
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    _apply_overrides(experiment_config, args)

    if args.verify_only:
        summary, _sample = datasets.verify_config_loads_one_batch(experiment_config)
        print(json.dumps(summary, indent=2))
        return

    if not experiment_config.run.model_output_dir:
        PARSER.error(
            "AR training is not implemented yet (Phase 1). "
            "Use --verify-only to smoke-test Phase 0 assets and data loading, "
            "or set run.model_output_dir once the trainer lands.",
        )

    print(
        "AR encoder-decoder training is Phase 1 (gate-tide-overfit). "
        "Phase 0 verify passed if you ran --verify-only.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
