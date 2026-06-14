"""Post-hoc sweep of VAL_ONSET_F1_SCORE callback checkpoints at a fixed threshold."""

import argparse
import glob
import os
import sys

import tensorflow as tf

from stepcovnet import config
from stepcovnet import dense_overfit_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Onset experiment config JSON")
    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob for checkpoint paths (e.g. callbacks/.../VAL_ONSET_F1_SCORE*.keras)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Peak-pick confidence threshold (default 0.25)",
    )
    args = parser.parse_args(argv)

    experiment = config.OnsetExperimentConfig.from_json(args.config)
    paths = sorted(glob.glob(args.pattern), key=os.path.getmtime)
    if not paths:
        print(f"No checkpoints matched: {args.pattern}", file=sys.stderr)
        return 1

    best_tag = ""
    best_micro = 0.0
    for path in paths:
        model = tf.keras.models.load_model(path, compile=False)
        report = dense_overfit_eval.eval_dense_val_event_f1(
            model,
            experiment.dataset,
            experiment.model,
            confidence_threshold=args.threshold,
        )
        micro = float(report["micro_event_f1"])
        tag = os.path.basename(path)
        print(f"{micro:.4f}  {tag}", flush=True)
        if micro > best_micro:
            best_tag = tag
            best_micro = micro

    print(f"BEST {best_micro:.4f}  {best_tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
