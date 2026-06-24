#!/usr/bin/env python
"""Build ``training_index.json`` train/val manifest for prepared ``final_data``."""

from __future__ import annotations

import argparse
import json
import sys

from stepcovnet.dataset_prep import constants, training_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign stratified song-level train/val splits and write "
            "training_index.json under the preprocess output root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=constants.DEFAULT_OUTPUT_DIR,
        help="Prepared output root (e.g. data/final_data)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of songs per bundle assigned to validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed for per-bundle shuffles",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing training_index.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run build_training_index CLI."""
    args = _build_parser().parse_args(argv)
    out_path = training_index.training_index_path(args.output_dir)
    if out_path.is_file() and not args.overwrite:
        print(
            f"error: {out_path} exists; pass --overwrite to replace",
            file=sys.stderr,
        )
        return 1

    try:
        index = training_index.build_training_index(
            args.output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        saved = training_index.save_training_index(index)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "output_dir": index.output_dir,
        "path": str(saved),
        "split_policy": index.split_policy,
        "val_fraction": index.val_fraction,
        "split_seed": index.split_seed,
        "counts": training_index._index_to_dict(index)["counts"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
