#!/usr/bin/env python
"""Sample a fixed-size train/val subset from an existing ``training_index.json``."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from stepcovnet.dataset_prep import constants, training_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write a smaller training_index.json with fixed train/val row counts "
            "sampled from the full P8 manifest."
        ),
    )
    parser.add_argument(
        "--source",
        default=str(
            training_index.training_index_path(constants.DEFAULT_OUTPUT_DIR),
        ),
        help="Full training_index.json to sample from",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output manifest path (default: "
            "{output_dir}/training_index_{policy_tag}_{train}t_{val}v.json)"
        ),
    )
    parser.add_argument(
        "--policy-tag",
        default=training_index.SUBSET_POLICY_LADDER_V1,
        help="Tag recorded in split_policy and used in the default filename",
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=50,
        help="Number of train chart rows to include",
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=100,
        help="Number of val chart rows to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output manifest",
    )
    return parser


def default_subset_path(
    output_dir: str,
    train_rows: int,
    val_rows: int,
    policy_tag: str = training_index.SUBSET_POLICY_LADDER_V1,
) -> str:
    """Return the default subset manifest path under ``output_dir``.

    Args:
        output_dir: Preprocess output root recorded in the source manifest.
        train_rows: Number of train chart rows in the subset.
        val_rows: Number of val chart rows in the subset.
        policy_tag: Tag recorded in ``split_policy``; also names the file.

    Returns:
        Path to ``{output_dir}/training_index_{policy_tag}_{train}t_{val}v.json``.
    """
    name = f"training_index_{policy_tag}_{train_rows}t_{val_rows}v.json"
    return str(training_index.training_index_path(output_dir).with_name(name))


def main(argv: list[str] | None = None) -> int:
    """Run build_training_index_subset CLI."""
    args = _build_parser().parse_args(argv)
    source_path = pathlib.Path(args.source).resolve()
    if not source_path.is_file():
        default_source = training_index.training_index_path(args.source)
        if default_source.is_file():
            source_path = default_source.resolve()
        else:
            print(f"error: source manifest not found: {args.source}", file=sys.stderr)
            return 1

    try:
        subset = training_index.build_training_index_subset(
            source_path,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            seed=args.seed,
            policy_tag=args.policy_tag,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output = args.output or default_subset_path(
        subset.output_dir,
        args.train_rows,
        args.val_rows,
        args.policy_tag,
    )
    output_path = pathlib.Path(output)
    if output_path.is_file() and not args.overwrite:
        print(
            f"error: {output_path} exists; pass --overwrite to replace",
            file=sys.stderr,
        )
        return 1

    saved = training_index.save_training_index(subset, output_path)
    train_audio = training_index.unique_audio_relpaths(
        [
            entry
            for entry in subset.entries
            if entry.split == training_index.SPLIT_TRAIN
        ],
    )
    val_audio = training_index.unique_audio_relpaths(
        [entry for entry in subset.entries if entry.split == training_index.SPLIT_VAL],
    )
    all_audio = training_index.unique_audio_relpaths(subset.entries)

    summary = {
        "source": str(source_path),
        "source_sha256": subset.source_sha256,
        "path": str(saved),
        "split_policy": subset.split_policy,
        "split_seed": subset.split_seed,
        "counts": training_index._index_to_dict(subset)["counts"],
        "unique_audio": {
            "train": len(train_audio),
            "val": len(val_audio),
            "total": len(all_audio),
        },
        "mert_files_needed": len(all_audio),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
