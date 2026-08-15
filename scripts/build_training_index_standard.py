#!/usr/bin/env python
"""Write a standard-difficulty (no edit) training_index beside Dataset A."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from stepcovnet.dataset_prep import training_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a training_index.json to standard DDR difficulties "
            "(beginner/easy/medium/hard/challenge), dropping edit charts."
        ),
    )
    parser.add_argument(
        "--source",
        default="data/literature_fraxtil_orig/training_index.json",
        help="Full training_index.json to filter",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path (default: {source_dir}/training_index_standard.json)",
    )
    parser.add_argument(
        "--policy-tag",
        default=training_index.STANDARD_POLICY_TAG,
        help="Suffix recorded in split_policy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output manifest",
    )
    return parser


def resolve_output_path(source_path: pathlib.Path, output: str) -> pathlib.Path:
    """Return the standard-index path for a source manifest.

    Args:
        source_path: Path to the full ``training_index.json``.
        output: Optional explicit output path; empty uses the default name.

    Returns:
        Destination path for the filtered manifest.
    """
    if output:
        return pathlib.Path(output)
    return source_path.with_name(training_index.STANDARD_INDEX_FILENAME)


def main(argv: list[str] | None = None) -> int:
    """Run the standard-difficulty index filter CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    source_path = pathlib.Path(args.source)
    if not source_path.is_file():
        print(f"error: source manifest not found: {args.source}", file=sys.stderr)
        return 1
    output_path = resolve_output_path(source_path, args.output)
    if output_path.is_file() and not args.overwrite:
        print(
            f"error: {output_path} exists; pass --overwrite to replace",
            file=sys.stderr,
        )
        return 1
    try:
        filtered = training_index.filter_training_index(
            source_path,
            policy_tag=args.policy_tag,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    saved = training_index.save_training_index(filtered, output_path)
    summary = {
        "source": str(source_path),
        "path": str(saved),
        "split_policy": filtered.split_policy,
        "counts": training_index._index_to_dict(filtered)["counts"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
