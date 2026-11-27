#!/usr/bin/env python3
"""Print or export a ranked scoreboard of ITGPT onset placement runs.

Usage:
    python scripts/itgpt_scoreboard.py --filter fraxtil_orig
    python scripts/itgpt_scoreboard.py --filter 300t_50v --format json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import typing

from stepcovnet.itgpt import scoreboard

REPO = pathlib.Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the ITGPT scoreboard tool.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Rank local/Colab ITGPT placement runs by M-slot48 & diagnostics.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(scoreboard.DEFAULT_SEARCH_ROOTS),
        help=(
            "Run parent directories to scan (default: experiments/colab models_wsl/ddc)."
        ),
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Case-insensitive substring filter on run directory name.",
    )
    parser.add_argument(
        "--weights",
        choices=scoreboard.VALID_WEIGHT_MODES,
        default="auto",
        help="Checkpoint weight file preference (default: auto).",
    )
    parser.add_argument(
        "--sort-by",
        choices=scoreboard.VALID_SORT_KEYS,
        default="f1_at_05",
        help="Ranking column (default: f1_at_05).",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="Run name (or substring) to use as delta/regression baseline.",
    )
    parser.add_argument(
        "--min-beats",
        type=int,
        default=0,
        help="Ignore runs with fewer evaluated beats than this threshold.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Maximum rows to output (0 = all).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "jsonl"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional file path to write output in addition to printing.",
    )
    return parser


def _resolve_roots(raw_roots: typing.Sequence[str]) -> list[pathlib.Path]:
    """Resolve relative root paths against the repository root.

    Args:
        raw_roots: CLI root directory strings.

    Returns:
        List of resolved :class:`pathlib.Path` directories.
    """
    resolved: list[pathlib.Path] = []
    for raw in raw_roots:
        path = pathlib.Path(raw)
        if not path.is_absolute():
            path = REPO / path
        resolved.append(path)
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Scan ITGPT evaluation directories and print a ranked scoreboard.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    roots = _resolve_roots(args.roots)
    rows = scoreboard.scan_runs(
        roots,
        weights=args.weights,
        name_filter=args.filter,
        min_beats=args.min_beats,
    )
    ranked = scoreboard.rank_rows(rows, sort_by=args.sort_by)
    visible = ranked[: args.top] if args.top > 0 else ranked

    if args.format == "json":
        text = json.dumps(
            {
                "plateau": scoreboard.detect_plateau(ranked),
                "runs": [row.as_dict() for row in visible],
            },
            indent=2,
        )
    elif args.format == "jsonl":
        text = "\n".join(json.dumps(row.as_dict()) for row in visible)
    else:
        text = scoreboard.format_markdown_table(
            ranked,
            baseline_name=args.baseline,
            top_k=args.top,
        )

    print(text)
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
