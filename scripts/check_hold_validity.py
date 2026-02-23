r"""Check training chart files for hold validity (same rules as ChartValidityMetric).

Reports any chart that has hold violations: a 3 (hold end) with no preceding 2 (hold start)
in that column, or a 3 immediately after a 1 (tap) in the same column.
Use --show_examples to print example locations (step, time, arrow, line number) for lookup in the .txt file.

Usage:
    python scripts/check_hold_validity.py --data_dir=/path/to/training/data
    python scripts/check_hold_validity.py --data_dir=/path/to/training/data --show_examples
"""

import argparse
import pathlib
import sys

from stepcovnet import datasets
from stepcovnet import generator
from stepcovnet import metrics

# Chart .txt has 4 header lines (TITLE, BPM, NOTES, difficulty), then one line per step (1-based).
_CHART_HEADER_LINES = 4

PARSER = argparse.ArgumentParser(
    description="Check training data charts for hold validity violations."
)
PARSER.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing chart (.txt) and audio files (same layout as arrow training data).",
)
PARSER.add_argument(
    "--show_examples",
    action="store_true",
    help="Print example violation locations (step, time, arrow, line number) for lookup in each chart file.",
)


def main() -> int:
    args = PARSER.parse_args()
    data_dir = args.data_dir
    show_examples = args.show_examples
    pairs = datasets._load_and_pair_files(data_dir)  # noqa: SLF001
    if not pairs:
        print(f"No audio-chart pairs found in {data_dir!r}.", file=sys.stderr)
        return 2
    # (song_name, violations, hold_ends, chart_path, times, cols, examples or None)
    songs_in_violation: list[
        tuple[str, int, int, str, list | None, list | None, list | None]
    ] = []
    for _audio_path, chart_path in pairs:
        try:
            times, cols = datasets._parse_step_chart(  # noqa: SLF001
                chart_path, binary_timings=False
            )
        except Exception as e:
            print(f"Failed to parse {chart_path!r}: {e}", file=sys.stderr)
            songs_in_violation.append(
                (pathlib.Path(chart_path).stem, -1, -1, chart_path, None, None, None)
            )
            continue
        violations, hold_ends, examples = metrics.compute_hold_validity_violations(cols)
        if violations > 0:
            song_name = pathlib.Path(chart_path).stem
            songs_in_violation.append(
                (
                    song_name,
                    violations,
                    hold_ends,
                    chart_path,
                    times.tolist(),
                    cols.tolist(),
                    examples,
                )
            )
    if songs_in_violation:
        print(f"Found {len(songs_in_violation)} songs in violation.")
        print("Hold validity violations found in the following chart(s):")
        for entry in songs_in_violation:
            name, violations, hold_ends, chart_path, times, cols, examples = entry
            if violations == -1:
                print(f"  - {name!s} (parse error)")
            else:
                print(
                    f"  - {name!s} ({violations} violation(s), {hold_ends} hold end(s))"
                )
                if (
                    show_examples
                    and examples
                    and times is not None
                    and cols is not None
                ):
                    print(f"    Chart: {chart_path!s}")
                    for step_idx, column, kind in examples:
                        line_num = _CHART_HEADER_LINES + step_idx + 1
                        time_s = times[step_idx]
                        arrow_str = generator._int_to_base4_string(  # noqa: SLF001
                            int(cols[step_idx]), min_digits=4
                        )
                        kind_desc = (
                            "hold end (3) with no preceding hold start (2)"
                            if kind == "unmatched_3"
                            else "hold end (3) immediately after tap (1)"
                        )
                        print(
                            f"    Example: step {step_idx}, time {time_s:.4f}s, "
                            f"arrow {arrow_str!r}, column {column} — {kind_desc} "
                            f"(line {line_num} in file)"
                        )
        return 1
    print("All charts passed hold validity check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
