r"""Visualize arrow training data: aggregate stats and plots for a training data directory.

Usage:
    python scripts/visualize_arrow_data.py --data_dir=data/v2/train
    python scripts/visualize_arrow_data.py --data_dir=data/v2/train --output_dir=output/viz --no_show
    python scripts/visualize_arrow_data.py --data_dir=data/v2/train --top_arrows=15
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(
        "matplotlib is required for this script. Install with: pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from stepcovnet import constants
from stepcovnet import datasets
from stepcovnet import generator
from stepcovnet import metrics

# Note-kind labels: 0=empty, 1=single, 2=chord, 3=hold_start, 4=hold_end, 5=hold_both
NOTE_KIND_LABELS = [
    "Empty",
    "Single",
    "Chord",
    "Hold start",
    "Hold end",
    "Hold both",
]
COLUMN_LABELS = ["Left", "Down", "Up", "Right"]


def _build_note_kind_lookup() -> np.ndarray:
    """Build length-256 array mapping arrow code (0..255) to note kind (0..5).

    Same logic as metrics._build_arrow_note_kind_table().
    """
    table = np.zeros(constants.N_ARROW_TYPES, dtype=np.int32)
    for n in range(constants.N_ARROW_TYPES):
        d0 = (n // 64) % 4
        d1 = (n // 16) % 4
        d2 = (n // 4) % 4
        d3 = n % 4
        n1 = int(d0 == 1) + int(d1 == 1) + int(d2 == 1) + int(d3 == 1)
        n2 = int(d0 == 2) + int(d1 == 2) + int(d2 == 2) + int(d3 == 2)
        n3 = int(d0 == 3) + int(d1 == 3) + int(d2 == 3) + int(d3 == 3)
        if n == 0:
            kind = 0
        elif n2 >= 1 and n3 >= 1:
            kind = 5
        elif n2 >= 1:
            kind = 3
        elif n3 >= 1:
            kind = 4
        elif n1 == 1:
            kind = 1
        elif n1 >= 2:
            kind = 2
        else:
            kind = 0
        table[n] = kind
    return table


def _parse_bpm(chart_path: str) -> float | None:
    """Read BPM from chart header (second line). Returns None on error."""
    try:
        with open(chart_path, "r") as f:
            f.readline()  # TITLE
            line = f.readline()
        return float(line.removeprefix("BPM").strip())
    except (ValueError, OSError):
        return None


def _column_activity_vectorized(cols: np.ndarray) -> np.ndarray:
    """Return (4,) array: count of steps with activity in each column."""
    cols = np.asarray(cols, dtype=np.int32).ravel()
    n = np.clip(cols, 0, 255)
    d0 = (n // 64) % 4
    d1 = (n // 16) % 4
    d2 = (n // 4) % 4
    d3 = n % 4
    c0 = np.sum((d0 >= 1) & (d0 <= 3))
    c1 = np.sum((d1 >= 1) & (d1 <= 3))
    c2 = np.sum((d2 >= 1) & (d2 <= 3))
    c3 = np.sum((d3 >= 1) & (d3 <= 3))
    return np.array([c0, c1, c2, c3], dtype=np.int64)


def _chord_sizes(cols: np.ndarray) -> np.ndarray:
    """Return 1d array of chord sizes (1-4) for each non-empty step."""
    cols = np.asarray(cols, dtype=np.int32).ravel()
    sizes = []
    for n in cols:
        n = int(np.clip(n, 0, 255))
        if n == 0:
            continue
        d0 = (n // 64) % 4
        d1 = (n // 16) % 4
        d2 = (n // 4) % 4
        d3 = n % 4
        count = sum(1 for d in (d0, d1, d2, d3) if d in (1, 2, 3))
        if count >= 1:
            sizes.append(min(count, 4))
    return np.array(sizes, dtype=np.int32) if sizes else np.array([], dtype=np.int32)


def collect_aggregates(data_dir: str):
    """Load all charts, aggregate arrow codes, note kinds, chart validity, etc."""
    pairs = datasets._load_and_pair_files(data_dir)  # noqa: SLF001
    if not pairs:
        return None

    note_kind_lookup = _build_note_kind_lookup()
    all_arrow_codes: list[int] = []
    all_note_kinds: list[int] = []
    steps_per_chart: list[int] = []
    bpms: list[float] = []
    violation_counts: list[int] = []
    valid_charts = 0
    invalid_charts = 0
    column_activity = np.zeros(4, dtype=np.int64)
    all_chord_sizes: list[int] = []
    chart_durations: list[float] = []
    all_inter_step_intervals: list[float] = []
    steps_per_second: list[float] = []

    for _audio_path, chart_path in pairs:
        try:
            times, cols = datasets._parse_step_chart(  # noqa: SLF001
                chart_path, binary_timings=False
            )
        except Exception:
            continue
        cols_flat = np.asarray(cols, dtype=np.int32).ravel()
        n_steps = len(cols_flat)
        steps_per_chart.append(n_steps)

        if n_steps >= 2:
            duration = abs(float(times[-1] - times[0]))
            chart_durations.append(duration)
            all_inter_step_intervals.extend(np.diff(times).tolist())
            steps_per_second.append(n_steps / duration if duration > 0 else 0.0)
        else:
            chart_durations.append(0.0)
            steps_per_second.append(0.0)

        bpm = _parse_bpm(chart_path)
        if bpm is not None:
            bpms.append(bpm)

        violations, _hold_ends, _examples = metrics.compute_chart_validity_violations(
            cols
        )
        violation_counts.append(violations)
        if violations > 0:
            invalid_charts += 1
        else:
            valid_charts += 1

        all_arrow_codes.extend(cols_flat.tolist())
        kinds = note_kind_lookup[np.clip(cols_flat, 0, 255)]
        all_note_kinds.extend(kinds.tolist())

        column_activity += _column_activity_vectorized(cols_flat)
        chord_sz = _chord_sizes(cols_flat)
        all_chord_sizes.extend(chord_sz.tolist())

    return {
        "n_charts": len(pairs),
        "all_arrow_codes": np.array(all_arrow_codes, dtype=np.int32),
        "all_note_kinds": np.array(all_note_kinds, dtype=np.int32),
        "steps_per_chart": np.array(steps_per_chart, dtype=np.int64),
        "bpms": np.array(bpms, dtype=np.float64) if bpms else np.array([]),
        "violation_counts": np.array(violation_counts, dtype=np.int32),
        "valid_charts": valid_charts,
        "invalid_charts": invalid_charts,
        "column_activity": column_activity,
        "all_chord_sizes": np.array(all_chord_sizes, dtype=np.int32),
        "chart_durations": np.array(chart_durations, dtype=np.float64),
        "inter_step_intervals": np.array(all_inter_step_intervals, dtype=np.float64),
        "steps_per_second": np.array(steps_per_second, dtype=np.float64),
    }


def write_summary(agg: dict, output_path: str | None) -> str:
    """Build summary text and optionally write to file. Returns summary string."""
    n_parsed = agg["valid_charts"] + agg["invalid_charts"]
    total_steps = int(np.sum(agg["steps_per_chart"]))
    spc = agg["steps_per_chart"]
    viol_pct = (
        f"{100 * agg['invalid_charts'] / n_parsed:.1f}%" if n_parsed > 0 else "N/A"
    )
    if n_parsed > 0 and spc.size > 0:
        steps_line = f"Steps per chart: min={int(spc.min())}, mean={float(spc.mean()):.1f}, max={int(spc.max())}"
    else:
        steps_line = "Steps per chart: N/A (no charts successfully parsed)"
    lines = [
        "Arrow training data summary",
        "=" * 40,
        f"Charts (parsed): {n_parsed}",
        f"Total steps: {total_steps}",
        steps_line,
        f"Charts with hold violations: {agg['invalid_charts']} ({viol_pct})",
    ]
    vc = agg["violation_counts"]
    with_viol = vc[vc > 0]
    if len(with_viol) > 0:
        lines.append(
            f"Mean violations per chart (among those with violations): {float(with_viol.mean()):.1f}"
        )

    cd = agg["chart_durations"]
    isi = agg["inter_step_intervals"]
    sps = agg["steps_per_second"]
    if cd.size > 0:
        lines.append(
            f"Chart duration (s): min={float(cd.min()):.1f}, mean={float(cd.mean()):.1f}, max={float(cd.max()):.1f}"
        )
    else:
        lines.append("Chart duration (s): N/A")
    if isi.size > 0:
        lines.append(
            f"Mean step interval (s): {float(isi.mean()):.3f}; "
            f"inter-step min={float(isi.min()):.3f}, max={float(isi.max()):.3f}"
        )
    else:
        lines.append("Mean step interval (s): N/A")
    if sps.size > 0:
        lines.append(
            f"Steps per second: min={float(sps.min()):.2f}, mean={float(sps.mean()):.2f}, max={float(sps.max()):.2f}"
        )
    else:
        lines.append("Steps per second: N/A")

    summary = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(summary)
            f.write("\n")
    return summary


def plot_note_kind(agg: dict, output_dir: str | None, show: bool) -> None:
    """Bar chart of note kind distribution."""
    counts = np.bincount(agg["all_note_kinds"], minlength=len(NOTE_KIND_LABELS))[
        : len(NOTE_KIND_LABELS)
    ]
    fig, ax = plt.subplots()
    ax.bar(NOTE_KIND_LABELS, counts, color="steelblue", edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Note kind distribution")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "note_kind_dist.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_top_arrow_types(
    agg: dict, top_n: int, output_dir: str | None, show: bool
) -> None:
    """Bar chart of top N arrow types (base-4 labels) + Other."""
    codes = agg["all_arrow_codes"]
    codes = codes[codes > 0]  # exclude empty
    if codes.size == 0:
        return
    unique, cnt = np.unique(codes, return_counts=True)
    order = np.argsort(-cnt)
    top_codes = unique[order[:top_n]]
    top_counts = cnt[order[:top_n]]
    other_count = int(cnt[order[top_n:]].sum()) if len(order) > top_n else 0
    labels = [
        generator._int_to_base4_string(int(c), min_digits=4)  # noqa: SLF001
        for c in top_codes
    ]
    if other_count > 0:
        labels.append("Other")
        top_counts = np.append(top_counts, other_count)
    fig, ax = plt.subplots()
    ax.bar(range(len(labels)), top_counts, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {top_n} arrow types")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "top_arrow_types.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_per_column_activity(agg: dict, output_dir: str | None, show: bool) -> None:
    """Bar chart of per-column activity."""
    fig, ax = plt.subplots()
    ax.bar(
        COLUMN_LABELS,
        agg["column_activity"],
        color="steelblue",
        edgecolor="black",
    )
    ax.set_ylabel("Count (steps with activity)")
    ax.set_title("Per-column activity")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "per_column_activity.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_chart_validity(agg: dict, output_dir: str | None, show: bool) -> None:
    """Bar chart: valid vs invalid charts."""
    fig, ax = plt.subplots()
    ax.bar(
        ["Valid", "With violations"],
        [agg["valid_charts"], agg["invalid_charts"]],
        color=["green", "coral"],
        edgecolor="black",
    )
    ax.set_ylabel("Number of charts")
    ax.set_title("Chart validity")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "chart_validity.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_steps_per_chart(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of steps per chart."""
    fig, ax = plt.subplots()
    n_plotted = len(agg["steps_per_chart"])
    ax.hist(
        agg["steps_per_chart"],
        bins=min(50, max(1, n_plotted)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Steps per chart")
    ax.set_ylabel("Number of charts")
    ax.set_title("Steps per chart distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "steps_per_chart.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_chord_size(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of chord sizes (1-4) on non-empty steps."""
    sz = agg["all_chord_sizes"]
    if sz.size == 0:
        return
    fig, ax = plt.subplots()
    ax.hist(
        sz, bins=np.arange(0.5, 5.5), rwidth=0.8, color="steelblue", edgecolor="black"
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Chord size")
    ax.set_ylabel("Count")
    ax.set_title("Chord size (non-empty steps)")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "chord_size.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_bpm(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of BPM (if parsed)."""
    bpms = agg["bpms"]
    if bpms.size == 0:
        return
    fig, ax = plt.subplots()
    ax.hist(bpms, bins=min(40, max(1, len(bpms))), color="steelblue", edgecolor="black")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Number of charts")
    ax.set_title("BPM distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "bpm.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_chart_duration(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of chart duration (seconds)."""
    cd = agg["chart_durations"]
    if cd.size == 0:
        return
    fig, ax = plt.subplots()
    n = len(cd)
    ax.hist(
        cd,
        bins=min(40, max(1, n)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Chart duration (s)")
    ax.set_ylabel("Number of charts")
    ax.set_title("Chart duration distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "chart_duration.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_inter_step_interval(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of inter-step intervals (seconds)."""
    isi = agg["inter_step_intervals"]
    if isi.size == 0:
        return
    fig, ax = plt.subplots()
    n = len(isi)
    ax.hist(
        isi,
        bins=min(50, max(1, n)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Inter-step interval (s)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-step interval distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "inter_step_interval.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_steps_per_second(agg: dict, output_dir: str | None, show: bool) -> None:
    """Histogram of steps per second (per chart)."""
    sps = agg["steps_per_second"]
    if sps.size == 0:
        return
    fig, ax = plt.subplots()
    n = len(sps)
    ax.hist(
        sps,
        bins=min(40, max(1, n)),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Steps per second")
    ax.set_ylabel("Number of charts")
    ax.set_title("Steps per second distribution")
    fig.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, "steps_per_second.png"), dpi=150)
    if show:
        plt.show()
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize arrow training data: stats and plots for a training data directory."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing chart (.txt) and audio files (same layout as arrow training data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save summary.txt and PNG figures. If not set, figures are only shown.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not call plt.show(); only save files when --output_dir is set.",
    )
    parser.add_argument(
        "--top_arrows",
        type=int,
        default=25,
        help="Number of top arrow types to show in bar chart (default 25).",
    )
    args = parser.parse_args()

    agg = collect_aggregates(args.data_dir)
    if agg is None:
        print(f"No audio-chart pairs found in {args.data_dir!r}.", file=sys.stderr)
        return 2

    show = not args.no_show
    out = args.output_dir
    if out:
        os.makedirs(out, exist_ok=True)

    summary = write_summary(
        agg,
        os.path.join(out, "summary.txt") if out else None,
    )
    print(summary)

    plot_note_kind(agg, out, show)
    plot_top_arrow_types(agg, args.top_arrows, out, show)
    plot_per_column_activity(agg, out, show)
    plot_chart_validity(agg, out, show)
    plot_steps_per_chart(agg, out, show)
    plot_chord_size(agg, out, show)
    plot_bpm(agg, out, show)
    plot_chart_duration(agg, out, show)
    plot_inter_step_interval(agg, out, show)
    plot_steps_per_second(agg, out, show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
