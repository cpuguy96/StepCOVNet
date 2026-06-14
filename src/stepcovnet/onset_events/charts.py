"""StepMania chart parsing for event-based onset detection."""

import numpy as np

MAX_STEPS_PER_CHART = 2048

_DIFFICULTY_MAP = {"beginner": 0, "easy": 1, "medium": 2, "hard": 3, "challenge": 4}


def _parse_step_times(chart_path: str) -> np.ndarray:
    """Parse a StepMania chart file and return step times in seconds."""
    with open(chart_path) as f:
        f.readline()  # TITLE
        f.readline()  # BPM
        f.readline()  # NOTES
        difficulty_level = f.readline().strip().lower().split(" ")[1]
        _ = _DIFFICULTY_MAP.get(difficulty_level, 2)
        times = []
        for line in f:
            if line.startswith("DIFFICULTY"):
                break
            _arrows, timing = line.strip().split(" ")
            times.append(float(timing))
    return np.sort(np.asarray(times, dtype=np.float64))


def count_steps(chart_path: str) -> int:
    """Return the number of steps in a StepMania chart.

    Args:
        chart_path: Path to the StepMania chart file (.txt or .sm).

    Returns:
        Number of step rows parsed from the chart difficulty section.
    """
    return int(len(_parse_step_times(chart_path)))


def chart_exceeds_step_cap(
    chart_path: str, max_steps: int = MAX_STEPS_PER_CHART
) -> bool:
    """Return whether a chart has more steps than the allowed cap.

    Args:
        chart_path: Path to the StepMania chart file (.txt or .sm).
        max_steps: Maximum allowed steps per chart.

    Returns:
        True when ``count_steps(chart_path)`` is greater than ``max_steps``.
    """
    return count_steps(chart_path) > max_steps


def load_onset_times(
    chart_path: str,
    *,
    max_steps: int | None = MAX_STEPS_PER_CHART,
) -> np.ndarray | None:
    """Load sorted step onset times in seconds from a StepMania chart.

    Args:
        chart_path: Path to the StepMania chart file (.txt or .sm).
        max_steps: When set, return ``None`` if the chart has more than this
            many steps. Pass ``None`` to disable the cap check.

    Returns:
        Sorted ascending array of step times in seconds, or ``None`` when
        ``max_steps`` is set and the chart exceeds that limit.
    """
    times = _parse_step_times(chart_path)
    if max_steps is not None and len(times) > max_steps:
        return None
    return times
