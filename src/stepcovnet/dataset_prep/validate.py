"""Pack and per-chart validation gates."""

from __future__ import annotations

from stepcovnet.dataset_prep import models


def times_monotonic(times_sec: list[float]) -> bool:
    """Return True when step times are strictly non-decreasing.

    Args:
        times_sec: Encoded beat times in seconds.

    Returns:
        True if every time is >= the previous time.
    """
    if len(times_sec) < 2:
        return True
    for previous, current in zip(times_sec[:-1], times_sec[1:], strict=True):
        if current < previous:
            return False
    return True


def validate_parsed_pack(pack: models.ParsedSongPack) -> list[str]:
    """Validate a parsed pack before writing to disk.

    Args:
        pack: Parsed song pack from simfile parsing.

    Returns:
        Error codes; empty list means validation passed.
    """
    errors: list[str] = []
    if not pack.audio_filename:
        errors.append("no_audio")
    if not pack.charts:
        errors.append("no_exportable_charts")
    for index, chart in enumerate(pack.charts):
        if not chart.arrow_rows:
            errors.append(f"chart_{index}_empty")
        if len(chart.times_sec) != len(chart.arrow_rows):
            errors.append(f"chart_{index}_length_mismatch")
        if not times_monotonic(chart.times_sec):
            errors.append(f"chart_{index}_non_monotonic_times")
    return errors
