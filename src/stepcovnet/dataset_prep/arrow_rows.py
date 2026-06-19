"""Beat-grouped quaternary arrow row encoding."""

from __future__ import annotations

import dataclasses

from simfile import notes

_ARROW_CHAR_RANK = {"0": 0, "1": 1, "2": 2, "3": 3}


@dataclasses.dataclass
class EncodeChartStats:
    """Per-chart counters from simfile note encoding.

    Attributes:
        mine_notes_unencoded: Mine notes ignored during encoding.
        fake_notes_unencoded: Fake notes ignored during encoding.
        lift_notes_unencoded: Lift notes ignored during encoding.
        beats_dropped_empty: Beats with no encodable player steps.
    """

    mine_notes_unencoded: int = 0
    fake_notes_unencoded: int = 0
    lift_notes_unencoded: int = 0
    beats_dropped_empty: int = 0


def note_type_to_arrow_char(note_type: notes.NoteType) -> str | None:
    """Map a simfile note type to a quaternary arrow character.

    Args:
        note_type: Simfile note type from ``NoteData``.

    Returns:
        ``1``, ``2``, or ``3`` for encodable types; ``None`` otherwise.
    """
    if note_type == notes.NoteType.TAP:
        return "1"
    if note_type in (notes.NoteType.HOLD_HEAD, notes.NoteType.ROLL_HEAD):
        return "2"
    if note_type == notes.NoteType.TAIL:
        return "3"
    return None


def _count_non_encodable(note_type: notes.NoteType, stats: EncodeChartStats) -> None:
    if note_type == notes.NoteType.MINE:
        stats.mine_notes_unencoded += 1
    elif note_type == notes.NoteType.FAKE:
        stats.fake_notes_unencoded += 1
    elif note_type == notes.NoteType.LIFT:
        stats.lift_notes_unencoded += 1


def _apply_char_to_row(row: list[str], column: int, char: str) -> None:
    if column < 0 or column > 3:
        return
    current = row[column]
    if _ARROW_CHAR_RANK[char] > _ARROW_CHAR_RANK[current]:
        row[column] = char


def build_arrow_row(beat_notes: list[notes.Note]) -> str | None:
    """Merge notes on one beat into a quaternary arrow row.

    Args:
        beat_notes: All simfile notes sharing the same beat.

    Returns:
        Four-character row, or ``None`` when the beat encodes to ``0000``.
    """
    row = ["0", "0", "0", "0"]
    for note in beat_notes:
        char = note_type_to_arrow_char(note.note_type)
        if char is None:
            continue
        _apply_char_to_row(row, note.column, char)
    merged = "".join(row)
    if merged == "0000":
        return None
    return merged


def encode_timed_chart_rows(
    timed_notes: list,
) -> tuple[list[float], list[str], list[int], EncodeChartStats]:
    """Group timed notes by beat and build chart row sequences.

    Args:
        timed_notes: Iterable of ``TimedNote`` objects from ``time_notes``.

    Returns:
        ``times_sec``, ``arrow_rows``, ``column_codes``, and encoding stats.
    """
    stats = EncodeChartStats()
    by_beat: dict[float, list[notes.Note]] = {}
    beat_to_time: dict[float, float] = {}

    for timed_note in timed_notes:
        note = timed_note.note
        char = note_type_to_arrow_char(note.note_type)
        if char is None:
            _count_non_encodable(note.note_type, stats)
        beat_key = float(note.beat)
        beat_to_time.setdefault(beat_key, float(timed_note.time))
        by_beat.setdefault(beat_key, []).append(note)

    times_sec: list[float] = []
    arrow_rows: list[str] = []
    column_codes: list[int] = []

    for beat_key in sorted(beat_to_time):
        row = build_arrow_row(by_beat[beat_key])
        if row is None:
            stats.beats_dropped_empty += 1
            continue
        times_sec.append(beat_to_time[beat_key])
        arrow_rows.append(row)
        column_codes.append(int(row, 4))

    return times_sec, arrow_rows, column_codes, stats
