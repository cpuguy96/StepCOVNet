"""Tests for dataset_prep.arrow_rows."""

import unittest

import simfile.notes
from simfile.notes import NoteType
from simfile.notes.timed import TimedNote
from simfile.timing import Beat

from stepcovnet.dataset_prep import arrow_rows


class _FakeNote:
    def __init__(self, note_type, column=0, beat=0.0):
        self.note_type = note_type
        self.column = column
        self.beat = Beat(beat)


class ArrowRowsTest(unittest.TestCase):
    def test_note_type_to_arrow_char_maps_encodable_types(self):
        self.assertEqual(arrow_rows.note_type_to_arrow_char(NoteType.TAP), "1")
        self.assertEqual(arrow_rows.note_type_to_arrow_char(NoteType.HOLD_HEAD), "2")
        self.assertEqual(arrow_rows.note_type_to_arrow_char(NoteType.ROLL_HEAD), "2")
        self.assertEqual(arrow_rows.note_type_to_arrow_char(NoteType.TAIL), "3")
        self.assertIsNone(arrow_rows.note_type_to_arrow_char(NoteType.MINE))

    def test_build_arrow_row_prefers_tail_over_hold_over_tap(self):
        notes = [
            simfile.notes.Note(beat=Beat(1.0), column=0, note_type=NoteType.TAP),
            simfile.notes.Note(beat=Beat(1.0), column=0, note_type=NoteType.HOLD_HEAD),
            simfile.notes.Note(beat=Beat(1.0), column=0, note_type=NoteType.TAIL),
        ]
        self.assertEqual(arrow_rows.build_arrow_row(notes), "3000")

    def test_build_arrow_row_drops_mine_only_beats(self):
        notes = [
            simfile.notes.Note(beat=Beat(1.0), column=2, note_type=NoteType.MINE),
        ]
        self.assertIsNone(arrow_rows.build_arrow_row(notes))

    def test_build_arrow_row_keeps_tap_when_mine_on_other_column(self):
        notes = [
            simfile.notes.Note(beat=Beat(1.0), column=0, note_type=NoteType.TAP),
            simfile.notes.Note(beat=Beat(1.0), column=2, note_type=NoteType.MINE),
        ]
        self.assertEqual(arrow_rows.build_arrow_row(notes), "1000")

    def test_encode_timed_chart_rows_groups_beats_and_counts_stats(self):
        timed = [
            TimedNote(
                time=1.0,
                note=simfile.notes.Note(
                    beat=Beat(1.0), column=0, note_type=NoteType.TAP
                ),
            ),
            TimedNote(
                time=1.0,
                note=simfile.notes.Note(
                    beat=Beat(1.0), column=2, note_type=NoteType.MINE
                ),
            ),
            TimedNote(
                time=2.0,
                note=simfile.notes.Note(
                    beat=Beat(2.0), column=1, note_type=NoteType.FAKE
                ),
            ),
            TimedNote(
                time=3.0,
                note=simfile.notes.Note(
                    beat=Beat(3.0), column=3, note_type=NoteType.TAP
                ),
            ),
        ]
        times_sec, rows, codes, stats = arrow_rows.encode_timed_chart_rows(timed)
        self.assertEqual(times_sec, [1.0, 3.0])
        self.assertEqual(rows, ["1000", "0001"])
        self.assertEqual(codes, [int("1000", 4), int("0001", 4)])
        self.assertEqual(stats.mine_notes_unencoded, 1)
        self.assertEqual(stats.fake_notes_unencoded, 1)
        self.assertEqual(stats.beats_dropped_empty, 1)

    def test_encode_timed_chart_rows_counts_lift_notes(self):
        timed = [
            TimedNote(
                time=1.0,
                note=simfile.notes.Note(
                    beat=Beat(1.0), column=0, note_type=NoteType.LIFT
                ),
            ),
        ]
        _times, _rows, _codes, stats = arrow_rows.encode_timed_chart_rows(timed)
        self.assertEqual(stats.lift_notes_unencoded, 1)
        self.assertEqual(stats.beats_dropped_empty, 1)

    def test_build_arrow_row_ignores_out_of_range_columns(self):
        notes = [
            simfile.notes.Note(beat=Beat(1.0), column=5, note_type=NoteType.TAP),
        ]
        self.assertIsNone(arrow_rows.build_arrow_row(notes))
