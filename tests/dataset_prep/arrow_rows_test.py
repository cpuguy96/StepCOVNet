"""Tests for dataset_prep.arrow_rows."""

import fractions
import unittest

from stepcovnet.dataset_prep import arrow_rows, simfile_parser


class ArrowRowsTest(unittest.TestCase):
    def test_note_type_to_arrow_char_maps_encodable_types(self):
        self.assertEqual(
            arrow_rows.note_type_to_arrow_char(simfile_parser.NoteType.TAP), "1"
        )
        self.assertEqual(
            arrow_rows.note_type_to_arrow_char(simfile_parser.NoteType.HOLD_HEAD), "2"
        )
        self.assertEqual(
            arrow_rows.note_type_to_arrow_char(simfile_parser.NoteType.ROLL_HEAD), "2"
        )
        self.assertEqual(
            arrow_rows.note_type_to_arrow_char(simfile_parser.NoteType.TAIL), "3"
        )
        self.assertIsNone(
            arrow_rows.note_type_to_arrow_char(simfile_parser.NoteType.MINE)
        )

    def test_build_arrow_row_prefers_tail_over_hold_over_tap(self):
        beat_notes = [
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=0,
                note_type=simfile_parser.NoteType.TAP,
            ),
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=0,
                note_type=simfile_parser.NoteType.HOLD_HEAD,
            ),
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=0,
                note_type=simfile_parser.NoteType.TAIL,
            ),
        ]
        self.assertEqual(arrow_rows.build_arrow_row(beat_notes), "3000")

    def test_build_arrow_row_drops_mine_only_beats(self):
        beat_notes = [
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=2,
                note_type=simfile_parser.NoteType.MINE,
            ),
        ]
        self.assertIsNone(arrow_rows.build_arrow_row(beat_notes))

    def test_build_arrow_row_keeps_tap_when_mine_on_other_column(self):
        beat_notes = [
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=0,
                note_type=simfile_parser.NoteType.TAP,
            ),
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=2,
                note_type=simfile_parser.NoteType.MINE,
            ),
        ]
        self.assertEqual(arrow_rows.build_arrow_row(beat_notes), "1000")

    def test_encode_timed_chart_rows_groups_beats_and_counts_stats(self):
        timed_notes = [
            simfile_parser.TimedNote(
                time=1.0,
                note=simfile_parser.Note(
                    beat=fractions.Fraction(1),
                    column=0,
                    note_type=simfile_parser.NoteType.TAP,
                ),
            ),
            simfile_parser.TimedNote(
                time=1.0,
                note=simfile_parser.Note(
                    beat=fractions.Fraction(1),
                    column=2,
                    note_type=simfile_parser.NoteType.MINE,
                ),
            ),
            simfile_parser.TimedNote(
                time=2.0,
                note=simfile_parser.Note(
                    beat=fractions.Fraction(2),
                    column=1,
                    note_type=simfile_parser.NoteType.FAKE,
                ),
            ),
            simfile_parser.TimedNote(
                time=3.0,
                note=simfile_parser.Note(
                    beat=fractions.Fraction(3),
                    column=3,
                    note_type=simfile_parser.NoteType.TAP,
                ),
            ),
        ]
        times_sec, rows, codes, stats = arrow_rows.encode_timed_chart_rows(timed_notes)
        self.assertEqual(times_sec, [1.0, 3.0])
        self.assertEqual(rows, ["1000", "0001"])
        self.assertEqual(codes, [int("1000", 4), int("0001", 4)])
        self.assertEqual(stats.mine_notes_unencoded, 1)
        self.assertEqual(stats.fake_notes_unencoded, 1)
        self.assertEqual(stats.beats_dropped_empty, 1)

    def test_encode_timed_chart_rows_counts_lift_notes(self):
        timed_notes = [
            simfile_parser.TimedNote(
                time=1.0,
                note=simfile_parser.Note(
                    beat=fractions.Fraction(1),
                    column=0,
                    note_type=simfile_parser.NoteType.LIFT,
                ),
            ),
        ]
        _times, _rows, _codes, stats = arrow_rows.encode_timed_chart_rows(timed_notes)
        self.assertEqual(stats.lift_notes_unencoded, 1)
        self.assertEqual(stats.beats_dropped_empty, 1)

    def test_build_arrow_row_ignores_out_of_range_columns(self):
        beat_notes = [
            simfile_parser.Note(
                beat=fractions.Fraction(1),
                column=5,
                note_type=simfile_parser.NoteType.TAP,
            ),
        ]
        self.assertIsNone(arrow_rows.build_arrow_row(beat_notes))
