"""Tests for pure-Python dataset_prep.simfile_parser."""

from __future__ import annotations

import fractions
import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import simfile_parser


class MsdParserTest(unittest.TestCase):
    def test_parses_escapes_comments_bom_and_missing_semicolon(self):
        text = (
            "\ufeff// header comment\n"
            "#TITLE:Title\\: With\\; Escapes\\# and \\\\Slash;\n"
            "#ARTIST:MissingSemi\n"
            "#OFFSET:-0.25;\n"
            "#DISPLAYBPM:120:180;\n"
        )
        sim = simfile_parser.parse_simfile_text(text, is_ssc=False)
        self.assertEqual(sim.title, "Title: With; Escapes# and \\Slash")
        self.assertEqual(sim.artist, "MissingSemi\n")
        self.assertEqual(sim.offset, "-0.25")
        self.assertEqual(sim.properties["DISPLAYBPM"], "120:180")

    def test_strict_mode_rejects_stray_text(self):
        with self.assertRaises(ValueError):
            simfile_parser.parse_simfile_text("junk\n#TITLE:Test;", strict=True)

    def test_auto_detects_ssc_vs_sm_from_first_tag(self):
        ssc = simfile_parser.parse_simfile_text("#VERSION:0.83;\n#TITLE:A;")
        sm = simfile_parser.parse_simfile_text("#TITLE:B;")
        self.assertTrue(ssc.is_ssc)
        self.assertFalse(sm.is_ssc)

    def test_sm_rejects_incomplete_notes_block(self):
        with self.assertRaises(ValueError):
            simfile_parser.parse_simfile_text(
                "#TITLE:Bad;\n#NOTES:dance-single:desc:Hard;",
                is_ssc=False,
            )


class TimingAndNotesTest(unittest.TestCase):
    def test_ssc_multi_bpm_stops_delays_and_overlapping_warps(self):
        sim_text = """#VERSION:0.83;
#TITLE:Complex;
#OFFSET:-0.125;
#BPMS:0.000=120.000,4.000=180.000,12.000=90.000;
#STOPS:4.000=0.500,8.000=0.250;
#DELAYS:4.000=0.300,10.000=0.400;
#WARPS:6.000=2.000,7.000=2.000;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Challenge;
#METER:13;
#NOTES:
1000
0100
0010
0001
,
2000
0000
1000
0000
1[0]000
000M
0000
0000
,
3000
0000
0100
0000
0010
0000
0001
0000
;
"""
        sim = simfile_parser.parse_simfile_text(sim_text)
        timed = simfile_parser.time_chart_notes(sim, sim.charts[0])

        # Beat 0..3 at 120 BPM (0.5s/beat) starting at -offset = +0.125s
        self.assertAlmostEqual(timed[0].time, 0.125)
        self.assertAlmostEqual(timed[1].time, 0.625)
        self.assertAlmostEqual(timed[2].time, 1.125)
        self.assertAlmostEqual(timed[3].time, 1.625)

        # Beat 4 has DELAY=0.3 before note hit, STOP=0.5 after note hit:
        # arrival at beat 4 is 2.125, after delay -> 2.425
        self.assertEqual(timed[4].note.beat, fractions.Fraction(4))
        self.assertEqual(timed[4].note.note_type, simfile_parser.NoteType.HOLD_HEAD)
        self.assertAlmostEqual(timed[4].time, 2.425)

        # Beat 5 at 180 BPM (1/3 s/beat) after 0.5s stop: 2.425 + 0.5 + 1/3
        self.assertEqual(timed[5].note.beat, fractions.Fraction(5))
        self.assertAlmostEqual(timed[5].time, 2.925 + 1.0 / 3.0)

        # Beat 6 is warp start (coalesced warp [6, 9)):
        # TAP with keysound 0 is converted to FAKE; MINE on beat 6.5 is dropped.
        self.assertEqual(timed[6].note.beat, fractions.Fraction(6))
        self.assertEqual(timed[6].note.note_type, simfile_parser.NoteType.FAKE)
        self.assertEqual(timed[6].note.keysound_index, 0)
        self.assertAlmostEqual(timed[6].time, 2.925 + 2.0 / 3.0)
        self.assertFalse(any(t.note.beat == fractions.Fraction(13, 2) for t in timed))

        # Beat 8 has STOP=0.25 inside warp [6, 9) -> hittable at warp time
        self.assertEqual(timed[7].note.beat, fractions.Fraction(8))
        self.assertEqual(timed[7].note.note_type, simfile_parser.NoteType.TAIL)
        self.assertAlmostEqual(timed[7].time, 2.925 + 2.0 / 3.0)

        # Beat 9 is warp end -> hittable after beat 8's 0.25s stop (0 time for warp)
        self.assertEqual(timed[8].note.beat, fractions.Fraction(9))
        self.assertAlmostEqual(timed[8].time, 2.925 + 2.0 / 3.0 + 0.25)

        # Beat 10 has DELAY=0.4 -> 1 beat at 180 BPM + 0.4s delay
        self.assertEqual(timed[9].note.beat, fractions.Fraction(10))
        self.assertAlmostEqual(timed[9].time, 2.925 + 1.0 + 0.25 + 0.4)

    def test_ssc_split_timing_respects_version_threshold(self):
        base = """#VERSION:{ver};
#TITLE:Split;
#OFFSET:0.0;
#BPMS:0.0=120.0;
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Hard;
#METER:9;
#OFFSET:0.25;
#BPMS:0.0=240.0;
#NOTES2:
1000
0100
0000
0000
;
"""
        sim_split = simfile_parser.parse_simfile_text(base.format(ver="0.83"))
        timed_split = simfile_parser.time_chart_notes(sim_split, sim_split.charts[0])
        # Uses chart offset (-0.25) and chart BPM (240 -> 0.25s/beat)
        self.assertAlmostEqual(timed_split[0].time, -0.25)
        self.assertAlmostEqual(timed_split[1].time, 0.0)

        sim_old = simfile_parser.parse_simfile_text(base.format(ver="0.6"))
        timed_old = simfile_parser.time_chart_notes(sim_old, sim_old.charts[0])
        # Uses song offset (0.0) and song BPM (120 -> 0.5s/beat)
        self.assertAlmostEqual(timed_old[0].time, 0.0)
        self.assertAlmostEqual(timed_old[1].time, 0.5)

    def test_sm_freezes_alias_and_encoding_retry(self):
        sm_text = (
            "#TITLE:日本語;\n"
            "#OFFSET:0.0;\n"
            "#BPMS:0.0=120.0;\n"
            "#FREEZES:1.0=0.5;\n"
            "#NOTES:\n"
            "  dance-single:\n"
            "  desc:\n"
            "  Hard:\n"
            "  8:\n"
            "  0,0,0,0,0:\n"
            "1000\n"
            "0100\n"
            "0010\n"
            "0000\n"
            ";\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "chart.sm"
            path.write_bytes(sm_text.encode("cp932"))
            sim = simfile_parser.load_simfile_path(
                path,
                encodings=("utf-8-sig", "utf-8", "cp932"),
            )
            self.assertEqual(sim.title, "日本語")
            timed = simfile_parser.time_chart_notes(sim, sim.charts[0])
            self.assertAlmostEqual(timed[0].time, 0.0)
            self.assertAlmostEqual(timed[1].time, 0.5)
            # Beat 2 is after beat 1's 0.5s freeze
            self.assertAlmostEqual(timed[2].time, 1.5)

    def test_timing_engine_validates_bpms(self):
        empty_td = simfile_parser.TimingData(
            bpms=[],
            stops=[],
            delays=[],
            warps=[],
            offset=simfile_parser.decimal.Decimal(0),
        )
        with self.assertRaises(ValueError):
            simfile_parser.TimingEngine(empty_td)

        bad_start = simfile_parser.TimingData(
            bpms=simfile_parser.parse_beat_values("1.0=120.0"),
            stops=[],
            delays=[],
            warps=[],
            offset=simfile_parser.decimal.Decimal(0),
        )
        with self.assertRaises(ValueError):
            simfile_parser.TimingEngine(bad_start)
