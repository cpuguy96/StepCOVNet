"""Tests for scripts/build_training_index_standard.py."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import build_training_index_standard  # noqa: E402

from stepcovnet.dataset_prep import constants, training_index


class BuildTrainingIndexStandardTest(unittest.TestCase):
    def test_resolve_output_path_default_and_override(self):
        source = pathlib.Path("data/literature_fraxtil_orig/training_index.json")
        default = build_training_index_standard.resolve_output_path(source, "")
        self.assertEqual(default.name, training_index.STANDARD_INDEX_FILENAME)
        override = build_training_index_standard.resolve_output_path(
            source,
            "out.json",
        )
        self.assertEqual(override, pathlib.Path("out.json"))

    def test_main_writes_filtered_index(self):
        entries = [
            training_index.TrainingIndexEntry(
                split="train",
                normalized_bundle="b",
                normalized_id="s",
                chart_index=0,
                output_relpath="b/s",
                difficulty="easy",
                meter=1,
                num_steps=1,
                audio_relpath="b/s/s.ogg",
                chart_relpath="b/s/s.chart.json",
            ),
            training_index.TrainingIndexEntry(
                split="train",
                normalized_bundle="b",
                normalized_id="s",
                chart_index=1,
                output_relpath="b/s",
                difficulty="edit",
                meter=1,
                num_steps=1,
                audio_relpath="b/s/s.ogg",
                chart_relpath="b/s/s.chart.json",
            ),
        ]
        index = training_index.TrainingIndex(
            schema_version=constants.SCHEMA_VERSION,
            output_dir="/tmp/out",
            split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
            split_seed=42,
            val_fraction=0.0,
            created_at="2026-01-01T00:00:00Z",
            counts=training_index._counts_from_entries(entries),
            entries=entries,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            source = training_index.save_training_index(
                index,
                pathlib.Path(tmpdir) / "training_index.json",
            )
            output = pathlib.Path(tmpdir) / "training_index_standard.json"
            code = build_training_index_standard.main(
                ["--source", str(source), "--output", str(output)]
            )
            self.assertEqual(code, 0)
            loaded = training_index.load_training_index(output)
            self.assertEqual(len(loaded.entries), 1)
            self.assertEqual(loaded.entries[0].difficulty, "easy")
            missing = build_training_index_standard.main(
                ["--source", str(pathlib.Path(tmpdir) / "nope.json")]
            )
            self.assertEqual(missing, 1)
            exists = build_training_index_standard.main(
                ["--source", str(source), "--output", str(output)]
            )
            self.assertEqual(exists, 1)

    def test_main_overwrite_and_bad_tag(self):
        entries = [
            training_index.TrainingIndexEntry(
                split="train",
                normalized_bundle="b",
                normalized_id="s",
                chart_index=0,
                output_relpath="b/s",
                difficulty="easy",
                meter=1,
                num_steps=1,
                audio_relpath="b/s/s.ogg",
                chart_relpath="b/s/s.chart.json",
            ),
        ]
        index = training_index.TrainingIndex(
            schema_version=constants.SCHEMA_VERSION,
            output_dir="/tmp/out",
            split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
            split_seed=42,
            val_fraction=0.0,
            created_at="2026-01-01T00:00:00Z",
            counts=training_index._counts_from_entries(entries),
            entries=entries,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            source = training_index.save_training_index(
                index,
                pathlib.Path(tmpdir) / "training_index.json",
            )
            output = pathlib.Path(tmpdir) / "std.json"
            self.assertEqual(
                build_training_index_standard.main(
                    ["--source", str(source), "--output", str(output)]
                ),
                0,
            )
            self.assertEqual(
                build_training_index_standard.main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--overwrite",
                    ]
                ),
                0,
            )
            self.assertEqual(
                build_training_index_standard.main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--overwrite",
                        "--policy-tag",
                        " ",
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
