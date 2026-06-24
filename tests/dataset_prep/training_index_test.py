"""Tests for dataset_prep.training_index (P8)."""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
import unittest

from stepcovnet import pairing
from stepcovnet.dataset_prep import config, pipeline, training_index, training_loader

_FIXTURES_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "dataset_prep"
)


class TrainingIndexTest(unittest.TestCase):
    @contextlib.contextmanager
    def _prepared_output(self, bundle_dir: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir) / "out"
            prep = config.PrepConfig(
                input_dir=str(_FIXTURES_ROOT / bundle_dir),
                output_dir=str(out_dir),
                overwrite=True,
            )
            pipeline.run_preprocess(prep)
            yield out_dir

    def test_build_assigns_all_charts_for_song_to_same_split(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            index = training_index.build_training_index(
                out_dir,
                val_fraction=0.5,
                seed=7,
            )
            splits = {
                training_index.song_key(
                    entry.normalized_bundle, entry.normalized_id
                ): entry.split
                for entry in index.entries
            }
            self.assertEqual(len(index.entries), 2)
            self.assertEqual(len(set(splits.values())), 1)

    def test_save_load_round_trip(self):
        with self._prepared_output("itl_challenge_ssc") as out_dir:
            built = training_index.build_training_index(
                out_dir, val_fraction=0.0, seed=1
            )
            path = training_index.save_training_index(built)
            loaded = training_index.load_training_index(path)
            self.assertEqual(loaded.counts.rows[training_index.SPLIT_TRAIN], 1)
            self.assertEqual(loaded.counts.rows[training_index.SPLIT_VAL], 0)
            self.assertEqual(len(loaded.entries), 1)

    def test_list_training_samples_filters_by_split(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            training_index.save_training_index(
                training_index.build_training_index(
                    out_dir,
                    val_fraction=0.0,
                    seed=3,
                )
            )
            train_samples = pairing.list_training_samples(out_dir, split="train")
            val_samples = pairing.list_training_samples(out_dir, split="val")
            all_samples = pairing.list_training_samples(out_dir)
            self.assertEqual(len(train_samples), 2)
            self.assertEqual(len(val_samples), 0)
            self.assertEqual(len(all_samples), 2)

    def test_manifest_split_enabled_requires_same_root(self):
        with self._prepared_output("itl_challenge_ssc") as out_dir:
            training_index.save_training_index(
                training_index.build_training_index(out_dir, val_fraction=0.0, seed=1)
            )
            self.assertTrue(training_index.manifest_split_enabled(out_dir, out_dir))
            with tempfile.TemporaryDirectory() as other:
                self.assertFalse(training_index.manifest_split_enabled(out_dir, other))

    def test_validate_rejects_mixed_splits_for_one_song(self):
        entry_train = training_index.TrainingIndexEntry(
            split="train",
            normalized_bundle="b",
            normalized_id="song",
            chart_index=0,
            output_relpath="b/song",
            difficulty="easy",
            meter=1,
            num_steps=1,
            audio_relpath="b/song/song.ogg",
            chart_relpath="b/song/song.chart.json",
        )
        entry_val = training_index.TrainingIndexEntry(
            split="val",
            normalized_bundle="b",
            normalized_id="song",
            chart_index=1,
            output_relpath="b/song",
            difficulty="hard",
            meter=9,
            num_steps=2,
            audio_relpath="b/song/song.ogg",
            chart_relpath="b/song/song.chart.json",
        )
        index = training_index.TrainingIndex(
            schema_version=1,
            output_dir="/tmp/out",
            split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
            split_seed=1,
            val_fraction=0.5,
            created_at="2026-01-01T00:00:00Z",
            counts=training_index.TrainingIndexCounts(
                songs={"train": 1, "val": 0},
                rows={"train": 1, "val": 1},
            ),
            entries=[entry_train, entry_val],
        )
        errors = training_index.validate_training_index(index)
        self.assertTrue(any("mixed splits" in err for err in errors))

    def test_assign_stratified_song_splits_reproducible(self):
        songs = {"bundle_a": ["s1", "s2", "s3", "s4"]}
        first = training_index.assign_stratified_song_splits(
            songs,
            val_fraction=0.25,
            seed=99,
        )
        second = training_index.assign_stratified_song_splits(
            songs,
            val_fraction=0.25,
            seed=99,
        )
        self.assertEqual(first, second)
        self.assertEqual(sum(split == "val" for split in first.values()), 1)

    def test_rows_for_split_matches_discover_count(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            index = training_index.build_training_index(
                out_dir, val_fraction=0.5, seed=5
            )
            training_index.save_training_index(index)
            train_rows = training_index.rows_for_split(out_dir, "train")
            val_rows = training_index.rows_for_split(out_dir, "val")
            self.assertEqual(
                len(train_rows) + len(val_rows),
                len(training_loader.discover_training_rows(out_dir)),
            )

    def test_written_json_has_schema_version(self):
        with self._prepared_output("itl_challenge_ssc") as out_dir:
            path = training_index.save_training_index(
                training_index.build_training_index(out_dir, val_fraction=0.0, seed=1)
            )
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["split_policy"], "stratified_song_v1")
            self.assertIn("counts", payload)
            self.assertIn("entries", payload)


if __name__ == "__main__":
    unittest.main()
