"""Tests for DDC placement datasets."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet.dataset_prep import constants, training_index
from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddc import config, datasets


def _entry(
    split: str,
    song: str,
    difficulty: str = "easy",
    chart_index: int = 0,
) -> training_index.TrainingIndexEntry:
    """Build a manifest row for tests.

    Args:
        split: ``train`` or ``val``.
        song: Song id.
        difficulty: Difficulty label.
        chart_index: Chart block index.

    Returns:
        Manifest entry.
    """
    return training_index.TrainingIndexEntry(
        split=split,
        normalized_bundle="bundle",
        normalized_id=song,
        chart_index=chart_index,
        output_relpath=f"bundle/{song}",
        difficulty=difficulty,
        meter=5,
        num_steps=10,
        audio_relpath=f"bundle/{song}/{song}.ogg",
        chart_relpath=f"bundle/{song}/{song}.chart.json",
    )


def _synthetic_chart(
    n_frames: int = 80,
    *,
    difficulty: str = "easy",
    first: int = 10,
    last: int = 70,
) -> datasets.PlacementChart:
    """Return an in-memory chart with a single onset span.

    Args:
        n_frames: Spectrogram length.
        difficulty: DDR difficulty label.
        first: First labeled onset frame.
        last: Last labeled onset frame.

    Returns:
        Placement chart.
    """
    spec = np.zeros((n_frames, 80, 3), dtype=np.float32)
    target = np.zeros((n_frames,), dtype=np.float32)
    target[first] = 1.0
    target[last] = 1.0
    return datasets.PlacementChart(
        song_key="bundle/song",
        difficulty=difficulty,
        spec=spec,
        target=target,
        gt_times=np.array([first * 0.01, last * 0.01], dtype=np.float32),
        first_onset=first,
        last_onset=last,
    )


class DdcDatasetsTest(unittest.TestCase):
    def test_valid_window_and_sample_batch(self):
        chart = _synthetic_chart()
        starts = datasets.valid_window_starts(chart, nunroll=32)
        self.assertGreater(starts.size, 0)
        audio, difficulty, target = datasets.extract_unroll_window(
            chart, int(starts[0]), 32
        )
        self.assertEqual(audio.shape, (32, 15, 80, 3))
        self.assertEqual(difficulty.shape, (32, 5))
        self.assertEqual(target.shape, (32, 1))
        self.assertEqual(float(difficulty[0, 1]), 1.0)
        rng = np.random.default_rng(0)
        inputs, labels = datasets.sample_train_batch(
            [chart],
            batch_size=3,
            nunroll=32,
            rng=rng,
        )
        self.assertEqual(inputs["audio"].shape[0], 3)
        self.assertEqual(labels.shape, (3, 32, 1))

    def test_sample_batch_requires_valid_span(self):
        short = _synthetic_chart(n_frames=20, first=5, last=8)
        with self.assertRaises(ValueError):
            datasets.sample_train_batch(
                [short],
                batch_size=1,
                nunroll=32,
                rng=np.random.default_rng(0),
            )

    def test_list_split_entries_caps_songs(self):
        entries = [
            _entry("train", "song_a", "beginner", 0),
            _entry("train", "song_a", "easy", 1),
            _entry("train", "song_b", "hard", 0),
            _entry("val", "song_c", "challenge", 0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            index = training_index.TrainingIndex(
                schema_version=constants.SCHEMA_VERSION,
                output_dir=tmpdir,
                split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
                split_seed=42,
                val_fraction=0.5,
                created_at="2026-01-01T00:00:00Z",
                counts=training_index._counts_from_entries(entries),
                entries=entries,
            )
            path = training_index.save_training_index(
                index,
                pathlib.Path(tmpdir) / "training_index.json",
            )
            dataset_config = config.PlacementDatasetConfig(
                training_index_path=str(path),
                max_train_songs=1,
                data_root=tmpdir,
            )
            train_rows = datasets.list_split_entries(dataset_config, "train")
            self.assertEqual(len(train_rows), 2)
            self.assertEqual({row.normalized_id for row in train_rows}, {"song_a"})
            self.assertEqual(datasets.resolve_data_root(dataset_config), tmpdir)
            inferred = config.PlacementDatasetConfig(
                training_index_path=str(path),
            )
            self.assertEqual(datasets.resolve_data_root(inferred), tmpdir)

    def test_load_placement_chart_uses_feature_loader(self):
        spec = np.zeros((40, 80, 3), dtype=np.float32)
        times = np.array([0.10, 0.30], dtype=np.float64)
        entry = _entry("train", "song_a")
        pack = mock.Mock()
        pack.metadata.offset_sec = 0.0
        pack.metadata.bpm_segments = [
            prep_models.BpmSegment(start_beat=0.0, bpm=120.0)
        ]
        with (
            mock.patch(
                "stepcovnet.ddc.datasets.features.load_or_compute_ddc_logmel",
                return_value=spec,
            ),
            mock.patch(
                "stepcovnet.ddc.datasets.training_loader.load_chart_times_sec",
                return_value=times,
            ),
            mock.patch(
                "stepcovnet.ddc.datasets.prep_models.load_parsed_song",
                return_value=pack,
            ),
        ):
            chart = datasets.load_placement_chart(
                entry, "/tmp/data", cache_features=False
            )
        self.assertEqual(chart.n_frames, 40)
        self.assertEqual(chart.difficulty, "easy")
        self.assertGreater(chart.target.sum(), 0)
        self.assertEqual(chart.offset_sec, 0.0)
        self.assertEqual(len(chart.bpm_segments), 1)
        self.assertEqual(chart.gt_times.dtype, np.float64)

    def test_load_placement_chart_rejects_empty_onsets(self):
        spec = np.zeros((10, 80, 3), dtype=np.float32)
        pack = mock.Mock()
        pack.metadata.offset_sec = 0.0
        pack.metadata.bpm_segments = [
            prep_models.BpmSegment(start_beat=0.0, bpm=120.0)
        ]
        with (
            mock.patch(
                "stepcovnet.ddc.datasets.features.load_or_compute_ddc_logmel",
                return_value=spec,
            ),
            mock.patch(
                "stepcovnet.ddc.datasets.training_loader.load_chart_times_sec",
                return_value=np.array([99.0]),
            ),
            mock.patch(
                "stepcovnet.ddc.datasets.prep_models.load_parsed_song",
                return_value=pack,
            ),
            self.assertRaises(ValueError),
        ):
            datasets.load_placement_chart(
                _entry("train", "song_a"),
                "/tmp/data",
                cache_features=False,
            )

    def test_list_split_entries_empty_raises(self):
        entries = [_entry("train", "song_a")]
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
            path = training_index.save_training_index(
                index,
                pathlib.Path(tmpdir) / "training_index.json",
            )
            dataset_config = config.PlacementDatasetConfig(
                training_index_path=str(path),
            )
            with self.assertRaises(ValueError):
                datasets.list_split_entries(dataset_config, "val")

    def test_load_split_charts_delegates(self):
        entry = _entry("train", "song_a")
        chart = _synthetic_chart()
        dataset_config = config.PlacementDatasetConfig(
            training_index_path="unused.json",
            data_root="/tmp/data",
        )
        with (
            mock.patch(
                "stepcovnet.ddc.datasets.list_split_entries",
                return_value=[entry],
            ),
            mock.patch(
                "stepcovnet.ddc.datasets.resolve_data_root",
                return_value="/tmp/data",
            ),
            mock.patch(
                "stepcovnet.ddc.datasets.load_placement_chart",
                return_value=chart,
            ) as load_chart,
        ):
            loaded = datasets.load_split_charts(dataset_config, "train")
        self.assertEqual(loaded, [chart])
        load_chart.assert_called_once()


if __name__ == "__main__":
    unittest.main()
