"""Tests for DDCL dataset batching."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from stepcovnet.dataset_prep import constants as prep_constants
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, constants, datasets


def _chart() -> datasets.DdclChart:
    n_beats = 4
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[1, 3] = 1.0
    return datasets.DdclChart(
        song_key="b/s",
        difficulty="hard",
        meter=8,
        beat_audio=np.zeros(
            (n_beats, 8, constants.N_MELS, constants.N_CHANNELS),
            dtype=np.float32,
        ),
        stream=np.zeros((n_beats, constants.STREAM_DIM), dtype=np.float32),
        slots=slots,
        memlen=1,
    )


class DdclDatasetsTest(unittest.TestCase):
    def test_sample_train_batch_shapes(self):
        rng = np.random.default_rng(0)
        inputs, labels = datasets.sample_train_batch(
            [_chart()],
            batch_size=3,
            rng=rng,
        )
        self.assertEqual(inputs["audio_fwd"].shape[0], 3)
        self.assertEqual(labels.shape, (3, constants.N_SLOTS))
        predict_inputs = datasets.chart_model_inputs(_chart())
        self.assertEqual(predict_inputs["audio_fwd"].shape[0], 4)
        batches = list(datasets.iter_window_batches(_chart(), batch_size=3))
        self.assertEqual([int(batch.shape[0]) for batch in batches], [3, 1])

    def test_windows_for_indices_rejects_empty(self):
        with self.assertRaises(ValueError):
            datasets.windows_for_indices(_chart(), np.array([], dtype=np.int64))

    def test_list_split_entries_excludes_song_ids(self):
        def entry(
            song_id: str, split: training_index.SplitName
        ) -> training_index.TrainingIndexEntry:
            return training_index.TrainingIndexEntry(
                split=split,
                normalized_bundle="b",
                normalized_id=song_id,
                chart_index=0,
                output_relpath=f"b/{song_id}",
                difficulty="hard",
                meter=8,
                num_steps=4,
                audio_relpath=f"b/{song_id}/{song_id}.ogg",
                chart_relpath=f"b/{song_id}/{song_id}.chart.json",
            )

        index = training_index.TrainingIndex(
            schema_version=prep_constants.SCHEMA_VERSION,
            output_dir="data/final_data",
            split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
            split_seed=42,
            val_fraction=0.5,
            created_at="2026-10-10T00:00:00Z",
            counts=training_index.TrainingIndexCounts(
                songs={"train": 1, "val": 1},
                rows={"train": 1, "val": 1},
            ),
            entries=[entry("keep", "train"), entry("news_39", "val")],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "index.json"
            training_index.save_training_index(index, path)
            dataset_config = config.DdclDatasetConfig(
                training_index_path=str(path),
                exclude_normalized_ids=["news_39"],
            )
            kept = datasets.list_split_entries(dataset_config, "train")
            self.assertEqual([row.normalized_id for row in kept], ["keep"])
            with self.assertRaises(ValueError):
                datasets.list_split_entries(dataset_config, "val")

    def test_sample_rejects_empty(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            datasets.sample_train_batch([], batch_size=1, rng=rng)


if __name__ == "__main__":
    unittest.main()
