"""Tests for ITGPT chart padding."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import config, constants, datasets


def _chart(*, n_beats: int = 6) -> ddcl_datasets.DdclChart:
    """Return a short chart for packing tests.

    Args:
        n_beats: Integer-beat length.

    Returns:
        DdclChart.
    """
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 1] = 140.0
    return ddcl_datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=8,
        beat_audio=np.ones(
            (
                n_beats,
                constants.N_FRAMES_PER_BEAT,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        ),
        stream=stream,
        slots=slots,
        memlen=0,
    )


class ItgptDatasetsTest(unittest.TestCase):
    def test_pad_length_aligns_to_64(self):
        self.assertEqual(datasets.pad_length(1, 256), constants.CHUNK_ALIGN)
        self.assertEqual(datasets.pad_length(64, 256), 64)
        self.assertEqual(datasets.pad_length(65, 256), 128)
        self.assertEqual(datasets.pad_length(300, 256), 256)

    def test_pack_chart_and_grid_weights(self):
        chart = _chart()
        inputs, slots, mask = datasets.pack_chart(
            chart, max_beats=constants.CHUNK_ALIGN
        )
        self.assertEqual(inputs["audio"].shape[1], constants.CHUNK_ALIGN)
        self.assertEqual(int(mask.sum()), chart.n_beats)
        self.assertEqual(float(inputs["bpm"][0, 0]), 140.0)
        self.assertEqual(float(inputs["difficulty"][0, 0]), 8.0)
        self.assertEqual(float(slots[0, 0, 0]), 1.0)
        weights = datasets.sample_weight(mask)
        self.assertEqual(weights.shape, (1, constants.CHUNK_ALIGN, constants.N_SLOTS))
        self.assertEqual(float(weights[0, 0, 0]), constants.GRID_WEIGHT_16TH)
        self.assertEqual(float(weights[0, chart.n_beats, 0]), 0.0)
        batch = next(
            datasets.batch_generator([chart], max_beats=constants.CHUNK_ALIGN, seed=0)
        )
        self.assertEqual(len(batch), 3)

    def test_pack_chart_mert_audio_shape(self):
        n_beats = 6
        slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
        slots[0, 0] = 1.0
        stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
        stream[:, 1] = 150.0
        mert_chart = ddcl_datasets.DdclChart(
            song_key="bundle/mert_song",
            difficulty="hard",
            meter=9,
            beat_audio=np.ones((n_beats, 24, 16), dtype=np.float32),
            stream=stream,
            slots=slots,
            memlen=0,
        )
        inputs, packed_slots, mask = datasets.pack_chart(
            mert_chart, max_beats=constants.CHUNK_ALIGN
        )
        self.assertEqual(inputs["audio"].shape, (1, constants.CHUNK_ALIGN, 24, 16))
        self.assertEqual(float(inputs["audio"][0, 0, 0, 0]), 1.0)
        self.assertEqual(float(inputs["audio"][0, n_beats, 0, 0]), 0.0)
        self.assertEqual(int(mask.sum()), n_beats)
        self.assertEqual(float(packed_slots[0, 0, 0]), 1.0)

    def test_load_split_charts_forwards_to_ddcl(self):
        chart = _chart()
        dataset_config = config.ItgptDatasetConfig(
            training_index_path="missing.json",
            data_root="root",
            max_train_songs=2,
            max_val_songs=1,
            cache_features=False,
            max_beats=constants.CHUNK_ALIGN,
            exclude_normalized_ids=["bad_song"],
            audio_feature_type="mert",
            n_frames_per_beat=24,
        )
        with mock.patch(
            "stepcovnet.ddcl.datasets.load_split_charts",
            return_value=[chart],
        ) as mock_load:
            loaded = datasets.load_split_charts(dataset_config, "train")
        self.assertEqual(loaded, [chart])
        ddcl_cfg = mock_load.call_args[0][0]
        self.assertEqual(ddcl_cfg.training_index_path, "missing.json")
        self.assertEqual(ddcl_cfg.memlen, 0)
        self.assertEqual(ddcl_cfg.max_train_songs, 2)
        self.assertEqual(ddcl_cfg.exclude_normalized_ids, ["bad_song"])
        self.assertEqual(ddcl_cfg.audio_feature_type, "mert")
        self.assertEqual(ddcl_cfg.n_frames_per_beat, 24)

    def test_build_dataloader_yields_variable_beat_lengths(self):
        short_chart = _chart(n_beats=6)
        long_chart = _chart(n_beats=70)
        loader = datasets.build_dataloader(
            [short_chart, long_chart], max_beats=128, seed=0
        )
        it = iter(loader)
        seen_lengths = {int(next(it)[0]["audio"].shape[1]) for _ in range(12)}
        self.assertEqual(seen_lengths, {64, 128})
        with self.assertRaises(ValueError):
            datasets.build_dataloader([], max_beats=128, seed=0)

    def test_pack_chart_and_generator_bar_shift_augmentation(self):
        chart = _chart(n_beats=6)
        inputs, slots, mask = datasets.pack_chart(
            chart, max_beats=128, lead_pad_beats=8
        )
        self.assertEqual(inputs["audio"].shape[1], 64)
        self.assertEqual(int(mask.sum()), 6)
        np.testing.assert_array_equal(mask[0, :8], 0.0)
        np.testing.assert_array_equal(mask[0, 8:14], 1.0)
        np.testing.assert_array_equal(mask[0, 14:], 0.0)
        self.assertEqual(float(inputs["audio"][0, 7].sum()), 0.0)
        self.assertGreater(float(inputs["audio"][0, 8].sum()), 0.0)
        self.assertEqual(float(slots[0, 8, 0]), 1.0)
        self.assertEqual(float(slots[0, 0, 0]), 0.0)

        gen = datasets.batch_generator([chart], max_beats=128, seed=42, max_bar_shift=4)
        leads = set()
        for _ in range(32):
            _, _, step_mask = next(gen)
            self.assertEqual(int(step_mask.sum()), 6)
            first_active = int(np.argmax(step_mask[0] > 0.5))
            self.assertEqual(first_active % 4, 0)
            self.assertLessEqual(first_active, 16)
            leads.add(first_active)
        self.assertGreater(len(leads), 1)


if __name__ == "__main__":
    unittest.main()
