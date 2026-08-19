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

    def test_load_split_charts_forwards_to_ddcl(self):
        chart = _chart()
        dataset_config = config.ItgptDatasetConfig(
            training_index_path="missing.json",
            data_root="root",
            max_train_songs=2,
            max_val_songs=1,
            cache_features=False,
            max_beats=constants.CHUNK_ALIGN,
        )
        with mock.patch(
            "stepcovnet.itgpt.datasets.ddcl_datasets.load_split_charts",
            return_value=[chart],
        ) as mock_load:
            loaded = datasets.load_split_charts(dataset_config, "train")
        self.assertEqual(loaded, [chart])
        ddcl_cfg = mock_load.call_args[0][0]
        self.assertEqual(ddcl_cfg.training_index_path, "missing.json")
        self.assertEqual(ddcl_cfg.memlen, 0)
        self.assertEqual(ddcl_cfg.max_train_songs, 2)


if __name__ == "__main__":
    unittest.main()
