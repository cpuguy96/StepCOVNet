"""Tests for pure NumPy/PyTorch ITGPT dataset helpers."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import constants
from stepcovnet.pytorch.itgpt import datasets


def _make_chart(n_beats: int = 10, *, mert: bool = False) -> ddcl_datasets.DdclChart:
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 1] = 135.0
    audio_shape = (n_beats, 24, 16) if mert else (n_beats, 32, 80, 3)
    return ddcl_datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=6,
        beat_audio=np.ones(audio_shape, dtype=np.float32),
        stream=stream,
        slots=slots,
        memlen=0,
    )


class PyTorchItgptDatasetsTest(unittest.TestCase):
    def test_pad_length_alignment(self):
        self.assertEqual(datasets.pad_length(1, 2000), 64)
        self.assertEqual(datasets.pad_length(64, 2000), 64)
        self.assertEqual(datasets.pad_length(65, 2000), 128)
        self.assertEqual(datasets.pad_length(500, 128), 128)

    def test_pack_chart_and_dataloader(self):
        chart = _make_chart(n_beats=10, mert=True)
        inputs, slots, mask = datasets.pack_chart(chart, max_beats=128)
        self.assertEqual(inputs["audio"].shape, (1, 64, 24, 16))
        self.assertEqual(slots.shape, (1, 64, 48))
        self.assertEqual(mask.shape, (1, 64))
        self.assertEqual(float(mask.sum()), 10.0)
        self.assertAlmostEqual(float(inputs["bpm"][0, 0]), 135.0)
        self.assertAlmostEqual(float(inputs["difficulty"][0, 0]), 6.0)

        loader = datasets.build_dataloader([chart], max_beats=128, seed=7)
        batch_inputs, batch_slots, batch_mask = next(iter(loader))
        self.assertEqual(tuple(batch_inputs["audio"].shape), (1, 64, 24, 16))
        self.assertEqual(tuple(batch_slots.shape), (1, 64, 48))
        self.assertEqual(tuple(batch_mask.shape), (1, 64))

    def test_empty_charts_raises(self):
        with self.assertRaises(ValueError):
            datasets.ItgptRandomChartDataset([], max_beats=64, seed=0)

    def test_sample_weight_grid(self):
        mask = np.ones((1, 4), dtype=np.float32)
        weights = datasets.sample_weight(mask)
        self.assertEqual(weights.shape, (1, 4, 48))
        self.assertAlmostEqual(float(weights[0, 0, 0]), constants.GRID_WEIGHT_16TH)
        self.assertAlmostEqual(float(weights[0, 0, 8]), constants.GRID_WEIGHT_24TH)
        self.assertAlmostEqual(float(weights[0, 0, 6]), 1.0)
        self.assertAlmostEqual(float(weights[0, 0, 1]), constants.GRID_WEIGHT_MICRO)
