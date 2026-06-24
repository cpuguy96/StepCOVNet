"""Tests for dataset_prep.training_loader (P9)."""

from __future__ import annotations

import contextlib
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet import datasets, pairing
from stepcovnet.dataset_prep import config, pipeline, training_index, training_loader
from stepcovnet.onset_events import charts

_FIXTURES_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "dataset_prep"
)


class TrainingLoaderTest(unittest.TestCase):
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

    def test_discover_training_rows_expands_multi_chart_sm(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            rows = training_loader.discover_training_rows(out_dir)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [row.chart_index for row in rows],
                [0, 1],
            )
            self.assertEqual(rows[1].difficulty, "challenge")
            self.assertTrue(pathlib.Path(rows[0].audio_path).is_file())
            self.assertTrue(pathlib.Path(rows[0].chart_json_path).is_file())

    def test_list_training_samples_prefers_prepared_layout(self):
        with self._prepared_output("itl_challenge_ssc") as out_dir:
            samples = pairing.list_training_samples(out_dir)
            self.assertEqual(len(samples), 1)
            audio_path, chart_path, chart_index = samples[0]
            self.assertEqual(chart_index, 0)
            self.assertTrue(chart_path.endswith(".chart.json"))
            times = charts.load_onset_times(chart_path, chart_index=chart_index)
            self.assertIsNotNone(times)
            assert times is not None
            np.testing.assert_allclose(times, [0.0])

    def test_parse_step_chart_reads_chart_json_block(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            rows = training_loader.discover_training_rows(out_dir)
            challenge = rows[1]
            times, cols = datasets._parse_step_chart(
                challenge.chart_json_path,
                binary_timings=False,
                chart_index=challenge.chart_index,
            )
            expected_times = training_loader.load_chart_times_sec(
                challenge.chart_json_path,
                challenge.chart_index,
            )
            np.testing.assert_allclose(times, expected_times)
            np.testing.assert_array_equal(cols, np.asarray([16], dtype=np.int32))

    def test_filter_rows_by_step_cap(self):
        row = training_loader.TrainingChartRow(
            normalized_bundle="b",
            normalized_id="s",
            chart_index=0,
            output_relpath="b/s",
            chart_json_path="/tmp/s.chart.json",
            audio_path="/tmp/s.ogg",
            difficulty="hard",
            meter=8,
            num_steps=10,
        )
        kept = training_loader.filter_rows_by_step_cap([row], max_steps=10)
        self.assertEqual(len(kept), 1)
        self.assertEqual(
            training_loader.filter_rows_by_step_cap([row], max_steps=9),
            [],
        )

    def test_list_dense_onset_samples_matches_pairing_on_prepared_output(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            dense_samples = datasets.list_dense_onset_samples(out_dir)
            pairing_samples = pairing.list_training_samples(out_dir)
            self.assertEqual(dense_samples, pairing_samples)
            self.assertEqual(len(dense_samples), 2)

    def test_list_dense_onset_samples_respects_train_split(self):
        with self._prepared_output("vocaloid_multi_sm") as out_dir:
            index_path = training_index.save_training_index(
                training_index.build_training_index(
                    out_dir,
                    val_fraction=0.0,
                    seed=11,
                )
            )
            train_samples = datasets.list_dense_onset_samples(
                str(index_path),
                split="train",
            )
            val_samples = datasets.list_dense_onset_samples(
                str(index_path),
                split="val",
            )
            self.assertEqual(len(train_samples), 2)
            self.assertEqual(len(val_samples), 0)

    def test_create_dataset_from_training_index_yields_batch(self):
        with self._prepared_output("itl_challenge_ssc") as out_dir:
            index_path = training_index.save_training_index(
                training_index.build_training_index(
                    out_dir,
                    val_fraction=0.0,
                    seed=1,
                )
            )
            stub_features = np.zeros((10, 128), dtype=np.float32)
            with mock.patch.object(
                datasets,
                "load_onset_features",
                return_value=stub_features,
            ):
                ds = datasets.create_dataset(
                    str(index_path),
                    split="train",
                    data_root=str(out_dir),
                )
                features, targets = next(iter(ds.take(1)))
            self.assertEqual(features.shape[0], 1)
            self.assertEqual(features.shape[2], 128)
            self.assertEqual(targets.shape[0], 1)
            self.assertEqual(targets.shape[2], 1)
            self.assertEqual(features.shape[1], targets.shape[1])


if __name__ == "__main__":
    unittest.main()
