import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from stepcovnet import config, constants, datasets

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _first_batch(ds):
    return next(iter(ds.take(1)))  # type: ignore


def _get_one_audio_chart_pair(data_dir):
    """Return (audio_path, chart_path) for one pair in data_dir. Uses no private API."""
    for root, _, files in os.walk(data_dir):
        audio_ext = (".mp3", ".ogg", ".wav")
        chart_ext = ".txt"
        audio_files = [f for f in files if f.endswith(audio_ext)]
        chart_files = [f for f in files if f.endswith(chart_ext)]
        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching = [f for f in chart_files if f.startswith(stem)]
            if matching:
                return (
                    os.path.join(root, audio_file),
                    os.path.join(root, matching[0]),
                )
    return None, None


class DatasetsTest(unittest.TestCase):
    def test_create_dataset(self):
        ds = datasets.create_dataset(TEST_DATA_DIR)
        features, targets = _first_batch(ds)

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[2], 128)  # Mel bins
        self.assertEqual(features.shape[1], 11726)  # Time steps

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[2], 1)  # Channels
        self.assertEqual(targets.shape[1], 11726)  # Time steps

        self.assertEqual(int(np.sum(targets[0])), 384)

    def test_create_dataset_with_empty_directory_raises_error(self):
        for create_fn in (datasets.create_dataset, datasets.create_arrow_dataset):
            with self.subTest(create_fn=create_fn.__name__):
                with self.assertRaises(ValueError):
                    create_fn("")

    def test_create_arrow_dataset(self):
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR)
        features, targets = _first_batch(ds)

        self.assertEqual(features.shape[0], 1)  # Batch size
        self.assertEqual(features.shape[1], 384)  # Timings

        self.assertEqual(targets.shape[0], 1)  # Batch size
        self.assertEqual(targets.shape[1], 384)  # Arrows

        self.assertTrue(np.all(targets[0] > 0))

    def _assert_snippet_batch_structure(
        self, features, targets, min_batch_size=1, max_batch_size=1
    ):
        self.assertIn("timing_input", features)
        self.assertIn("snippet_input", features)
        times = features["timing_input"]
        snippets = features["snippet_input"]
        batch_dim = times.shape[0]
        self.assertGreaterEqual(batch_dim, min_batch_size)
        self.assertLessEqual(batch_dim, max_batch_size)
        self.assertEqual(snippets.shape[0], batch_dim)
        self.assertEqual(targets.shape[0], batch_dim)
        self.assertEqual(times.shape[2], 1)
        self.assertEqual(snippets.shape[2], 11)
        self.assertEqual(snippets.shape[3], 128)
        self.assertEqual(times.shape[1], snippets.shape[1])
        self.assertEqual(times.shape[1], targets.shape[1])

    def test_create_arrow_dataset_batch_size_two_timing_only(self):
        """Multi-sample batching without snippets uses (times [None,1], cols [None])."""
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR, batch_size=2)
        times, cols = _first_batch(ds)
        self.assertGreaterEqual(times.shape[0], 1)
        self.assertLessEqual(times.shape[0], 2)
        self.assertEqual(times.shape[0], cols.shape[0])
        self.assertEqual(times.shape[2], 1)
        self.assertEqual(times.shape[1], cols.shape[1])

    def test_create_arrow_dataset_with_audio_snippets(self):
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            snippet_half_frames=5,
        )
        features, targets = _first_batch(ds)
        self._assert_snippet_batch_structure(features, targets)
        times = features["timing_input"]
        seq_len = times.shape[1]
        self.assertGreater(int(seq_len), 0)
        self.assertLessEqual(seq_len, constants.MAX_STEPS)
        self.assertGreater(int(np.sum(targets[0].numpy() > 0)), 0)

    def test_create_arrow_dataset_with_audio_snippets_batch_size_two(self):
        """Multi-sample batching with snippets uses correct padded_shapes (dict + cols)."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            batch_size=2,
            snippet_half_frames=5,
        )
        features, targets = _first_batch(ds)
        self._assert_snippet_batch_structure(
            features, targets, min_batch_size=1, max_batch_size=2
        )

    def test_create_arrow_dataset_with_use_interval(self):
        """With use_interval=True and no snippets, batch is dict with timing_input and interval_input."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            use_interval=True,
        )
        features, targets = _first_batch(ds)
        self.assertIsInstance(features, dict)
        self.assertIn("timing_input", features)
        self.assertIn("interval_input", features)
        t = features["timing_input"]
        iv = features["interval_input"]
        self.assertEqual(t.shape, iv.shape)
        self.assertEqual(t.shape[2], 1)
        self.assertEqual(iv.shape[2], 1)
        self.assertEqual(t.shape[0], targets.shape[0])
        self.assertEqual(t.shape[1], targets.shape[1])

    def test_create_arrow_dataset_with_snippets_and_use_interval(self):
        """With use_interval=True and snippets, batch dict includes interval_input."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            snippet_half_frames=5,
            use_interval=True,
        )
        features, targets = _first_batch(ds)
        self._assert_snippet_batch_structure(features, targets)
        self.assertIn("interval_input", features)
        self.assertEqual(
            features["timing_input"].shape[:2],
            features["interval_input"].shape[:2],
        )

    def test_normalized_intervals_from_times(self):
        """normalized_intervals_from_times returns [0,1] scaled, step 0 is 0, intervals are diffs."""
        times = np.array([1.0, 2.5, 3.0, 10.0], dtype=np.float64)
        out = datasets.normalized_intervals_from_times(times)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), len(times))
        self.assertEqual(out[0], 0.0)
        # Raw intervals [0, 1.5, 0.5, 7.0]; max 7.0 -> normalized [0, 1.5/7, 0.5/7, 1.0]
        np.testing.assert_array_almost_equal(
            out, [0.0, 1.5 / 7.0, 0.5 / 7.0, 1.0], decimal=5
        )
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_normalized_intervals_from_times_empty(self):
        """normalized_intervals_from_times with empty times returns empty float32 array."""
        out = datasets.normalized_intervals_from_times(np.array([], dtype=np.float64))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (0,))

    def test_normalized_intervals_from_times_single(self):
        """normalized_intervals_from_times with one time returns [0.0] (step 0 has no prior)."""
        times = np.array([5.0])
        out = datasets.normalized_intervals_from_times(times)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, (1,))
        self.assertEqual(out[0], 0.0)

    def test_log_normalized_intervals_from_times(self):
        """log_normalized_intervals_from_times returns log(1+interval) normalized; step 0 is 0."""
        times = np.array([1.0, 2.5, 3.0, 10.0], dtype=np.float64)
        out = datasets.log_normalized_intervals_from_times(times)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), len(times))
        self.assertEqual(out[0], 0.0)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_next_interval_normalized_from_times(self):
        """next_interval_normalized_from_times: time-to-next per step; last step 0."""
        times = np.array([1.0, 2.0, 5.0], dtype=np.float64)
        out = datasets.next_interval_normalized_from_times(times)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[2], 0.0)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_step_index_normalized(self):
        """step_index_normalized returns [0,1] for n_steps; 0 and 1 for single step."""
        out = datasets.step_index_normalized(4)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_almost_equal(out, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        self.assertEqual(datasets.step_index_normalized(1)[0], 0.0)
        self.assertEqual(len(datasets.step_index_normalized(0)), 0)

    def test_beat_phase_from_times_bpm(self):
        """beat_phase_from_times_bpm returns phase in [0,1); empty when bpm<=0."""
        times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        out = datasets.beat_phase_from_times_bpm(times, 60.0)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 3)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLess(out.max(), 1.0)
        empty = datasets.beat_phase_from_times_bpm(times, 0.0)
        np.testing.assert_array_equal(empty, 0.0)

    def test_aux_interval_target_from_times(self):
        """aux_interval_target_from_times matches next_interval (last step 0)."""
        times = np.array([1.0, 2.0, 5.0], dtype=np.float64)
        out = datasets.aux_interval_target_from_times(times)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[2], 0.0)

    def test_create_arrow_dataset_interval_encoding_log(self):
        """With interval_encoding=log, batch dict has interval_log_input."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.LOG,
        )
        features, targets = _first_batch(ds)
        self.assertIn("interval_log_input", features)
        self.assertEqual(
            features["timing_input"].shape[:2],
            features["interval_log_input"].shape[:2],
        )

    def test_create_arrow_dataset_interval_encoding_multi(self):
        """With interval_encoding=multi, batch dict has interval_log_input and interval_next_input."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.MULTI,
        )
        features, targets = _first_batch(ds)
        self.assertIn("interval_log_input", features)
        self.assertIn("interval_next_input", features)
        self.assertEqual(
            features["interval_log_input"].shape,
            features["interval_next_input"].shape,
        )

    def test_create_arrow_dataset_use_step_index(self):
        """With use_step_index=True, batch dict has step_index_input."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            use_step_index=True,
        )
        features, targets = _first_batch(ds)
        self.assertIn("step_index_input", features)
        self.assertEqual(features["step_index_input"].shape[-1], 1)

    def test_create_arrow_dataset_use_beat_phase_chart(self):
        """With use_beat_phase=True, batch dict has beat_phase_input (BPM from chart)."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            use_beat_phase=True,
        )
        features, targets = _first_batch(ds)
        self.assertIn("beat_phase_input", features)
        self.assertEqual(features["beat_phase_input"].shape[-1], 1)

    def test_create_arrow_dataset_use_aux_interval_target(self):
        """With use_aux_interval_target=True, batch dict has aux_interval_target and aux_interval_mask."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            use_aux_interval_target=True,
        )
        features, targets = _first_batch(ds)
        self.assertIn("aux_interval_target", features)
        self.assertEqual(features["aux_interval_target"].shape[-1], 1)
        self.assertIn("aux_interval_mask", features)
        self.assertEqual(features["aux_interval_mask"].shape[-1], 1)
        # Last step should be masked (0); at least one step valid (1) when we have multiple steps
        mask = features["aux_interval_mask"]
        self.assertGreaterEqual(float(mask.shape[0]), 1)
        self.assertGreaterEqual(float(mask.shape[1]), 1)

    def test_create_arrow_dataset_timing_jitter_values_in_range(self):
        """With timing_jitter_sigma > 0, timing_input and step_index_input stay in [0, 1]."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            timing_jitter_sigma=0.05,
            use_step_index=True,
        )
        for batch_features, _ in ds.take(3):
            timing = batch_features["timing_input"].numpy()
            step_idx = batch_features["step_index_input"].numpy()
            self.assertTrue(
                np.all((timing >= 0) & (timing <= 1)),
                msg="timing_input should be in [0, 1] after jitter",
            )
            self.assertTrue(
                np.all((step_idx >= 0) & (step_idx <= 1)),
                msg="step_index_input should be in [0, 1] after jitter",
            )

    def test_create_arrow_dataset_timing_jitter_interval_shape_and_recomputed(self):
        """With use_interval=True, interval_input matches timing shape; callback unit test covers recomputation."""
        ds_no_jitter = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            batch_size=1,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            timing_jitter_sigma=0.0,
        )
        for batch_features, _ in ds_no_jitter.take(2):
            timing = batch_features["timing_input"].numpy()
            interval_batch = batch_features["interval_input"].numpy()
            self.assertEqual(
                timing.shape,
                interval_batch.shape,
                msg="interval_input should have same shape as timing_input",
            )
            for b in range(timing.shape[0]):
                t = timing[b].flatten()
                interval_flat = interval_batch[b].flatten()
                if len(t) == 0:
                    continue
                expected = datasets.normalized_intervals_from_times(t)
                np.testing.assert_allclose(
                    interval_flat,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg="interval_input should match recomputed from timing (no jitter)",
                )

    def test_create_arrow_dataset_timing_jitter_off_deterministic(self):
        """With timing_jitter_sigma=0, reading the same sample twice yields identical values."""
        ds = datasets.create_arrow_dataset(TEST_DATA_DIR, timing_jitter_sigma=0.0)
        batch1 = next(iter(ds))
        batch2 = next(iter(ds))
        feats1, cols1 = batch1
        feats2, cols2 = batch2
        np.testing.assert_array_almost_equal(
            feats1.numpy(),
            feats2.numpy(),
            decimal=5,
            err_msg="Same sample without jitter should match across iterators",
        )
        np.testing.assert_array_equal(cols1.numpy(), cols2.numpy())

    def test_create_arrow_dataset_timing_jitter_stochastic_per_epoch(self):
        """With timing_jitter_sigma > 0, same sample read twice yields different timing (uncached jitter)."""
        ds = datasets.create_arrow_dataset(
            TEST_DATA_DIR,
            timing_jitter_sigma=0.03,
        )
        batch1 = next(iter(ds))
        batch2 = next(iter(ds))
        times1 = batch1[0].numpy()
        times2 = batch2[0].numpy()
        self.assertFalse(
            np.allclose(times1, times2),
            msg="Jittered dataset should yield different timing when same sample is read twice",
        )

    def test_apply_timing_jitter_py_callback_recomputes_intervals_from_jittered_timing(
        self,
    ):
        """_apply_timing_jitter_py_callback with use_interval recomputes interval_input from jittered times."""
        np.random.seed(42)
        timing = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
        cols = np.array([1, 2, 1, 2, 1], dtype=np.int32)
        feats = {
            "timing_input": timing,
            "interval_input": np.array(
                [[0.0], [0.33], [0.33], [0.33], [0.33]], dtype=np.float32
            ),
        }
        out, out_cols = datasets._apply_timing_jitter_py_callback(
            feats,
            cols,
            sigma=0.02,
            use_dict=True,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
        )
        self.assertIsInstance(out, dict)
        self.assertIn("timing_input", out)
        self.assertIn("interval_input", out)
        t_flat = out["timing_input"].flatten()
        expected_interval = datasets.normalized_intervals_from_times(t_flat)
        np.testing.assert_allclose(
            out["interval_input"].flatten(),
            expected_interval,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(out_cols, cols)

    def test_load_arrow_pair_py_callback_returns_aux_interval_mask_when_requested(self):
        """_load_arrow_pair_py_callback with use_aux_interval_target=True returns 9-tuple including aux_interval_mask."""
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        ap = tf.constant("")
        cp = tf.constant(chart_path)
        result = datasets._load_arrow_pair_py_callback(
            ap,
            cp,
            snippet_half_frames=0,
            use_interval=False,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=True,
        )
        self.assertEqual(len(result), 9)
        times, _, _, _, _, _, _, aux_interval, aux_interval_mask = result
        self.assertEqual(aux_interval.shape, aux_interval_mask.shape)
        if len(times) > 0:
            self.assertEqual(aux_interval_mask[-1], 0.0)
            if len(times) > 1:
                self.assertEqual(aux_interval_mask[0], 1.0)

    def test_load_arrow_pair_py_callback_with_snippets(self):
        """_load_arrow_pair_py_callback with snippet_half_frames > 0 returns correct shapes."""
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        ap = tf.constant(audio_path)
        cp = tf.constant(chart_path)
        (
            times,
            intervals,
            _,
            snippets,
            cols,
            _,
            _,
            _,
            _,
        ) = datasets._load_arrow_pair_py_callback(
            ap,
            cp,
            snippet_half_frames=5,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=False,
        )  # type: ignore[arg-type]
        self.assertEqual(times.ndim, 1)
        self.assertEqual(intervals.ndim, 1)
        self.assertEqual(intervals.shape[0], len(times))
        self.assertEqual(snippets.shape[0], len(times))
        self.assertEqual(snippets.shape[1], 11)
        self.assertEqual(snippets.shape[2], constants.N_MELS)
        self.assertEqual(cols.shape[0], len(times))

    def test_normalize_onset_spectrogram(self):
        spec = np.random.randn(100, constants.N_MELS).astype(np.float64)
        out = datasets.normalize_onset_spectrogram(spec)
        self.assertEqual(out.shape, spec.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.mean(out, axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(np.std(out, axis=0), 1.0, atol=1e-5)

    def test_audio_to_spectrogram(self):
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        spec = datasets.audio_to_spectrogram(audio_path)
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[0], constants.N_MELS)
        self.assertGreater(int(spec.shape[1]), 0)

    def test_extract_snippets_from_spec(self):
        n_times = 5
        half_frames = 2
        n_frames = 2 * half_frames + 1
        time_steps = 500
        spec = np.random.randn(time_steps, constants.N_MELS).astype(np.float32)
        times_seconds = np.linspace(0.1, 2.0, n_times)
        snippets = datasets.extract_snippets_from_spec(
            spec, times_seconds, half_frames=half_frames
        )
        self.assertEqual(snippets.shape, (n_times, n_frames, constants.N_MELS))
        self.assertEqual(snippets.dtype, np.float32)

    def test_load_arrow_pair_py_callback_empty_chart(self):
        """_load_arrow_pair_py_callback with empty chart returns empty arrays."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = os.path.join(tmpdir, "empty_chart.txt")
            with open(chart_path, "w") as f:
                f.write("TITLE Empty\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
            ap = tf.constant(audio_path)
            cp = tf.constant(chart_path)
            (
                times,
                intervals,
                _,
                snippets,
                cols,
                _,
                _,
                _,
                _,
            ) = datasets._load_arrow_pair_py_callback(
                ap,
                cp,
                snippet_half_frames=5,
                use_interval=True,
                interval_encoding=config.IntervalEncoding.DEFAULT,
                use_step_index=False,
                use_beat_phase=False,
                use_aux_interval_target=False,
            )  # type: ignore[arg-type]
            self.assertEqual(len(times), 0)
            self.assertEqual(len(intervals), 0)
            self.assertEqual(snippets.shape[0], 0)
            self.assertEqual(snippets.shape[1], 11)
            self.assertEqual(snippets.shape[2], constants.N_MELS)
            self.assertEqual(len(cols), 0)

    def test_load_arrow_pair_py_callback_invalid_chart_raises(self):
        """_load_arrow_pair_py_callback with invalid base-4 in chart raises ValueError."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = os.path.join(tmpdir, "bad_chart.txt")
            with open(chart_path, "w") as f:
                f.write("TITLE Bad\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                f.write("4012 7.5\n")  # invalid base-4 digit '4'
            ap = tf.constant(audio_path)
            cp = tf.constant(chart_path)
            with self.assertRaises(ValueError):
                datasets._load_arrow_pair_py_callback(
                    ap,
                    cp,
                    snippet_half_frames=5,
                    use_interval=True,
                    interval_encoding=config.IntervalEncoding.DEFAULT,
                    use_step_index=False,
                    use_beat_phase=False,
                    use_aux_interval_target=False,
                )  # type: ignore[arg-type]

    def test_load_arrow_pair_py_callback_empty_arrows_line_raises(self):
        """_load_arrow_pair_py_callback with empty arrows line raises ValueError."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_path = os.path.join(tmpdir, "bad_chart_empty_arrows.txt")
            with open(chart_path, "w") as f:
                f.write("TITLE Bad\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                f.write(" 7.5\n")  # empty arrows -> ValueError
            ap = tf.constant(audio_path)
            cp = tf.constant(chart_path)
            with self.assertRaises(ValueError):
                datasets._load_arrow_pair_py_callback(
                    ap,
                    cp,
                    snippet_half_frames=5,
                    use_interval=True,
                    interval_encoding=config.IntervalEncoding.DEFAULT,
                    use_step_index=False,
                    use_beat_phase=False,
                    use_aux_interval_target=False,
                )  # type: ignore[arg-type]

    def test_create_dataset_use_gaussian_target(self):
        ds = datasets.create_dataset(
            TEST_DATA_DIR, use_gaussian_target=True, gaussian_sigma=1.5
        )
        features, targets = _first_batch(ds)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(targets.shape[0], 1)
        self.assertGreater(int(np.sum(targets[0].numpy() > 0)), 0)

    def test_create_dataset_use_gaussian_target_empty_chart(self):
        """Covers create_dataset with use_gaussian_target when chart has 0 steps."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_chart = os.path.join(tmpdir, pathlib.Path(audio_path).stem + ".txt")
            with open(empty_chart, "w") as f:
                f.write("TITLE Empty\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
            shutil.copy2(
                audio_path,
                os.path.join(tmpdir, os.path.basename(audio_path)),
            )
            ds = datasets.create_dataset(
                tmpdir, use_gaussian_target=True, gaussian_sigma=1.0
            )
            features, targets = _first_batch(ds)
            self.assertEqual(features.shape[0], 1)
            self.assertEqual(targets.shape[0], 1)
            self.assertEqual(int(np.sum(targets[0].numpy())), 0)

    def test_create_dataset_batch_size_two(self):
        ds = datasets.create_dataset(TEST_DATA_DIR, batch_size=2)
        features, targets = _first_batch(ds)
        self.assertGreaterEqual(features.shape[0], 1)
        self.assertLessEqual(features.shape[0], 2)
        self.assertEqual(features.shape[0], targets.shape[0])
        self.assertEqual(features.shape[2], 128)
        self.assertEqual(targets.shape[2], 1)

    def test_base4_to_int_valid(self):
        self.assertEqual(datasets._base4_to_int("0"), 0)
        self.assertEqual(datasets._base4_to_int("123"), 27)

    def test_base4_to_int_empty_raises(self):
        with self.assertRaises(ValueError):
            datasets._base4_to_int("")

    def test_base4_to_int_invalid_char_raises(self):
        with self.assertRaises(ValueError):
            datasets._base4_to_int("4012")

    def test_create_target(self):
        # frame_idx < spec_length (in range) and frame_idx >= spec_length (skipped)
        spec_length = 100
        times = np.array([0.5, 0.99, 2.0])  # frames 50, 99, 200 (200 out of range)
        cols = np.array([0, 0, 0], dtype=np.int32)
        target = datasets._create_target(times, cols, spec_length)
        self.assertEqual(target.shape, (spec_length, 1))
        self.assertEqual(target[50, 0], 1.0)
        self.assertEqual(target[99, 0], 1.0)
        self.assertEqual(int(np.sum(target)), 2)

    def test_create_target_gaussian_empty_times(self):
        out = datasets._create_target_gaussian(
            np.array([]), np.array([], dtype=np.int32), 100, sigma=1.0
        )
        self.assertEqual(out.shape, (100, 1))
        self.assertEqual(int(np.sum(out)), 0)

    def test_create_target_gaussian_skips_col_ge_n_target(self):
        # cols with value >= _N_TARGET (1) are skipped
        spec_length = 50
        times = np.array([0.0, 0.5])
        cols = np.array([0, 1], dtype=np.int32)  # col 1 >= _N_TARGET=1, skipped
        target = datasets._create_target_gaussian(times, cols, spec_length, sigma=1.0)
        self.assertEqual(target.shape, (spec_length, 1))
        self.assertGreater(float(target[0, 0]), 0)
        # Second onset (col 1) is skipped; only one peak from first onset
        self.assertLessEqual(int(np.sum(target > 0)), 7)  # kernel width ~3*sigma

    def test_temporal_augment_scipy_warp_longer(self):
        spec = np.random.randn(constants.N_MELS, 100).astype(np.float32)
        labels = np.zeros((100, 1), dtype=np.float32)
        with mock.patch("numpy.random.uniform", return_value=1.1):
            out_spec, out_labels = datasets._temporal_augment_scipy(spec, labels)
        self.assertEqual(out_spec.shape, spec.shape)
        self.assertEqual(out_labels.shape, labels.shape)

    def test_temporal_augment_scipy_warp_shorter(self):
        spec = np.random.randn(constants.N_MELS, 100).astype(np.float32)
        labels = np.zeros((100, 1), dtype=np.float32)
        with mock.patch("numpy.random.uniform", return_value=0.9):
            out_spec, out_labels = datasets._temporal_augment_scipy(spec, labels)
        self.assertEqual(out_spec.shape, spec.shape)
        self.assertEqual(out_labels.shape, labels.shape)

    def test_apply_spec_augment(self):
        spec = np.random.randn(constants.N_MELS, 200).astype(np.float32)
        out = datasets._apply_spec_augment(spec, F=10, T=20)
        self.assertEqual(out.shape, spec.shape)
        self.assertFalse(np.array_equal(out, spec))
        self.assertGreater(int(np.sum(out == 0)), 0)

    def test_load_and_preprocess_paths(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, target = datasets._load_and_preprocess_paths(
            audio_path, chart_path, use_gaussian_target=False, gaussian_sigma=1.0
        )
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertEqual(target.ndim, 2)
        self.assertEqual(target.shape[1], 1)
        self.assertGreater(int(np.sum(target > 0)), 0)

    def test_load_and_preprocess_paths_gaussian(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, target = datasets._load_and_preprocess_paths(
            audio_path, chart_path, use_gaussian_target=True, gaussian_sigma=1.5
        )
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertGreater(int(np.sum(target > 0)), 0)

    def test_augment_features_numpy_no_aug(self):
        features = np.random.randn(200, constants.N_MELS).astype(np.float32)
        target = np.zeros((200, 1), dtype=np.float32)
        out_f, out_t = datasets._augment_features_numpy(features, target, False, False)
        self.assertEqual(out_f.shape, features.shape)
        self.assertEqual(out_t.shape, target.shape)

    def test_augment_features_numpy_temporal_and_spec(self):
        features = np.random.randn(200, constants.N_MELS).astype(np.float32)
        target = np.zeros((200, 1), dtype=np.float32)
        with mock.patch("numpy.random.uniform", return_value=1.0):
            out_f, out_t = datasets._augment_features_numpy(
                features, target, True, True
            )
        self.assertEqual(out_f.shape, features.shape)
        self.assertEqual(out_t.shape, target.shape)

    def test_load_and_preprocess_py_callback(self):
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        ap = tf.constant(audio_path)
        cp = tf.constant(chart_path)
        features, target = datasets._load_and_preprocess_py_callback(
            ap,  # type: ignore[arg-type]
            cp,  # type: ignore[arg-type]
            False,
            1.0,
        )
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertEqual(target.shape[1], 1)

    def test_augment_py_callback(self):
        features = np.random.randn(100, constants.N_MELS).astype(np.float32)
        target = np.zeros((100, 1), dtype=np.float32)
        f_t = tf.constant(features)
        t_t = tf.constant(target)
        out_f, out_t = datasets._augment_py_callback(f_t, t_t, False, False)  # type: ignore[arg-type]
        self.assertEqual(out_f.shape, features.shape)
        self.assertEqual(out_t.shape, target.shape)

    def test_load_arrow_pair_py_callback_chart_only(self):
        """_load_arrow_pair_py_callback with snippet_half_frames=0 returns times, zeros intervals, empty snippets."""
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        ap = tf.constant("")
        cp = tf.constant(chart_path)
        (
            times,
            intervals,
            _,
            snippets,
            cols,
            _,
            _,
            _,
            _,
        ) = datasets._load_arrow_pair_py_callback(
            ap,
            cp,
            snippet_half_frames=0,
            use_interval=False,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=False,
        )  # type: ignore[arg-type]
        self.assertEqual(times.ndim, 1)
        self.assertEqual(cols.ndim, 1)
        self.assertEqual(len(times), len(cols))
        self.assertEqual(intervals.shape, times.shape)
        self.assertEqual(snippets.shape[1], 0)
        self.assertEqual(snippets.shape[2], constants.N_MELS)

    def test_parse_step_chart_difficulty_break(self):
        """Chart with second DIFFICULTY line stops parsing (break branch)."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multi_diff.txt")
            with open(path, "w") as f:
                f.write("TITLE X\nBPM 128.0\nNOTES\nDIFFICULTY Challenge\n")
                f.write("0000 0.5\n")
                f.write("DIFFICULTY Easy\n")
                f.write("0000 1.0\n")
            times, _ = datasets._parse_step_chart(path, binary_timings=False)
            self.assertEqual(len(times), 1)
            self.assertEqual(times[0], 0.5)

    def test_parse_step_chart_with_bpm(self):
        """_parse_step_chart_with_bpm returns (times, cols, bpm) with BPM from chart."""
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        times, cols, bpm = datasets._parse_step_chart_with_bpm(
            chart_path, binary_timings=False
        )
        self.assertEqual(len(times), len(cols))
        self.assertIsInstance(bpm, float)
        self.assertGreater(bpm, 0)

    def test_audio_to_spectrogram_resample_branch(self):
        """Cover the sr != _TARGET_SR resample path in audio_to_spectrogram."""
        audio_path, _ = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        assert audio_path is not None
        with mock.patch("stepcovnet.datasets.librosa") as mock_librosa:
            y_orig = np.random.randn(44100 * 2).astype(np.float32)
            mock_librosa.load.return_value = (y_orig, 22050)
            mock_librosa.resample.return_value = np.random.randn(44100 * 2).astype(
                np.float32
            )
            mock_librosa.power_to_db.side_effect = lambda x, **kw: x.astype(np.float64)
            mock_librosa.feature.melspectrogram.return_value = np.random.rand(
                constants.N_MELS, 200
            ).astype(np.float64)
            _ = datasets.audio_to_spectrogram(audio_path)
            mock_librosa.resample.assert_called_once()

    def test_load_and_preprocess_tf_map(self):
        """Cover _load_and_preprocess_tf_map (tf.data map helper)."""
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, target = datasets._load_and_preprocess_tf_map(
            tf.constant(audio_path),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            use_gaussian_target=False,
            gaussian_sigma=1.0,
        )
        self.assertEqual(features.shape[1], constants.N_MELS)
        self.assertEqual(target.shape[1], 1)

    def test_apply_augmentations_tf_map(self):
        """Cover _apply_augmentations_tf_map (tf.data map helper)."""
        features = np.random.randn(100, constants.N_MELS).astype(np.float32)
        target = np.zeros((100, 1), dtype=np.float32)
        aug_f, aug_t = datasets._apply_augmentations_tf_map(
            tf.constant(features),  # type: ignore[arg-type]
            tf.constant(target),  # type: ignore[arg-type]
            apply_temporal_augment=False,
            should_apply_spec_augment=False,
        )
        self.assertEqual(aug_f.shape[1], constants.N_MELS)
        self.assertEqual(aug_t.shape[1], 1)

    def test_process_pair_with_snippets_tf_map(self):
        """Cover _process_arrow_pair_tf_map with snippets (tf.data map helper)."""
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, _ = datasets._process_arrow_pair_tf_map(
            tf.constant(audio_path),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            snippet_half_frames=5,
            use_interval=False,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=False,
        )
        self.assertIn("timing_input", features)
        self.assertIn("snippet_input", features)
        assert isinstance(features, dict)
        self.assertEqual(features["timing_input"].shape[-1], 1)
        self.assertEqual(features["snippet_input"].shape[1], 11)
        self.assertEqual(features["snippet_input"].shape[2], constants.N_MELS)

    def test_process_pair_tf_map(self):
        """Cover _process_arrow_pair_tf_map without snippets (tf.data map helper)."""
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        times, cols = datasets._process_arrow_pair_tf_map(
            tf.constant(""),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            snippet_half_frames=0,
            use_interval=False,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=False,
        )
        assert isinstance(times, tf.Tensor)
        self.assertEqual(times.shape[-1], 1)
        self.assertEqual(len(times.shape), 2)
        self.assertEqual(len(cols.shape), 1)

    def test_process_arrow_pair_tf_map_interval_only(self):
        """Cover _process_arrow_pair_tf_map with use_interval=True, no snippets (dict with timing_input + interval_input)."""
        _, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(chart_path)
        assert chart_path is not None
        features, cols = datasets._process_arrow_pair_tf_map(
            tf.constant(""),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            snippet_half_frames=0,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=False,
        )
        self.assertIsInstance(features, dict)
        self.assertIn("timing_input", features)
        self.assertIn("interval_input", features)
        self.assertNotIn("snippet_input", features)
        self.assertEqual(features["timing_input"].shape[-1], 1)  # type: ignore
        self.assertEqual(features["interval_input"].shape[-1], 1)  # type: ignore
        self.assertEqual(len(cols.shape), 1)

    def test_process_arrow_pair_tf_map_snippets_and_interval(self):
        """Cover _process_arrow_pair_tf_map with use_snippets and use_interval (dict with all three inputs)."""
        audio_path, chart_path = _get_one_audio_chart_pair(TEST_DATA_DIR)
        self.assertIsNotNone(audio_path)
        self.assertIsNotNone(chart_path)
        assert audio_path is not None and chart_path is not None
        features, cols = datasets._process_arrow_pair_tf_map(
            tf.constant(audio_path),  # type: ignore[arg-type]
            tf.constant(chart_path),  # type: ignore[arg-type]
            snippet_half_frames=5,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.DEFAULT,
            use_step_index=False,
            use_beat_phase=False,
            use_aux_interval_target=False,
        )
        self.assertIsInstance(features, dict)
        self.assertIn("timing_input", features)
        self.assertIn("interval_input", features)
        self.assertIn("snippet_input", features)
        self.assertEqual(features["timing_input"].shape[-1], 1)  # type: ignore[union-attr]
        self.assertEqual(features["interval_input"].shape[-1], 1)  # type: ignore[union-attr]
        self.assertEqual(features["snippet_input"].shape[1], 11)  # type: ignore[union-attr]
        self.assertEqual(features["snippet_input"].shape[2], constants.N_MELS)  # type: ignore[union-attr]
        self.assertEqual(len(cols.shape), 1)


if __name__ == "__main__":
    unittest.main()
