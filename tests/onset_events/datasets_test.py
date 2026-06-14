import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.io.wavfile
import tensorflow as tf

from stepcovnet import constants
from stepcovnet.onset_events import audio
from stepcovnet.onset_events import charts
from stepcovnet.onset_events import datasets
from stepcovnet.onset_events import preprocess
from stepcovnet.onset_events import targets


def _write_chart(path: str, step_lines: list[str]) -> None:
    with open(path, "w") as chart_file:
        chart_file.write("TITLE Test\nBPM 120.0\nNOTES\nDIFFICULTY Challenge\n")
        chart_file.write("".join(step_lines))


def _write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    int_samples = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, int_samples)


def _write_valid_pair(
    directory: str,
    stem: str,
    *,
    step_lines: list[str] | None = None,
    duration_sec: float = 0.2,
) -> tuple[str, str]:
    if step_lines is None:
        step_lines = ["1000 0.05\n", "0100 0.10\n", "0010 0.15\n"]
    audio_path = os.path.join(directory, f"{stem}.wav")
    chart_path = os.path.join(directory, f"{stem}.txt")
    sr = constants.TARGET_SR
    n = max(1, int(duration_sec * sr))
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float64)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    _write_wav(audio_path, samples, sr)
    _write_chart(chart_path, step_lines)
    return audio_path, chart_path


def _first_batch(data_dir: str, **kwargs) -> dict[str, np.ndarray]:
    ds = datasets.create_onset_event_dataset(data_dir, **kwargs)
    for batch in ds:
        return {key: value.numpy() for key, value in batch.items()}
    raise AssertionError("dataset yielded no batches")


class DatasetsTest(unittest.TestCase):
    def test_onset_event_dataset_config_defaults(self):
        config = datasets.OnsetEventDatasetConfig()
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.max_audio_seconds, 300.0)
        self.assertEqual(config.n_max_onsets, 1024)
        self.assertEqual(config.max_steps_per_chart, 2048)
        self.assertEqual(config.target_sample_rate, 44100)

    def test_first_valid_pair_returns_first_kept_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = _write_valid_pair(tmpdir, "aaa")
            _write_valid_pair(tmpdir, "bbb")
            self.assertEqual(
                datasets.first_valid_pair(tmpdir),
                first,
            )

    def test_create_dataset_from_pairs_single_song(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pair = _write_valid_pair(tmpdir, "only")
            ds = datasets.create_onset_event_dataset_from_pairs(
                [pair],
                max_audio_seconds=0.25,
                n_max_onsets=8,
            )
            batches = list(ds.take(2))
            self.assertEqual(len(batches), 1)
            self.assertEqual(
                batches[0]["audio"].shape, (1, int(0.25 * constants.TARGET_SR))
            )

    def test_filter_valid_pairs_skips_missing_and_over_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_audio, valid_chart = _write_valid_pair(tmpdir, "good")
            over_cap_chart = os.path.join(tmpdir, "big.txt")
            step_lines = [f"1000 {i * 0.01}\n" for i in range(1025)]
            _write_chart(over_cap_chart, step_lines)
            missing_audio = os.path.join(tmpdir, "ghost.wav")

            pairs = [
                (valid_audio, valid_chart),
                (missing_audio, valid_chart),
                (valid_audio, over_cap_chart),
            ]
            filtered = datasets._filter_valid_pairs(pairs, max_steps_per_chart=1024)
            self.assertEqual(filtered, [(valid_audio, valid_chart)])

    def test_create_dataset_yields_expected_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_valid_pair(tmpdir, "song")
            batch = _first_batch(tmpdir)

        self.assertEqual(
            set(batch.keys()),
            {
                "audio",
                "audio_length",
                "features",
                "gt_times",
                "gt_mask",
                "duration",
            },
        )
        self.assertEqual(batch["audio"].shape, (1, audio.DEFAULT_MAX_SAMPLES))
        self.assertEqual(batch["audio_length"].shape, (1,))
        self.assertEqual(batch["gt_times"].shape, (1, targets.N_MAX_ONSETS))
        self.assertEqual(batch["gt_mask"].shape, (1, targets.N_MAX_ONSETS))
        self.assertEqual(batch["duration"].shape, (1,))

        audio_length = int(batch["audio_length"][0])
        duration = float(batch["duration"][0])
        self.assertAlmostEqual(duration, audio_length / constants.TARGET_SR, places=5)
        self.assertGreater(audio_length, 0)
        self.assertAlmostEqual(
            float(np.max(np.abs(batch["audio"][0, :audio_length]))), 1.0, places=4
        )
        np.testing.assert_allclose(batch["audio"][0, audio_length:], 0.0)

        mask = batch["gt_mask"][0]
        gt_times = batch["gt_times"][0]
        active = int(mask.sum())
        self.assertEqual(active, 3)
        np.testing.assert_allclose(gt_times[:active], [0.05, 0.10, 0.15])
        np.testing.assert_allclose(mask[active:], 0.0)

    def test_create_dataset_truncates_audio_and_clips_gt_times(self):
        sr = constants.TARGET_SR
        max_seconds = 0.05
        max_samples = audio.max_samples_for_cap(max_seconds, sr)
        with tempfile.TemporaryDirectory() as tmpdir:
            step_lines = ["1000 0.01\n", "0100 0.03\n", "0010 0.06\n", "0001 0.10\n"]
            _write_valid_pair(tmpdir, "clip", step_lines=step_lines, duration_sec=0.2)
            batch = _first_batch(
                tmpdir,
                max_audio_seconds=max_seconds,
                n_max_onsets=8,
            )

        audio_length = int(batch["audio_length"][0])
        self.assertEqual(audio_length, max_samples)
        duration = float(batch["duration"][0])
        self.assertAlmostEqual(duration, max_seconds, places=5)

        mask = batch["gt_mask"][0]
        active = int(mask.sum())
        self.assertEqual(active, 2)
        np.testing.assert_allclose(batch["gt_times"][0, :active], [0.01, 0.03])

    def test_create_dataset_skips_over_cap_chart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_valid_pair(tmpdir, "ok")
            over_path = os.path.join(tmpdir, "big.txt")
            step_lines = [f"1000 {i * 0.01}\n" for i in range(1025)]
            _write_chart(over_path, step_lines)
            big_audio = os.path.join(tmpdir, "big.wav")
            _write_wav(
                big_audio,
                np.array([0.5, -0.5], dtype=np.float32),
                constants.TARGET_SR,
            )

            ds = datasets.create_onset_event_dataset(tmpdir)
            batches = list(ds.take(2))
            self.assertEqual(len(batches), 1)
            self.assertEqual(int(np.sum(batches[0]["gt_mask"][0])), 3)

    def test_create_dataset_raises_when_no_valid_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                datasets.create_onset_event_dataset(tmpdir)

    def test_create_dataset_shuffle_uses_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx in range(3):
                _write_valid_pair(
                    tmpdir,
                    f"song{idx}",
                    step_lines=[f"1000 {idx + 0.1}\n"],
                )

            ds_a = datasets.create_onset_event_dataset(tmpdir, shuffle=True, seed=7)
            ds_b = datasets.create_onset_event_dataset(tmpdir, shuffle=True, seed=7)
            order_a = [float(b["gt_times"][0, 0]) for b in ds_a.take(3)]
            order_b = [float(b["gt_times"][0, 0]) for b in ds_b.take(3)]
            self.assertEqual(order_a, order_b)

    def test_load_onset_event_sample_direct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, chart_path = _write_valid_pair(tmpdir, "direct")
            max_samples = int(0.2 * constants.TARGET_SR)
            sample = datasets._load_onset_event_sample(
                audio_path,
                chart_path,
                target_sample_rate=constants.TARGET_SR,
                max_samples=max_samples,
                max_audio_seconds=0.2,
                n_max_onsets=4,
                max_steps_per_chart=charts.MAX_STEPS_PER_CHART,
            )

        audio_arr, audio_length, gt_times, gt_mask, duration, features = sample
        self.assertEqual(audio_arr.shape, (max_samples,))
        self.assertEqual(features.shape[0], 20)
        self.assertEqual(audio_length.dtype, np.int32)
        self.assertLessEqual(int(audio_length), max_samples)
        np.testing.assert_allclose(gt_times[:3], [0.05, 0.10, 0.15])
        np.testing.assert_allclose(gt_mask, [1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(float(duration), int(audio_length) / constants.TARGET_SR)

    def test_load_onset_event_sample_raises_when_chart_over_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, chart_path = _write_valid_pair(
                tmpdir,
                "cap",
                step_lines=[f"1000 {i * 0.01}\n" for i in range(1025)],
            )
            with self.assertRaises(ValueError):
                datasets._load_onset_event_sample(
                    audio_path,
                    chart_path,
                    target_sample_rate=constants.TARGET_SR,
                    max_samples=100,
                    max_audio_seconds=0.25,
                    n_max_onsets=4,
                    max_steps_per_chart=1024,
                )

    def test_map_pair_to_sample_sets_static_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, chart_path = _write_valid_pair(tmpdir, "map")
            config = datasets.OnsetEventDatasetConfig(
                n_max_onsets=8,
                max_audio_seconds=0.05,
            )
            max_samples = datasets._max_samples(config)
            sample = datasets._map_pair_to_sample(
                tf.constant(audio_path),
                tf.constant(chart_path),
                config,
                max_samples,
            )

        self.assertEqual(sample["audio"].shape.as_list(), [max_samples])
        self.assertEqual(sample["audio_length"].shape.as_list(), [])
        self.assertEqual(sample["gt_times"].shape.as_list(), [8])
        self.assertEqual(sample["gt_mask"].shape.as_list(), [8])
        self.assertEqual(sample["duration"].shape.as_list(), [])
        self.assertEqual(sample["features"].shape.as_list(), [5, 1])

    def test_load_py_callback_decodes_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, chart_path = _write_valid_pair(tmpdir, "py")
            result = datasets._load_onset_event_py_callback(
                tf.constant(audio_path),
                tf.constant(chart_path),
                constants.TARGET_SR,
                500,
                0.25,
                4,
                charts.MAX_STEPS_PER_CHART,
                preprocess.FRONTEND_CONV1D,
                "",
                "",
            )
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0].shape, (500,))

    def test_max_samples_helper(self):
        config = datasets.OnsetEventDatasetConfig(
            max_audio_seconds=10.0,
            target_sample_rate=22050,
        )
        self.assertEqual(datasets._max_samples(config), 220500)

    def test_create_dataset_batch_size_two(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_valid_pair(tmpdir, "a", step_lines=["1000 0.1\n"])
            _write_valid_pair(tmpdir, "b", step_lines=["1000 0.2\n"])
            batch = _first_batch(tmpdir, batch_size=2, n_max_onsets=4)

        self.assertEqual(batch["audio"].shape[0], 2)
        self.assertEqual(batch["gt_times"].shape, (2, 4))

    def test_filter_valid_pairs_load_onset_times_none_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, chart_path = _write_valid_pair(tmpdir, "edge")
            pairs = [(audio_path, chart_path)]
            with mock.patch.object(
                charts,
                "load_onset_times",
                return_value=None,
                autospec=True,
            ):
                filtered = datasets._filter_valid_pairs(pairs, max_steps_per_chart=1024)
            self.assertEqual(filtered, [])
