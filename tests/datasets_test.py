import pathlib
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from stepcovnet import config, constants, datasets


def _write_dummy_song(
    root: pathlib.Path,
    stem: str,
    *,
    duration_sec: float = 1.5,
    bpm: float = 120.0,
) -> tuple[pathlib.Path, pathlib.Path]:
    song_dir = root / stem
    song_dir.mkdir(parents=True, exist_ok=True)
    audio_path = song_dir / f"{stem}.wav"
    sr = constants.TARGET_SR
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False)
    wav = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(audio_path), wav, sr)
    chart_path = song_dir / f"{stem}.txt"
    chart_path.write_text(
        f"TITLE {stem}\n"
        f"BPM {bpm}\n"
        "NOTES\n"
        "DIFFICULTY Challenge\n"
        "1000 0.25\n"
        "0100 0.50\n"
        "2000 0.75\n"
        "3000 1.00\n",
        encoding="utf-8",
    )
    return audio_path, chart_path


class DatasetsHelpersTest(unittest.TestCase):
    def test_base4_to_int(self):
        self.assertEqual(datasets._base4_to_int("0000"), 0)
        self.assertEqual(datasets._base4_to_int("0001"), 1)
        self.assertEqual(datasets._base4_to_int("1000"), 64)
        with self.assertRaises(ValueError):
            datasets._base4_to_int("")
        with self.assertRaises(ValueError):
            datasets._base4_to_int("1004")

    def test_interval_and_phase_helpers(self):
        times = np.array([0.5, 1.0, 1.25, 2.0], dtype=np.float64)
        iv = datasets.normalized_intervals_from_times(times)
        self.assertEqual(iv.shape, (4,))
        self.assertAlmostEqual(float(iv[0]), 0.0)
        self.assertAlmostEqual(float(iv[-1]), 1.0, places=5)

        log_iv = datasets.log_normalized_intervals_from_times(times)
        self.assertEqual(log_iv.shape, (4,))
        self.assertAlmostEqual(float(log_iv[0]), 0.0)

        nxt = datasets.next_interval_normalized_from_times(times)
        self.assertEqual(nxt.shape, (4,))
        self.assertAlmostEqual(float(nxt[-1]), 0.0)

        idx = datasets.step_index_normalized(4)
        np.testing.assert_allclose(idx, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], atol=1e-5)

        phase = datasets.beat_phase_from_times_bpm(times, 120.0)
        np.testing.assert_allclose(phase, [0.0, 0.0, 0.5, 0.0], atol=1e-5)

    def test_create_target_and_gaussian(self):
        times = np.array([0.1, 0.5], dtype=np.float64)
        cols = np.array([0, 0], dtype=np.int32)
        binary = datasets._create_target(times, cols, spec_length=100)
        self.assertEqual(binary.shape, (100, 1))
        self.assertEqual(float(binary.sum()), 2.0)

        gauss = datasets._create_target_gaussian(
            times, cols, spec_length=100, sigma=1.5
        )
        self.assertEqual(gauss.shape, (100, 1))
        self.assertAlmostEqual(float(gauss.max()), 1.0, places=5)
        self.assertGreater(float(gauss.sum()), 2.0)

    def test_density_conditioning_helpers(self):
        d = datasets.onset_density_scalar(20, 10.0)
        self.assertGreater(d, 0.0)
        self.assertLessEqual(d, 1.0)
        feats = np.zeros((10, 8), dtype=np.float32)
        out = datasets.append_density_channel(feats, d)
        self.assertEqual(out.shape, (10, 9))
        np.testing.assert_allclose(out[:, -1], d)

    def test_select_song_pairs_reproducible(self):
        pairs = [(f"a{i}.ogg", f"a{i}.txt") for i in range(10)]
        s1 = datasets.select_song_pairs(pairs, max_songs=4, seed=7)
        s2 = datasets.select_song_pairs(pairs, max_songs=4, seed=7)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1), 4)

    def test_timing_jitter_preserves_order(self):
        feats = {
            "timing_input": np.array([[0.1], [0.4], [0.7], [0.9]], dtype=np.float32),
            "interval_input": np.array([[0.0], [1.0], [1.0], [0.6]], dtype=np.float32),
        }
        cols = np.array([1, 2, 3, 4], dtype=np.int32)
        out, out_cols = datasets._apply_timing_jitter_py_callback(
            feats,
            cols,
            sigma=0.05,
            use_dict=True,
            use_interval=True,
            interval_encoding=config.IntervalEncoding.DEFAULT,
        )
        assert isinstance(out, dict)
        t = out["timing_input"].flatten()
        self.assertTrue(np.all(np.diff(t) >= 0.0))
        np.testing.assert_array_equal(out_cols, cols)


class PyTorchDataLoadersTest(unittest.TestCase):
    def test_create_dataset_full_and_windowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            _write_dummy_song(root, "song_a", duration_sec=1.2)
            _write_dummy_song(root, "song_b", duration_sec=1.6)

            loader = datasets.create_dataset(str(root), batch_size=2)
            x, y = next(iter(loader))
            self.assertIsInstance(x, torch.Tensor)
            self.assertEqual(x.ndim, 3)
            self.assertEqual(x.shape[0], 2)
            self.assertEqual(x.shape[2], constants.N_MELS)
            self.assertEqual(y.shape[:2], x.shape[:2])
            self.assertEqual(y.shape[2], 1)

            win_loader = datasets.create_dataset(
                str(root),
                batch_size=2,
                window_frames=32,
                windows_per_song=2,
                apply_temporal_augment=True,
                should_apply_spec_augment=True,
            )
            wx, wy = next(iter(win_loader))
            self.assertEqual(tuple(wx.shape), (2, 32, constants.N_MELS))
            self.assertEqual(tuple(wy.shape), (2, 32, 1))

    def test_create_arrow_dataset_plain_and_rich_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            _write_dummy_song(root, "song_a", duration_sec=1.2)
            _write_dummy_song(root, "song_b", duration_sec=1.5)

            plain_loader = datasets.create_arrow_dataset(str(root), batch_size=2)
            feats, cols = next(iter(plain_loader))
            self.assertIsInstance(feats, torch.Tensor)
            self.assertEqual(tuple(feats.shape), (2, 4, 1))
            self.assertEqual(tuple(cols.shape), (2, 4))

            rich_loader = datasets.create_arrow_dataset(
                str(root),
                batch_size=2,
                snippet_half_frames=2,
                use_interval=True,
                interval_encoding=config.IntervalEncoding.MULTI,
                use_step_index=True,
                use_beat_phase=True,
                use_aux_interval_target=True,
                timing_jitter_sigma=0.01,
            )
            r_feats, r_cols = next(iter(rich_loader))
            self.assertIsInstance(r_feats, dict)
            for key in (
                "timing_input",
                "interval_log_input",
                "interval_next_input",
                "step_index_input",
                "beat_phase_input",
                "snippet_input",
                "aux_interval_target",
                "aux_interval_mask",
            ):
                self.assertIn(key, r_feats)
            self.assertEqual(
                tuple(r_feats["snippet_input"].shape), (2, 4, 5, constants.N_MELS)
            )
            self.assertEqual(tuple(r_cols.shape), (2, 4))

    def test_spec_augment_handles_short_spectrograms(self):
        short_spec = np.ones((8, 12), dtype=np.float32)
        for _ in range(50):
            aug = datasets._apply_spec_augment(
                short_spec, max_freq_mask=32, max_time_mask=50
            )
            self.assertEqual(aug.shape, short_spec.shape)


if __name__ == "__main__":
    unittest.main()
