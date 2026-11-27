"""Tests for Beat This! / librosa beat tracking and ITGPT-MERT audio generation."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from stepcovnet import beat_tracking, constants, generator, models
from stepcovnet.pytorch.itgpt import models as itgpt_models
from stepcovnet.pytorch.itgpt import trainers as itgpt_trainers

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import generate as generate_script  # noqa: E402


class BeatGridTest(unittest.TestCase):
    def test_beat_grid_properties_and_validation(self) -> None:
        beats = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        grid = beat_tracking.BeatGrid(
            beat_times=beats,
            downbeat_times=np.array([0.5], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        self.assertEqual(grid.n_beats, 4)
        with_end = grid.beat_times_with_end
        self.assertEqual(with_end.size, 5)
        self.assertAlmostEqual(float(with_end[-1]), 2.5, places=6)
        np.testing.assert_allclose(grid.per_beat_bpm(), np.full(4, 120.0), atol=1e-3)

        with self.assertRaises(ValueError):
            beat_tracking.BeatGrid(
                beat_times=np.array([0.5]),
                downbeat_times=np.zeros(0),
                bpm=120.0,
                tracker="librosa",
            )
        with self.assertRaises(ValueError):
            beat_tracking.BeatGrid(
                beat_times=np.array([1.0, 1.0]),
                downbeat_times=np.zeros(0),
                bpm=120.0,
                tracker="librosa",
            )
        with self.assertRaises(ValueError):
            beat_tracking.BeatGrid(
                beat_times=np.array([0.5, 1.0]),
                downbeat_times=np.zeros(0),
                bpm=0.0,
                tracker="librosa",
            )


class BeatTrackerBackendsTest(unittest.TestCase):
    def test_estimate_beats_beat_this_mocked(self) -> None:
        fake_mod = types.ModuleType("beat_this.inference")

        class _FakeFile2Beats:
            def __init__(self, *, checkpoint_path: str, device: str, dbn: bool) -> None:
                self.checkpoint_path = checkpoint_path
                self.device = device
                self.dbn = dbn

            def __call__(self, audio_path: str):
                self.last_audio = audio_path
                return (
                    np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
                    np.array([0.5, 2.5], dtype=np.float64),
                )

        fake_mod.File2Beats = _FakeFile2Beats  # type: ignore[attr-defined]
        with mock.patch.object(
            beat_tracking.importlib, "import_module", return_value=fake_mod
        ):
            grid = beat_tracking.estimate_beats_beat_this("song.ogg", device="cpu")
        self.assertEqual(grid.tracker, beat_tracking.TRACKER_BEAT_THIS)
        self.assertAlmostEqual(grid.bpm, 120.0, places=4)
        self.assertEqual(grid.n_beats, 4)
        self.assertEqual(grid.downbeat_times.size, 2)

    def test_estimate_beats_beat_this_missing_dependency(self) -> None:
        with (
            mock.patch.object(
                beat_tracking.importlib,
                "import_module",
                side_effect=ImportError("No module named 'beat_this'"),
            ),
            self.assertRaises(ImportError) as ctx,
        ):
            beat_tracking.estimate_beats_beat_this("song.ogg", device="cpu")
        self.assertIn("beat_this", str(ctx.exception))

    def test_estimate_beats_librosa_and_auto_fallback(self) -> None:
        fake_beats = np.array([0.4, 0.8, 1.2, 1.6], dtype=np.float64)
        with (
            mock.patch.object(
                beat_tracking.librosa,
                "load",
                return_value=(np.zeros(44100, dtype=np.float32), constants.TARGET_SR),
            ),
            mock.patch.object(
                beat_tracking.librosa.beat,
                "beat_track",
                return_value=(np.array([150.0]), fake_beats),
            ),
            mock.patch.object(beat_tracking, "beat_this_available", return_value=False),
        ):
            grid = beat_tracking.estimate_beats(
                "song.ogg", tracker=beat_tracking.TRACKER_AUTO
            )
        self.assertEqual(grid.tracker, beat_tracking.TRACKER_LIBROSA)
        self.assertAlmostEqual(grid.bpm, 150.0, places=4)
        self.assertEqual(grid.n_beats, 4)

        with self.assertRaises(ValueError):
            beat_tracking.estimate_beats("song.ogg", tracker="unknown_tracker")

    def test_snap_beats_to_onsets_aligns_constant_phase_offset(self) -> None:
        sr = constants.TARGET_SR
        hop_length = 256
        n_env = 400
        env_times = beat_tracking.librosa.frames_to_time(
            np.arange(n_env), sr=sr, hop_length=hop_length
        )
        # Put sharp onset-strength spikes at exact 0.500s intervals
        true_beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
        onset_env = np.zeros(n_env, dtype=np.float32)
        for tb in true_beats:
            onset_env += np.exp(-0.5 * ((env_times - tb) / 0.006) ** 2).astype(
                np.float32
            )

        # Start with a +14 ms late beat grid (typical neural beat tracker bias)
        late_grid = beat_tracking.BeatGrid(
            beat_times=true_beats + 0.014,
            downbeat_times=np.array([0.514], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        with (
            mock.patch.object(
                beat_tracking.librosa,
                "load",
                return_value=(np.zeros(sr * 3, dtype=np.float32), sr),
            ),
            mock.patch.object(
                beat_tracking.librosa.onset,
                "onset_strength",
                return_value=onset_env,
            ),
        ):
            snapped = beat_tracking.snap_beats_to_onsets(
                late_grid, "song.ogg", max_shift_ms=25.0, step_ms=1.0
            )
        residual_ms = float(np.mean(np.abs(snapped.beat_times - true_beats)) * 1000.0)
        self.assertLess(residual_ms, 3.0)


class ItgptRawAudioGenerationTest(unittest.TestCase):
    def test_decode_itgpt_slots_to_times(self) -> None:
        grid = beat_tracking.BeatGrid(
            beat_times=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            downbeat_times=np.zeros(0),
            bpm=120.0,
            tracker="beat_this",
        )
        probs = np.zeros((3, 48), dtype=np.float32)
        probs[0, 0] = 0.9  # 0.0 + (0.5/48)*0.5
        probs[1, 24] = 0.8  # 0.5 + (24.5/48)*0.5
        times = beat_tracking.decode_itgpt_slots_to_times(probs, grid, threshold=0.5)
        self.assertEqual(times.size, 2)
        self.assertAlmostEqual(float(times[0]), (0.5 / 48.0) * 0.5, places=6)
        self.assertAlmostEqual(float(times[1]), 0.5 + (24.5 / 48.0) * 0.5, places=6)

    def test_generate_output_data_with_itgpt_mert_and_beat_this(self) -> None:
        itgpt_model = itgpt_models.build_itgpt_placement_model(
            d_model=32,
            n_heads=4,
            n_enc_layers=1,
            dropout_rate=0.0,
            max_beats=64,
            audio_feature_type="mert",
            n_frames_per_beat=24,
            feature_dim=16,
            difficulty_source="density",
            beat_aux_head=True,
        )
        arrow_model = models.build_arrow_model()
        fake_grid = beat_tracking.BeatGrid(
            beat_times=np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            downbeat_times=np.array([0.5], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        fake_mert = np.zeros((300, 16), dtype=np.float32)
        fake_probs = np.zeros((4, 48), dtype=np.float32)
        fake_probs[:, 0] = 0.9  # 4 quarter-note onsets

        with (
            mock.patch.object(
                beat_tracking, "estimate_beats", return_value=fake_grid
            ) as mock_est,
            mock.patch.object(
                beat_tracking,
                "_load_or_extract_mert_for_audio",
                return_value=fake_mert,
            ),
            mock.patch(
                "stepcovnet.pytorch.itgpt.evaluation.predict_chart_slots",
                return_value=fake_probs,
            ),
        ):
            out = generator.generate_output_data(
                audio_path="song.ogg",
                song_title="ITGPT BeatThis Song",
                bpm=None,
                onset_model=itgpt_model,
                arrow_model=arrow_model,
                beat_tracker=beat_tracking.TRACKER_BEAT_THIS,
                target_density=1.5,
            )
        mock_est.assert_called_once()
        self.assertEqual(out.title, "ITGPT BeatThis Song")
        self.assertEqual(out.bpm, 120)
        self.assertIn("Challenge", out.notes)

    def test_generate_script_cli_and_itgpt_checkpoint_detection(self) -> None:
        args = generate_script.PARSER.parse_args(
            [
                "--audio_path",
                "song.ogg",
                "--song_title",
                "Demo",
                "--output_file",
                "out.txt",
                "--beat_tracker",
                "beat_this",
                "--snap_beats",
                "--target_density",
                "2.5",
                "--threshold",
                "0.55",
            ]
        )
        self.assertEqual(args.beat_tracker, "beat_this")
        self.assertTrue(args.snap_beats)
        self.assertAlmostEqual(args.target_density, 2.5)
        self.assertAlmostEqual(args.threshold, 0.55)

        with tempfile.TemporaryDirectory() as tmpdir:
            itgpt_model = itgpt_models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                n_enc_layers=1,
                dropout_rate=0.0,
                max_beats=64,
                audio_feature_type="mert",
                n_frames_per_beat=24,
                feature_dim=16,
            )
            ckpt_path = pathlib.Path(tmpdir) / "itgpt.pt"
            itgpt_trainers.save_checkpoint(itgpt_model, ckpt_path)
            self.assertTrue(generate_script._is_itgpt_checkpoint(str(ckpt_path)))
            loaded = generate_script._load_onset_model_auto(
                str(ckpt_path), device=torch.device("cpu")
            )
            self.assertTrue(generator._is_itgpt_placement_model(loaded))


if __name__ == "__main__":
    unittest.main()
