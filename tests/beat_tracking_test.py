"""Tests for Beat This! / librosa beat tracking and ITGPT-MERT audio generation."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from stepcovnet import (
    beat_this_model,
    beat_tracking,
    constants,
    generator,
    models,
)
from stepcovnet.pytorch.itgpt import models as itgpt_models
from stepcovnet.pytorch.itgpt import trainers as itgpt_trainers

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import generate as generate_script  # noqa: E402


class BeatThisModelUnitTest(unittest.TestCase):
    def test_tiny_beat_this_forward_and_state_dict_roundtrip(self) -> None:
        for sum_head in (True, False):
            with self.subTest(sum_head=sum_head):
                model = beat_this_model.BeatThis(
                    spect_dim=32,
                    transformer_dim=32,
                    ff_mult=2,
                    n_layers=1,
                    head_dim=16,
                    stem_dim=16,
                    sum_head=sum_head,
                    partial_transformers=True,
                ).eval()
                x = torch.randn(2, 24, 32)
                with torch.inference_mode():
                    out = model(x)
                self.assertEqual(tuple(out["beat"].shape), (2, 24))
                self.assertEqual(tuple(out["downbeat"].shape), (2, 24))
                self.assertTrue(torch.isfinite(out["beat"]).all())
                self.assertTrue(torch.isfinite(out["downbeat"]).all())

        # Verify Lightning + torch.compile prefix cleaning matches strict=True load.
        model = beat_this_model.BeatThis(
            spect_dim=32,
            transformer_dim=32,
            ff_mult=2,
            n_layers=1,
            head_dim=16,
            stem_dim=16,
        )
        prefixed = {
            f"model._orig_mod.{k}": v.clone() for k, v in model.state_dict().items()
        }
        cleaned = beat_this_model._clean_state_dict(prefixed)
        model.load_state_dict(cleaned, strict=True)

    def test_log_mel_spectrogram_and_chunked_inference(self) -> None:
        spect_fn = beat_this_model.LogMelSpectrogram()
        wav = torch.randn(beat_this_model.BEAT_THIS_SR * 2, dtype=torch.float32) * 0.1
        spect = spect_fn(wav)
        self.assertEqual(spect.ndim, 2)
        self.assertEqual(spect.shape[1], 128)
        self.assertGreater(spect.shape[0], 90)
        self.assertTrue(torch.isfinite(spect).all())
        self.assertGreaterEqual(float(spect.min()), 0.0)

        # Multi-chunk split_predict_aggregate on a tiny model
        tiny_model = beat_this_model.BeatThis(
            spect_dim=32,
            transformer_dim=32,
            ff_mult=2,
            n_layers=1,
            head_dim=16,
            stem_dim=16,
        ).eval()
        fake_spect = torch.randn(100, 32)
        with torch.inference_mode():
            beat_logits, db_logits = beat_this_model.split_predict_aggregate(
                fake_spect,
                tiny_model,
                chunk_size=40,
                border_size=4,
                overlap_mode="keep_first",
            )
        self.assertEqual(tuple(beat_logits.shape), (100,))
        self.assertEqual(tuple(db_logits.shape), (100,))
        self.assertTrue(torch.isfinite(beat_logits).all())
        self.assertTrue(torch.isfinite(db_logits).all())

    def test_postprocess_minimal_peak_picking_and_downbeat_snapping(self) -> None:
        beat_logits = torch.full((200,), -5.0)
        db_logits = torch.full((200,), -5.0)
        # Beats at frames 25, 50, 75, 100 (0.5, 1.0, 1.5, 2.0 s at 50 FPS)
        # Include an adjacent tied peak at 50 & 51 to exercise deduplication.
        for f in (25, 50, 51, 75, 100):
            beat_logits[f] = 4.0
        # Downbeats slightly offset by 1 frame (should snap to nearest beat)
        db_logits[26] = 3.0
        db_logits[101] = 3.0

        beats, downbeats = beat_this_model.postprocess_minimal(
            beat_logits, db_logits, fps=50
        )
        np.testing.assert_allclose(beats, [0.5, 1.01, 1.5, 2.0], atol=1e-6)
        np.testing.assert_allclose(downbeats, [0.5, 2.0], atol=1e-6)

    def test_local_checkpoint_loading_and_waveform_prediction_offline(self) -> None:
        tiny_hparams = {
            "spect_dim": 128,
            "transformer_dim": 32,
            "ff_mult": 2,
            "n_layers": 1,
            "head_dim": 16,
            "stem_dim": 16,
            "dropout": {"frontend": 0.0, "transformer": 0.0},
            "sum_head": True,
            "partial_transformers": True,
        }
        tiny = beat_this_model.BeatThis(**tiny_hparams).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = pathlib.Path(tmpdir) / "tiny_beat_this.ckpt"
            torch.save(
                {
                    "hyper_parameters": tiny_hparams,
                    "state_dict": {
                        f"model.{k}": v for k, v in tiny.state_dict().items()
                    },
                },
                ckpt_path,
            )
            beat_this_model._load_cached_bundle.cache_clear()
            loaded = beat_this_model.load_beat_this_model(str(ckpt_path), device="cpu")
            self.assertIsInstance(loaded, beat_this_model.BeatThis)
            stereo_wav = np.zeros((22050, 2), dtype=np.float32)
            beats, downbeats = beat_this_model.predict_beats_from_waveform(
                stereo_wav,
                sr=22050,
                checkpoint_path=str(ckpt_path),
                device="cpu",
            )
            self.assertEqual(beats.ndim, 1)
            self.assertEqual(downbeats.ndim, 1)
            beat_this_model._load_cached_bundle.cache_clear()


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
    def test_beat_this_is_built_in(self) -> None:
        self.assertTrue(beat_tracking.beat_this_available())

    def test_estimate_beats_beat_this_mocked(self) -> None:
        with mock.patch.object(
            beat_tracking.beat_this_model,
            "predict_beats_from_audio",
            return_value=(
                np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
                np.array([0.5, 2.5], dtype=np.float64),
            ),
        ) as mock_pred:
            grid = beat_tracking.estimate_beats_beat_this("song.ogg", device="cpu")
        mock_pred.assert_called_once()
        self.assertEqual(grid.tracker, beat_tracking.TRACKER_BEAT_THIS)
        self.assertAlmostEqual(grid.bpm, 120.0, places=4)
        self.assertEqual(grid.n_beats, 4)
        self.assertEqual(grid.downbeat_times.size, 2)

    def test_estimate_beats_beat_this_too_few_beats_raises(self) -> None:
        with (
            mock.patch.object(
                beat_tracking.beat_this_model,
                "predict_beats_from_audio",
                return_value=(np.array([0.5]), np.zeros(0)),
            ),
            self.assertRaises(ValueError),
        ):
            beat_tracking.estimate_beats_beat_this("song.ogg", device="cpu")

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
