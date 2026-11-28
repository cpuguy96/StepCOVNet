"""Tests for Beat This! / librosa beat tracking and ITGPT-MERT audio generation."""

from __future__ import annotations

import pathlib
import queue
import sys
import tempfile
import unittest
import warnings
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
from stepcovnet.dataset_prep import simfile_parser
from stepcovnet.pytorch.itgpt import models as itgpt_models
from stepcovnet.pytorch.itgpt import trainers as itgpt_trainers

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import generate as generate_script  # noqa: E402
import generate_ui as generate_ui_script  # noqa: E402


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

    def test_parabolic_sub_frame_peak_refinement(self) -> None:
        # True quadratic peak at frame 50.25 (1.005 s at 50 FPS):
        # y(k) = 5 - 4 * (k - 50.25)^2
        idx = np.arange(120, dtype=np.float64)
        beat_logits = torch.tensor(5.0 - 4.0 * (idx - 50.25) ** 2, dtype=torch.float32)
        db_logits = torch.full((120,), -5.0, dtype=torch.float32)

        raw_beats, _ = beat_this_model.postprocess_minimal(
            beat_logits, db_logits, fps=50, sub_frame=False
        )
        refined_beats, _ = beat_this_model.postprocess_minimal(
            beat_logits, db_logits, fps=50, sub_frame=True
        )
        self.assertAlmostEqual(float(raw_beats[0]), 1.000, places=5)
        self.assertAlmostEqual(float(refined_beats[0]), 1.005, places=5)

    def test_offline_auto_fallback_to_librosa_on_checkpoint_error(self) -> None:
        fake_beats = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        with (
            mock.patch.object(
                beat_tracking.beat_this_model,
                "predict_beats_from_audio",
                side_effect=beat_this_model.BeatThisCheckpointError("offline"),
            ),
            mock.patch.object(
                beat_tracking.librosa,
                "load",
                return_value=(np.zeros(44100, dtype=np.float32), constants.TARGET_SR),
            ),
            mock.patch.object(
                beat_tracking.librosa.beat,
                "beat_track",
                return_value=(np.array([120.0]), fake_beats),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            grid = beat_tracking.estimate_beats(
                "song.ogg", tracker=beat_tracking.TRACKER_AUTO
            )
            self.assertEqual(grid.tracker, beat_tracking.TRACKER_LIBROSA)
            self.assertTrue(any(w.category is RuntimeWarning for w in caught))

            # Explicit tracker='beat_this' must still raise BeatThisCheckpointError.
            with self.assertRaises(beat_this_model.BeatThisCheckpointError):
                beat_tracking.estimate_beats(
                    "song.ogg", tracker=beat_tracking.TRACKER_BEAT_THIS
                )

    def test_beat_grid_phase_and_sm_export_roundtrip(self) -> None:
        # 120 BPM (0.5 s/beat), first beat at 0.25 s, first downbeat at 1.25 s
        # (includes a pickup beat before the first downbeat).
        beats = 0.25 + np.arange(16, dtype=np.float64) * 0.5
        downbeats = np.array([1.25, 3.25, 5.25, 7.25], dtype=np.float64)
        grid = beat_tracking.BeatGrid(
            beat_times=beats,
            downbeat_times=downbeats,
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        self.assertAlmostEqual(grid.anchor_time_sec, 1.25, places=6)
        self.assertAlmostEqual(grid.sm_offset_sec, -1.25, places=6)

        # Beat phases for quarter + eighth notes on the grid
        test_times = np.array([0.75, 1.25, 1.50, 1.75], dtype=np.float64)
        phases = grid.beat_phase_for_times(test_times)
        np.testing.assert_allclose(phases, [0.0, 0.0, 0.5, 0.0], atol=1e-5)

        out = generator.OutputData(
            title="Aligned Song",
            bpm=120,
            notes={
                "Challenge": [
                    ("0.75", "1000"),  # pickup beat before first downbeat
                    ("1.25", "0100"),  # downbeat
                    ("1.5", "0010"),  # 8th note
                    ("1.875", "0001"),  # 16th note
                ]
            },
            offset=grid.sm_offset_sec,
            beat_grid=grid,
        )
        sm_text = out.generate_sm_output(music_filename="song.ogg", meter=9)
        self.assertIn("#TITLE:Aligned Song;", sm_text)
        self.assertIn("#MUSIC:song.ogg;", sm_text)
        self.assertIn("#BPMS:0.000=120.000;", sm_text)

        # Round-trip through StepCOVNet's own StepMania simfile parser + TimingEngine
        parsed = simfile_parser.parse_simfile_text(sm_text)
        self.assertEqual(len(parsed.charts), 1)
        timing = simfile_parser.TimingEngine(
            simfile_parser.TimingData.from_simfile(parsed, parsed.charts[0])
        )
        notes = list(simfile_parser.iter_chart_notes(parsed.charts[0]))
        recovered_times = np.asarray(
            [timing.time_at(n.beat) for n in notes], dtype=np.float64
        )
        expected_times = np.asarray([0.75, 1.25, 1.5, 1.875], dtype=np.float64)
        np.testing.assert_allclose(recovered_times, expected_times, atol=0.005)

    def test_generate_ui_advanced_options_and_sm_output(self) -> None:
        ok, vals = generate_ui_script._validate_advanced_options(
            "beat_this", "2.5", "0.55"
        )
        self.assertTrue(ok)
        self.assertEqual(vals, ("beat_this", 2.5, 0.55))

        ok_bad_tracker, _ = generate_ui_script._validate_advanced_options(
            "invalid", "2.0", "0.5"
        )
        self.assertFalse(ok_bad_tracker)
        ok_bad_density, _ = generate_ui_script._validate_advanced_options(
            "auto", "0", "0.5"
        )
        self.assertFalse(ok_bad_density)
        ok_bad_thr, _ = generate_ui_script._validate_advanced_options(
            "auto", "2.0", "1.5"
        )
        self.assertFalse(ok_bad_thr)

        # _run_generation writes .sm when output_path ends with .sm and forwards args.
        fake_out = generator.OutputData(
            title="UI SM Test",
            bpm=120,
            notes={"Challenge": [("0.5", "1000"), ("1.0", "0001")]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_sm = pathlib.Path(tmpdir) / "chart.sm"
            q: queue.Queue = queue.Queue()
            with (
                mock.patch.object(
                    generate_ui_script.pretrained,
                    "resolve_onset_model_path",
                    return_value="/fake/onset.pt",
                ),
                mock.patch.object(
                    generate_ui_script.pretrained,
                    "resolve_arrow_model_path",
                    return_value="/fake/arrow.pt",
                ),
                mock.patch.object(
                    generate_ui_script.trainers,
                    "load_onset_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_ui_script.trainers,
                    "load_arrow_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_ui_script.generator,
                    "generate_output_data",
                    return_value=fake_out,
                ) as mock_gen,
            ):
                generate_ui_script._run_generation(
                    audio_path="/songs/demo.ogg",
                    song_title="UI SM Test",
                    bpm=None,
                    onset_model_path=None,
                    arrow_model_path=None,
                    output_path=str(out_sm),
                    use_post_processing=False,
                    result_queue=q,
                    beat_tracker="beat_this",
                    snap_beats=True,
                    target_density=2.5,
                    threshold=0.55,
                )
            src, success, val = q.get_nowait()
            self.assertEqual(src, "generation")
            self.assertTrue(success, msg=str(val))
            kw = mock_gen.call_args.kwargs
            self.assertEqual(kw["beat_tracker"], "beat_this")
            self.assertTrue(kw["snap_beats"])
            self.assertAlmostEqual(kw["target_density"], 2.5)
            self.assertAlmostEqual(kw["threshold"], 0.55)
            sm_written = out_sm.read_text()
            self.assertIn("#TITLE:UI SM Test;", sm_written)
            self.assertIn("#MUSIC:demo.ogg;", sm_written)

    def test_regularize_beat_grid_steady_and_variable_tempo(self) -> None:
        # 128 BPM steady song with 2 dropped beats (breakdown) and 1 spurious extra beat.
        true_ibi = 60.0 / 128.0
        true_beats = 0.5 + np.arange(32, dtype=np.float64) * true_ibi
        rng = np.random.default_rng(7)
        noisy = true_beats + rng.normal(0.0, 0.004, size=true_beats.size)
        # Drop beats 10 & 11 and inject one spurious off-beat at beat 20.5
        keep_mask = np.ones(noisy.size, dtype=bool)
        keep_mask[[10, 11]] = False
        gapped = np.sort(
            np.concatenate([noisy[keep_mask], [true_beats[20] + 0.5 * true_ibi]])
        )
        downbeats = noisy[::4]

        raw_grid = beat_tracking.BeatGrid(
            beat_times=gapped,
            downbeat_times=downbeats,
            bpm=128.4,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        reg = beat_tracking.regularize_beat_grid(raw_grid)
        self.assertAlmostEqual(reg.bpm, 128.0, places=1)
        # Exact constant spacing with zero dropped-beat drift
        self.assertLess(float(np.std(np.diff(reg.beat_times))), 1e-9)
        self.assertGreater(reg.downbeat_times.size, 0)
        # Verify exact-grid slot decoding puts quarter notes right on beat edges
        probs = np.zeros((2, 48), dtype=np.float32)
        probs[0, 0] = 0.9
        probs[0, 24] = 0.9
        exact_times = beat_tracking.decode_itgpt_slots_to_times(
            probs, reg, threshold=0.5, use_slot_centers=False
        )
        self.assertAlmostEqual(
            float(exact_times[0]), float(reg.beat_times[0]), places=6
        )
        self.assertAlmostEqual(
            float(exact_times[1]),
            float(0.5 * (reg.beat_times[0] + reg.beat_times[1])),
            places=6,
        )

        # Variable-tempo song (100 BPM -> 140 BPM halfway through) must stay untouched.
        first_half = 0.5 + np.arange(16, dtype=np.float64) * 0.6
        second_half = first_half[-1] + np.arange(1, 17, dtype=np.float64) * (
            60.0 / 140.0
        )
        var_grid = beat_tracking.BeatGrid(
            beat_times=np.concatenate([first_half, second_half]),
            downbeat_times=np.zeros(0),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        unchanged = beat_tracking.regularize_beat_grid(var_grid)
        np.testing.assert_allclose(unchanged.beat_times, var_grid.beat_times, atol=1e-9)

    def test_difficulty_preset_validation_and_defaults(self) -> None:
        self.assertEqual(
            [p.name for p in beat_tracking.DEFAULT_DIFFICULTY_PACK],
            ["Beginner", "Easy", "Medium", "Hard", "Challenge"],
        )
        self.assertEqual(
            [p.meter for p in beat_tracking.DEFAULT_DIFFICULTY_PACK],
            [3, 5, 7, 9, 11],
        )
        densities = [p.target_density for p in beat_tracking.DEFAULT_DIFFICULTY_PACK]
        self.assertEqual(densities, sorted(densities))

        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(name="  ", meter=5, target_density=1.0)
        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(name="Easy", meter=0, target_density=1.0)
        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(name="Easy", meter=5, target_density=0.0)

    def test_predict_itgpt_pack_runs_beat_and_mert_once_with_topk_fallback(
        self,
    ) -> None:
        grid = beat_tracking.BeatGrid(
            beat_times=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            downbeat_times=np.array([0.5, 2.5], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        fake_mert = np.zeros((400, 1024), dtype=np.float32)
        seen_charts: list[tuple[str, int, float]] = []

        def fake_predict_chart_slots(_model, chart, **_kwargs) -> np.ndarray:
            mean_density = float(np.sum(chart.slots)) / float(chart.slots.shape[0])
            seen_charts.append((chart.difficulty, int(chart.meter), mean_density))
            probs = np.zeros((chart.slots.shape[0], 48), dtype=np.float32)
            if chart.difficulty == "Beginner":
                # All below threshold=0.5 -> exercises fallback_top_k
                probs[:, 0] = 0.2
                probs[0, 0] = 0.35
                probs[2, 0] = 0.30
            elif chart.difficulty == "Easy":
                probs[:, 0] = 0.8
            elif chart.difficulty == "Medium":
                probs[:, 0] = 0.8
                probs[:, 24] = 0.8
            elif chart.difficulty == "Hard":
                probs[:, [0, 12, 24]] = 0.85
            else:
                probs[:, [0, 12, 24, 36]] = 0.9
            return probs

        dummy_model = mock.Mock(n_frames_per_beat=24, max_beats=64)
        with (
            mock.patch.object(
                beat_tracking, "estimate_beats", return_value=grid
            ) as mock_beats,
            mock.patch.object(
                beat_tracking,
                "_load_or_extract_mert_for_audio",
                return_value=fake_mert,
            ) as mock_mert,
            mock.patch.object(
                beat_tracking.itgpt_eval,
                "predict_chart_slots",
                side_effect=fake_predict_chart_slots,
            ) as mock_slots,
        ):
            onsets_by_diff, out_grid = beat_tracking.predict_itgpt_pack_from_audio(
                "/songs/demo.ogg",
                dummy_model,
                threshold=0.5,
                device="cpu",
            )

        mock_beats.assert_called_once()
        mock_mert.assert_called_once()
        self.assertEqual(mock_slots.call_count, 5)
        self.assertIs(out_grid, grid)
        self.assertEqual(
            list(onsets_by_diff.keys()),
            ["Beginner", "Easy", "Medium", "Hard", "Challenge"],
        )
        counts = [int(onsets_by_diff[d].size) for d in onsets_by_diff]
        # Beginner must be non-empty thanks to top-k fallback, and counts rise monotonically.
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])
        self.assertEqual([m for _, m, _ in seen_charts], [3, 5, 7, 9, 11])

        with self.assertRaises(ValueError):
            beat_tracking.predict_itgpt_pack_from_audio(
                "/songs/demo.ogg", dummy_model, difficulties=()
            )

    def test_generate_output_data_all_difficulties_itgpt_and_dense_sm_roundtrip(
        self,
    ) -> None:
        grid = beat_tracking.BeatGrid(
            beat_times=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64),
            downbeat_times=np.array([0.5, 2.5], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
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
        ).eval()
        arrow_model = models.build_arrow_model().eval()

        fake_pack = {
            "Beginner": np.array([0.5, 1.5], dtype=np.float64),
            "Easy": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            "Medium": np.array([0.5, 1.0, 1.25, 1.5, 2.0, 2.5], dtype=np.float64),
            "Hard": np.array(
                [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5], dtype=np.float64
            ),
            "Challenge": np.array(
                [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75],
                dtype=np.float64,
            ),
        }

        with mock.patch.object(
            beat_tracking,
            "predict_itgpt_pack_from_audio",
            return_value=(fake_pack, grid),
        ) as mock_pack:
            out = generator.generate_output_data(
                audio_path="/songs/demo.ogg",
                song_title="Five Pack",
                onset_model=itgpt_model,
                arrow_model=arrow_model,
                all_difficulties=True,
            )

        mock_pack.assert_called_once()
        self.assertEqual(out.bpm, 120)
        self.assertEqual(
            out.meters,
            {"Beginner": 3, "Easy": 5, "Medium": 7, "Hard": 9, "Challenge": 11},
        )
        sm_text = out.generate_sm_output(music_filename="demo.ogg")
        parsed = simfile_parser.parse_simfile_text(sm_text)
        self.assertEqual(len(parsed.charts), 5)
        self.assertEqual(
            [(c.difficulty, int(c.meter)) for c in parsed.charts],
            [
                ("Beginner", 3),
                ("Easy", 5),
                ("Medium", 7),
                ("Hard", 9),
                ("Challenge", 11),
            ],
        )
        for chart in parsed.charts:
            self.assertGreater(len(list(simfile_parser.iter_chart_notes(chart))), 0)

        # Also verify legacy dense onset model supports all_difficulties=True in one pass.
        probs = np.zeros((1, 400, 1), dtype=np.float32)
        for idx in range(10, 380, 10):
            probs[0, idx, 0] = 0.9 if idx % 40 == 0 else 0.7
        dense_onset = mock.Mock(return_value=probs)
        del dense_onset.forward_logits
        del dense_onset.predict
        fake_spec = np.zeros((constants.N_MELS, 400), dtype=np.float32)
        with (
            mock.patch.object(
                generator.datasets, "audio_to_spectrogram", return_value=fake_spec
            ),
            mock.patch.object(
                generator.datasets,
                "normalize_onset_spectrogram",
                side_effect=lambda x: x,
            ),
        ):
            dense_out = generator.generate_output_data(
                audio_path="/songs/demo.ogg",
                song_title="Dense Pack",
                bpm=128,
                onset_model=dense_onset,
                arrow_model=arrow_model,
                all_difficulties=True,
                use_post_processing=True,
            )
        dense_onset.assert_called_once()
        self.assertEqual(
            list(dense_out.notes.keys()),
            ["Beginner", "Easy", "Medium", "Hard", "Challenge"],
        )
        dense_counts = [len(dense_out.notes[d]) for d in dense_out.notes]
        self.assertGreater(dense_counts[0], 0)
        self.assertLessEqual(dense_counts[0], dense_counts[-1])

    def test_cli_and_ui_forward_all_difficulties_flag(self) -> None:
        fake_out = generator.OutputData(
            title="Pack CLI",
            bpm=128,
            notes={
                "Beginner": [("0.5", "1000")],
                "Challenge": [("0.5", "1000"), ("1.0", "0001")],
            },
            meters={"Beginner": 3, "Challenge": 11},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_sm = pathlib.Path(tmpdir) / "pack.sm"
            with (
                mock.patch.object(
                    generate_script.pretrained,
                    "resolve_onset_model_path",
                    return_value="/fake/onset.pt",
                ),
                mock.patch.object(
                    generate_script.pretrained,
                    "resolve_arrow_model_path",
                    return_value="/fake/arrow.pt",
                ),
                mock.patch.object(
                    generate_script, "_load_onset_model_auto", return_value=mock.Mock()
                ),
                mock.patch.object(
                    generate_script.trainers,
                    "load_arrow_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_script.generator,
                    "generate_output_data",
                    return_value=fake_out,
                ) as mock_gen,
            ):
                generate_script.main(
                    [
                        "--audio_path",
                        "/songs/demo.ogg",
                        "--song_title",
                        "Pack CLI",
                        "--output_file",
                        str(out_sm),
                        "--all_difficulties",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertTrue(mock_gen.call_args.kwargs["all_difficulties"])
            parsed = simfile_parser.parse_simfile_text(out_sm.read_text())
            self.assertEqual(
                [(c.difficulty, int(c.meter)) for c in parsed.charts],
                [("Beginner", 3), ("Challenge", 11)],
            )

            # UI _run_generation forwarding
            q: queue.Queue = queue.Queue()
            out_ui_sm = pathlib.Path(tmpdir) / "ui_pack.sm"
            with (
                mock.patch.object(
                    generate_ui_script.pretrained,
                    "resolve_onset_model_path",
                    return_value="/fake/onset.pt",
                ),
                mock.patch.object(
                    generate_ui_script.pretrained,
                    "resolve_arrow_model_path",
                    return_value="/fake/arrow.pt",
                ),
                mock.patch.object(
                    generate_ui_script.trainers,
                    "load_onset_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_ui_script.trainers,
                    "load_arrow_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_ui_script.generator,
                    "generate_output_data",
                    return_value=fake_out,
                ) as mock_ui_gen,
            ):
                generate_ui_script._run_generation(
                    audio_path="/songs/demo.ogg",
                    song_title="Pack UI",
                    bpm=None,
                    onset_model_path=None,
                    arrow_model_path=None,
                    output_path=str(out_ui_sm),
                    use_post_processing=False,
                    result_queue=q,
                    all_difficulties=True,
                )
            src, ok, val = q.get_nowait()
            self.assertEqual(src, "generation")
            self.assertTrue(ok, msg=str(val))
            self.assertTrue(mock_ui_gen.call_args.kwargs["all_difficulties"])


if __name__ == "__main__":
    unittest.main()
