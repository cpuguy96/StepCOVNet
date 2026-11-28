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
from stepcovnet.itgpt import models as itgpt_models
from stepcovnet.itgpt import trainers as itgpt_trainers

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_raw_audio_onset  # noqa: E402
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
        # Synthetic 120 BPM kick + synth-pad waveform with exact attacks at 0.5s intervals
        true_beats = np.array(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float64
        )
        n = int(round(4.6 * sr))
        t = np.arange(n, dtype=np.float64) / sr
        y = 0.10 * np.sin(2.0 * np.pi * 220.0 * t)
        for tb in true_beats:
            i0 = int(round(tb * sr))
            length = int(round(0.12 * sr))
            lt = np.arange(length, dtype=np.float64) / sr
            env = (1.0 - np.exp(-lt / 0.004)) * np.exp(-lt / 0.045)
            phase = 2.0 * np.pi * np.cumsum(50.0 + 110.0 * np.exp(-lt / 0.02)) / sr
            y[i0 : i0 + length] += 0.75 * env * np.sin(phase)
        y_f32 = y.astype(np.float32)

        # Start with a +24 ms late beat grid AND a beat near t=0 (0.015s) to verify
        # negative shifts are not blocked by Beat 0 proximity.
        late_beats = np.concatenate([[0.015], true_beats + 0.024])
        late_grid = beat_tracking.BeatGrid(
            beat_times=late_beats,
            downbeat_times=np.array([0.015, 2.024], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        with mock.patch.object(
            beat_tracking.librosa,
            "load",
            return_value=(y_f32, sr),
        ):
            snapped = beat_tracking.snap_beats_to_onsets(
                late_grid, "song.ogg", max_shift_ms=35.0, step_ms=0.5
            )
        # Beat 0 shifted slightly negative and was dropped cleanly; remaining 8 beats align within 3 ms.
        self.assertEqual(snapped.beat_times.size, true_beats.size)
        self.assertTrue(np.all(snapped.beat_times >= 0.0))
        residual_ms = float(np.mean(np.abs(snapped.beat_times - true_beats)) * 1000.0)
        self.assertLess(residual_ms, 3.0)
        self.assertAlmostEqual(float(snapped.downbeat_times[0]), 2.0, delta=0.003)


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
                "stepcovnet.itgpt.evaluation.predict_chart_slots",
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
            [2, 4, 6, 9, 11],
        )
        densities = [p.target_density for p in beat_tracking.DEFAULT_DIFFICULTY_PACK]
        self.assertEqual(densities, sorted(densities))

        # BPM-adaptive NPS lowers steps/beat at higher tempos to keep physical NPS steady.
        challenge = beat_tracking.DEFAULT_DIFFICULTY_PACK[-1]
        spb_95 = challenge.resolve_target_density(95.0)
        spb_140 = challenge.resolve_target_density(140.0)
        spb_180 = challenge.resolve_target_density(180.0)
        self.assertGreater(spb_95, spb_140)
        self.assertGreater(spb_140, spb_180)
        self.assertAlmostEqual(spb_140 * (140.0 / 60.0), 4.90, places=2)

        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(name="  ", meter=5, target_density=1.0)
        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(name="Easy", meter=0, target_density=1.0)
        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(name="Easy", meter=5, target_density=0.0)
        with self.assertRaises(ValueError):
            beat_tracking.DifficultyPreset(
                name="Easy", meter=5, target_density=1.0, target_nps=-1.0
            )

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
            # Inject a noisy non-musical slot (index 5) across all charts to verify filtering!
            probs[:, 5] = 0.99
            if chart.difficulty == "Beginner":
                # All valid slots below threshold=0.5 -> exercises fallback_top_k
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
        # Beginner must be non-empty thanks to top-k fallback, noisy slot 5 excluded everywhere.
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[1], 6)  # Easy: only slot 0 on 6 beats (slot 5 ignored)
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])
        self.assertEqual([m for _, m, _ in seen_charts], [2, 4, 6, 9, 11])

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
            {"Beginner": 2, "Easy": 4, "Medium": 6, "Hard": 9, "Challenge": 11},
        )
        sm_text = out.generate_sm_output(music_filename="demo.ogg")
        parsed = simfile_parser.parse_simfile_text(sm_text)
        self.assertEqual(len(parsed.charts), 5)
        self.assertEqual(
            [(c.difficulty, int(c.meter)) for c in parsed.charts],
            [
                ("Beginner", 2),
                ("Easy", 4),
                ("Medium", 6),
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
            meters={"Beginner": 2, "Challenge": 11},
        )
        # CLI defaults snap_beats=True and supports --no-snap_beats opt-out.
        default_args = generate_script.PARSER.parse_args(
            ["--audio_path", "a.ogg", "--song_title", "T", "--output_file", "o.sm"]
        )
        self.assertTrue(default_args.snap_beats)
        nosnap_args = generate_script.PARSER.parse_args(
            [
                "--audio_path",
                "a.ogg",
                "--song_title",
                "T",
                "--output_file",
                "o.sm",
                "--no-snap_beats",
            ]
        )
        self.assertFalse(nosnap_args.snap_beats)

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
            self.assertTrue(mock_gen.call_args.kwargs["snap_beats"])
            parsed = simfile_parser.parse_simfile_text(out_sm.read_text())
            self.assertEqual(
                [(c.difficulty, int(c.meter)) for c in parsed.charts],
                [("Beginner", 2), ("Challenge", 11)],
            )

            # UI _run_generation forwarding (defaults snap_beats=True)
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
            self.assertTrue(mock_ui_gen.call_args.kwargs["snap_beats"])

    def test_integer_bpm_lock_and_variable_tempo_sm_roundtrip(self) -> None:
        # 1. Near-integer steady tempo (139.985 BPM) locks cleanly to exact 140.000 BPM.
        ibi_near = 60.0 / 139.985
        near_beats = 0.5 + np.arange(80, dtype=np.float64) * ibi_near
        near_grid = beat_tracking.BeatGrid(
            beat_times=near_beats,
            downbeat_times=near_beats[::4],
            bpm=139.985,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        locked = beat_tracking.regularize_beat_grid(near_grid)
        self.assertAlmostEqual(locked.bpm, 140.0, places=6)

        # Genuinely fractional tempo (128.35 BPM) must NOT snap to 128.0.
        frac_beats = 0.5 + np.arange(80, dtype=np.float64) * (60.0 / 128.35)
        frac_grid = beat_tracking.BeatGrid(
            beat_times=frac_beats,
            downbeat_times=frac_beats[::4],
            bpm=128.35,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        kept_frac = beat_tracking.regularize_beat_grid(frac_grid)
        self.assertAlmostEqual(kept_frac.bpm, 128.35, places=2)

        # 2. Variable-tempo song (100 BPM for 16 beats -> 140 BPM for 16 beats)
        # with pickup beats before first downbeat round-trips through StepMania parser.
        first_half = 0.5 + np.arange(16, dtype=np.float64) * (60.0 / 100.0)
        second_half = first_half[-1] + np.arange(1, 17, dtype=np.float64) * (
            60.0 / 140.0
        )
        var_beats = np.concatenate([first_half, second_half])
        var_grid = beat_tracking.BeatGrid(
            beat_times=var_beats,
            downbeat_times=np.array([var_beats[2], var_beats[18]], dtype=np.float64),
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        note_times = np.array(
            [
                var_beats[0],
                var_beats[2],
                var_beats[10],
                var_beats[16],
                var_beats[24],
                var_beats[30],
            ],
            dtype=np.float64,
        )
        out = generator.OutputData(
            title="Tempo Change Song",
            bpm=120,
            notes={"Challenge": [(str(t), "1000") for t in note_times]},
            offset=var_grid.sm_offset_sec,
            beat_grid=var_grid,
        )
        sm_text = out.generate_sm_output(music_filename="var.ogg", meter=10)
        parsed = simfile_parser.parse_simfile_text(sm_text)
        timing_data = simfile_parser.TimingData.from_simfile(parsed, parsed.charts[0])
        # Exactly 2 clean constant-BPM plateaus (100 BPM and 140 BPM), no jitter!
        self.assertEqual(len(timing_data.bpms), 2)
        self.assertAlmostEqual(timing_data.bpms[0].value, 100.0, places=1)
        self.assertAlmostEqual(timing_data.bpms[1].value, 140.0, places=1)
        engine = simfile_parser.TimingEngine(timing_data)
        recovered = np.asarray(
            [
                engine.time_at(n.beat)
                for n in simfile_parser.iter_chart_notes(parsed.charts[0])
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(recovered, note_times, atol=0.005)

    def test_eval_raw_audio_onset_modes(self) -> None:
        beats = 0.5 + np.arange(8, dtype=np.float64) * 0.5
        oracle_grid = beat_tracking.BeatGrid(
            beat_times=beats,
            downbeat_times=beats[::4],
            bpm=120.0,
            tracker="oracle",
        )
        song = eval_raw_audio_onset.SongEvalGroup(
            song_key="pack/demo",
            audio_path="/songs/demo.ogg",
            oracle_grid=oracle_grid,
            gt_bpm=120.0,
            charts=(
                eval_raw_audio_onset.SongEvalChart(
                    song_key="pack/demo",
                    difficulty="challenge",
                    meter=10,
                    gt_times_sec=beats[:6],
                    n_gt_beats=8,
                    target_density=6.0 / 8.0,
                ),
            ),
        )
        # Raw Beat This! sits +25 ms late; snap pulls it right onto GT beats!
        raw_late_grid = beat_tracking.BeatGrid(
            beat_times=beats + 0.025,
            downbeat_times=beats[::4] + 0.025,
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        fake_mert = np.zeros((600, 16), dtype=np.float32)

        def fake_predict(_model, chart, **_kwargs) -> np.ndarray:
            probs = np.zeros((chart.slots.shape[0], 48), dtype=np.float32)
            probs[:6, 0] = 0.9
            return probs

        dummy_model = mock.Mock(n_frames_per_beat=24, max_beats=64)
        with (
            mock.patch.object(
                beat_tracking,
                "_load_or_extract_mert_for_audio",
                return_value=fake_mert,
            ) as mock_mert,
            mock.patch.object(
                beat_tracking,
                "estimate_beats",
                return_value=raw_late_grid,
            ) as mock_est,
            mock.patch.object(
                beat_tracking,
                "snap_beats_to_onsets",
                return_value=oracle_grid,
            ),
            mock.patch.object(
                beat_tracking.itgpt_eval,
                "predict_chart_slots",
                side_effect=fake_predict,
            ),
        ):
            report = eval_raw_audio_onset.evaluate_raw_audio_modes(
                [song],
                dummy_model,
                modes=("oracle", "beat_this_raw", "beat_this_full"),
                device="cpu",
            )

        # MERT loaded once and Beat This! neural inference cached across raw/full!
        mock_mert.assert_called_once()
        mock_est.assert_called_once()
        self.assertAlmostEqual(report["modes"]["oracle"]["ev_f1_20ms"], 1.0, places=4)
        self.assertAlmostEqual(
            report["modes"]["beat_this_raw"]["ev_f1_20ms"], 0.0, places=4
        )
        self.assertAlmostEqual(
            report["modes"]["beat_this_raw"]["ev_f1_50ms"], 1.0, places=4
        )
        self.assertAlmostEqual(
            report["modes"]["beat_this_full"]["ev_f1_20ms"], 1.0, places=4
        )

    def test_estimate_sample_preview_finds_chorus_and_snaps_to_downbeat(self) -> None:
        sr = constants.TARGET_SR
        duration = 60.0
        n_samples = int(duration * sr)
        t = np.arange(n_samples, dtype=np.float32) / float(sr)
        # Quiet intro/outro with a loud percussive chorus from 22.0s to 40.0s.
        y = 0.02 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
        chorus_mask = (t >= 22.0) & (t <= 40.0)
        # Rhythmic bursts every 0.5s (120 BPM) inside the chorus.
        pulse = 0.5 + 0.5 * np.maximum(0.0, np.sin(2.0 * np.pi * 2.0 * t))
        y[chorus_mask] = (
            0.8 * pulse[chorus_mask] * np.sin(2.0 * np.pi * 440.0 * t[chorus_mask])
        ).astype(np.float32)

        beats = np.arange(0.0, 60.0, 0.5, dtype=np.float64)
        downbeats = beats[::4]  # 0.0, 2.0, 4.0, ..., 22.0, 24.0, ...
        grid = beat_tracking.BeatGrid(
            beat_times=beats,
            downbeat_times=downbeats,
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )

        with mock.patch.object(
            beat_tracking.librosa,
            "load",
            return_value=(y, sr),
            autospec=True,
        ):
            start_sec, length_sec = beat_tracking.estimate_sample_preview(
                "synthetic.ogg", beat_grid=grid, sample_length_sec=15.0
            )

        self.assertEqual(length_sec, 15.0)
        self.assertGreaterEqual(start_sec, 20.0)
        self.assertLessEqual(start_sec, 26.0)
        # Snapped cleanly onto an exact 2.0-second downbeat.
        self.assertAlmostEqual(start_sec % 2.0, 0.0, places=3)

        # Short clip (< 15s) starts at 0.0 and clamps sample length to duration.
        short_y = y[: int(8.0 * sr)]
        with mock.patch.object(
            beat_tracking.librosa,
            "load",
            return_value=(short_y, sr),
            autospec=True,
        ):
            short_start, short_len = beat_tracking.estimate_sample_preview(
                "short.ogg", sample_length_sec=15.0
            )
        self.assertEqual(short_start, 0.0)
        self.assertAlmostEqual(short_len, 8.0, places=2)

    def test_export_stepmania_song_folder_and_sanitize(self) -> None:
        self.assertEqual(
            generator.sanitize_song_folder_name('My: Song/Title*?"<>|... '),
            "My_ Song_Title_",
        )
        self.assertEqual(
            generator.sanitize_song_folder_name("   ...  ", fallback="fallback_name"),
            "fallback_name",
        )

        beats = np.arange(0.5, 16.5, 0.5, dtype=np.float64)
        grid = beat_tracking.BeatGrid(
            beat_times=beats,
            downbeat_times=beats[::4],
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        out = generator.OutputData(
            title="Test: Song!",
            bpm=120,
            notes={
                "Easy": [("0.5", "1000"), ("1.5", "0001")],
                "Challenge": [("0.5", "1000"), ("1.0", "0100"), ("1.5", "0001")],
            },
            offset=grid.sm_offset_sec,
            beat_grid=grid,
            meters={"Easy": 4, "Challenge": 11},
            sample_start=24.0,
            sample_length=15.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            src_audio = tmp / "track.ogg"
            src_audio.write_bytes(b"fake-ogg-payload")
            pack_dir = tmp / "Songs" / "StepCOVNet"

            exported = generator.export_stepmania_song_folder(
                out,
                audio_path=src_audio,
                output_dir=pack_dir,
            )

            self.assertEqual(exported.song_dir, pack_dir / "Test_ Song!")
            self.assertTrue(exported.sm_path.is_file())
            self.assertTrue(exported.audio_path.is_file())
            self.assertEqual(exported.audio_path.read_bytes(), b"fake-ogg-payload")

            sm_text = exported.sm_path.read_text(encoding="utf-8")
            self.assertIn("#MUSIC:track.ogg;", sm_text)
            self.assertIn("#SAMPLESTART:24.000;", sm_text)
            self.assertIn("#SAMPLELENGTH:15.000;", sm_text)

            parsed = simfile_parser.parse_simfile_text(sm_text)
            self.assertEqual(len(parsed.charts), 2)
            self.assertEqual(
                [c.difficulty for c in parsed.charts], ["Easy", "Challenge"]
            )

            # Re-exporting when source audio already lives inside song_dir works cleanly.
            exported_same = generator.export_stepmania_song_folder(
                out,
                audio_path=exported.audio_path,
                output_dir=pack_dir,
            )
            self.assertEqual(exported_same.audio_path, exported.audio_path)

    def test_cli_output_dir_mert_cache_and_ui_song_folder_export(self) -> None:
        # CLI defaults mert_cache=True and allows --no-mert_cache + --output_dir alone.
        args_default = generate_script.PARSER.parse_args(
            [
                "--audio_path",
                "song.ogg",
                "--song_title",
                "My Song",
                "--output_dir",
                "out_pack",
            ]
        )
        self.assertTrue(args_default.mert_cache)
        self.assertIsNone(args_default.output_file)
        self.assertEqual(args_default.output_dir, "out_pack")

        args_nocache = generate_script.PARSER.parse_args(
            [
                "--audio_path",
                "song.ogg",
                "--song_title",
                "My Song",
                "--output_dir",
                "out_pack",
                "--no-mert_cache",
            ]
        )
        self.assertFalse(args_nocache.mert_cache)

        dummy_out = generator.OutputData(
            title="CLI Song",
            bpm=120,
            notes={"Challenge": [("0.5", "1000"), ("1.0", "0001")]},
            sample_start=12.0,
            sample_length=15.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            audio = tmp / "song.ogg"
            audio.write_bytes(b"audio-bytes")
            pack_dir = tmp / "pack"

            # CLI --output_dir exports playable folder and forwards use_mert_cache.
            with (
                mock.patch.object(generate_script.wsl_gpu, "guard_pytorch_gpu_job"),
                mock.patch.object(
                    generate_script.pretrained,
                    "resolve_onset_model_path",
                    return_value=str(tmp / "onset.pt"),
                ),
                mock.patch.object(
                    generate_script.pretrained,
                    "resolve_arrow_model_path",
                    return_value=str(tmp / "arrow.pt"),
                ),
                mock.patch.object(
                    generate_script,
                    "_load_onset_model_auto",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_script.trainers,
                    "load_arrow_model",
                    return_value=mock.Mock(),
                ),
                mock.patch.object(
                    generate_script.generator,
                    "generate_output_data",
                    return_value=dummy_out,
                ) as mock_gen,
            ):
                generate_script.main(
                    [
                        "--audio_path",
                        str(audio),
                        "--song_title",
                        "CLI Song",
                        "--output_dir",
                        str(pack_dir),
                        "--no-mert_cache",
                    ]
                )
            self.assertFalse(mock_gen.call_args.kwargs["use_mert_cache"])
            self.assertTrue((pack_dir / "CLI Song" / "CLI Song.sm").is_file())
            self.assertTrue((pack_dir / "CLI Song" / "song.ogg").is_file())

            # GUI _run_generation with export_song_folder=True creates playable folder.
            ui_pack = tmp / "ui_pack"
            ui_pack.mkdir()
            result_q: queue.Queue = queue.Queue()
            with (
                mock.patch.object(
                    generate_ui_script.pretrained,
                    "resolve_onset_model_path",
                    return_value=str(tmp / "onset.pt"),
                ),
                mock.patch.object(
                    generate_ui_script.pretrained,
                    "resolve_arrow_model_path",
                    return_value=str(tmp / "arrow.pt"),
                ),
                mock.patch.object(
                    generate_ui_script.beat_tracking,
                    "is_itgpt_checkpoint",
                    return_value=True,
                ),
                mock.patch.object(
                    generate_ui_script.beat_tracking,
                    "load_onset_model_auto",
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
                    return_value=dummy_out,
                ),
            ):
                generate_ui_script._run_generation(
                    audio_path=str(audio),
                    song_title="CLI Song",
                    bpm=None,
                    onset_model_path=None,
                    arrow_model_path=None,
                    output_path=str(ui_pack / "ignored_name.sm"),
                    use_post_processing=False,
                    result_queue=result_q,
                    all_difficulties=True,
                    export_song_folder=True,
                )
            src, ok, val = result_q.get_nowait()
            self.assertEqual((src, ok), ("generation", True))
            self.assertEqual(pathlib.Path(val), ui_pack / "CLI Song")
            self.assertTrue((ui_pack / "CLI Song" / "CLI Song.sm").is_file())
            self.assertTrue((ui_pack / "CLI Song" / "song.ogg").is_file())

    def test_piecewise_regularize_and_per_section_snap_for_medley(self) -> None:
        # 1. Noisy 2-section medley: 120 BPM (24 beats, 1 dropped) -> 160 BPM (24 beats).
        rng = np.random.default_rng(42)
        sec1_true = 0.5 + np.arange(24, dtype=np.float64) * 0.5
        sec2_true = sec1_true[-1] + 0.375 + np.arange(24, dtype=np.float64) * 0.375
        sec1_noisy = np.delete(sec1_true + rng.normal(0.0, 0.008, 24), 10)
        sec2_noisy = sec2_true + rng.normal(0.0, 0.008, 24)
        raw_medley = beat_tracking.BeatGrid(
            beat_times=np.concatenate([sec1_noisy, sec2_noisy]),
            downbeat_times=np.concatenate([sec1_true[::4], sec2_true[::4]]),
            bpm=138.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        reg = beat_tracking.regularize_beat_grid(raw_medley)
        segs, _ = reg.sm_bpm_segments()
        self.assertLessEqual(len(segs), 3)
        seg_bpms = [bpm for _, bpm in segs]
        self.assertTrue(any(abs(b - 120.0) < 0.6 for b in seg_bpms))
        self.assertTrue(any(abs(b - 160.0) < 0.6 for b in seg_bpms))

        # 2. Opposite phase errors across medley sections: sec1 +20ms late, sec2 -18ms early.
        sr = constants.TARGET_SR
        duration = float(sec2_true[-1]) + 1.5
        y = np.zeros(int(duration * sr), dtype=np.float32)
        true_all = np.concatenate([sec1_true, sec2_true])
        burst = np.exp(-np.arange(256, dtype=np.float32) / 40.0) * np.sin(
            2.0 * np.pi * 1000.0 * np.arange(256, dtype=np.float32) / float(sr)
        )
        for t in true_all:
            idx = int(round(float(t) * sr))
            end = min(y.size, idx + burst.size)
            y[idx:end] += burst[: end - idx]

        biased_grid = beat_tracking.BeatGrid(
            beat_times=np.concatenate([sec1_true + 0.020, sec2_true - 0.018]),
            downbeat_times=np.concatenate(
                [sec1_true[::4] + 0.020, sec2_true[::4] - 0.018]
            ),
            bpm=140.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        with mock.patch.object(
            beat_tracking.librosa,
            "load",
            return_value=(y, sr),
            autospec=True,
        ):
            snapped = beat_tracking.snap_beats_to_onsets(
                biased_grid, "medley.ogg", per_section=True
            )

        self.assertEqual(snapped.beat_times.size, true_all.size)
        err_sec1 = np.abs(snapped.beat_times[:24] - sec1_true)
        err_sec2 = np.abs(snapped.beat_times[24:] - sec2_true)
        self.assertLess(float(np.median(err_sec1)), 0.004)
        self.assertLess(float(np.median(err_sec2)), 0.004)

    def test_octave_aware_regularize_unifies_flickers_and_folds_double_time(
        self,
    ) -> None:
        rng = np.random.default_rng(123)
        # 1. Constant 210 BPM track where tracker flickers 105 -> 420 -> 210 BPM.
        ibi_210 = 60.0 / 210.0
        base_210 = 0.4 + np.arange(160, dtype=np.float64) * ibi_210
        part_half = base_210[:80:2]  # 105 BPM (0.5x)
        part_double = 0.4 + np.arange(160, 200, dtype=np.float64) * (0.5 * ibi_210)
        part_full = base_210[100:160]  # 210 BPM (1x)
        raw_flicker = np.sort(
            np.concatenate([part_half, part_double, part_full])
            + rng.normal(
                0.0, 0.003, size=part_half.size + part_double.size + part_full.size
            )
        )
        flicker_grid = beat_tracking.BeatGrid(
            beat_times=raw_flicker,
            downbeat_times=base_210[::4],
            bpm=140.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        unified = beat_tracking.regularize_beat_grid(flicker_grid)
        segs, _ = unified.sm_bpm_segments()
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(unified.bpm, 210.0, places=2)
        # Every beat sits within 5ms of the true 210 BPM lattice (exact parity!).
        dists = [float(np.min(np.abs(base_210 - t))) for t in unified.beat_times[:120]]
        self.assertLess(float(np.median(dists)), 0.005)

        # 2. Genuine medley (186 BPM -> 422 BPM double-time of 211 BPM) must NOT
        # collapse to 1 BPM, and must fold the 422 BPM section down to ~211 BPM.
        sec1 = 0.5 + np.arange(40, dtype=np.float64) * (60.0 / 186.0)
        sec2_211 = sec1[-1] + 0.3 + np.arange(32, dtype=np.float64) * (60.0 / 211.0)
        sec2_422 = sec1[-1] + 0.3 + np.arange(64, dtype=np.float64) * (60.0 / 422.0)
        medley_grid = beat_tracking.BeatGrid(
            beat_times=np.concatenate([sec1, sec2_422]),
            downbeat_times=np.concatenate([sec1[::4], sec2_211[::4]]),
            bpm=280.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        reg_medley = beat_tracking.regularize_beat_grid(medley_grid)
        m_segs, _ = reg_medley.sm_bpm_segments()
        self.assertGreaterEqual(len(m_segs), 2)
        seg_bpms = [bpm for _, bpm in m_segs]
        self.assertTrue(any(abs(b - 186.0) < 0.6 for b in seg_bpms))
        self.assertTrue(any(abs(b - 211.0) < 0.6 for b in seg_bpms))
        self.assertTrue(all(b <= 255.0 for b in seg_bpms))

    def test_downbeat_bar_phase_alignment_and_octave_folded_medley(self) -> None:
        # 1. Octave-folded medley: 186 BPM with mid-section 93 BPM (0.5x) dip ->
        #    211/422 BPM mixed-octave section must stay 2 clean sections (~186 & ~211).
        rng = np.random.default_rng(77)
        ibi1 = 60.0 / 186.0
        s1_full = 0.5 + np.arange(80, dtype=np.float64) * ibi1
        s1_raw = np.concatenate([s1_full[:40], s1_full[40:60:2], s1_full[60:]])
        ibi2 = 60.0 / 211.0
        s2_start = float(s1_full[-1]) + ibi2
        s2_base = s2_start + np.arange(60, dtype=np.float64) * ibi2
        s2_dbl = s2_start + np.arange(40, 100, dtype=np.float64) * (0.5 * ibi2)
        s2_raw = np.sort(np.concatenate([s2_base[:20], s2_dbl, s2_base[50:]]))
        medley_bt = np.sort(
            np.concatenate([s1_raw, s2_raw])
            + rng.normal(0.0, 0.003, size=s1_raw.size + s2_raw.size)
        )
        medley_db = np.concatenate([s1_full[::4], s2_base[::4]])
        grid = beat_tracking.BeatGrid(
            beat_times=medley_bt,
            downbeat_times=medley_db,
            bpm=186.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        reg = beat_tracking.regularize_beat_grid(grid)
        segs, _ = reg.sm_bpm_segments()
        seg_bpms = [b for _, b in segs]
        self.assertTrue(any(abs(b - 186.0) < 0.5 for b in seg_bpms))
        self.assertTrue(any(abs(b - 211.0) < 0.6 for b in seg_bpms))
        self.assertTrue(all(b <= 255.0 for b in seg_bpms))

        # 2. Anti-phantom-sublattice guard: noisy single-octave 124 BPM grid with
        #    zero 248 BPM intervals must NOT double into 248 BPM.
        ibi124 = 60.0 / 124.0
        noisy124 = 0.3 + np.arange(64, dtype=np.float64) * ibi124
        noisy124 = noisy124 + rng.normal(0.0, 0.022, size=noisy124.size)
        noisy124 = np.sort(noisy124)
        g124 = beat_tracking.BeatGrid(
            beat_times=noisy124,
            downbeat_times=np.zeros(0, dtype=np.float64),
            bpm=124.0,
            tracker=beat_tracking.TRACKER_LIBROSA,
        )
        reg124 = beat_tracking.regularize_beat_grid(g124)
        self.assertLess(reg124.bpm, 150.0)

        # 3. Downbeat bar-phase alignment:
        #    (a) db0_idx in 1..3 trims leading beats so beat 0 is a downbeat.
        bt16 = 0.2 + np.arange(16, dtype=np.float64) * 0.5
        bm16 = np.arange(16 * 24 * 4, dtype=np.float32).reshape(16, 24, 4)
        g_lead2 = beat_tracking.BeatGrid(
            beat_times=bt16,
            downbeat_times=bt16[2::4],
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        a_bm, a_grid, min_t = beat_tracking._align_inputs_to_downbeat_phase(
            bm16, g_lead2
        )
        self.assertEqual(a_bm.shape[0], 14)
        self.assertAlmostEqual(float(a_grid.beat_times[0]), float(bt16[2]), places=6)
        self.assertIsNone(min_t)

        #    (b) db0_idx >= 4 (e.g. idx=5) left-pads 3 zero beats without dropping audio.
        g_deep = beat_tracking.BeatGrid(
            beat_times=bt16,
            downbeat_times=bt16[5::4],
            bpm=120.0,
            tracker=beat_tracking.TRACKER_BEAT_THIS,
        )
        p_bm, p_grid, p_min_t = beat_tracking._align_inputs_to_downbeat_phase(
            bm16, g_deep
        )
        self.assertEqual(p_bm.shape[0], 19)
        self.assertEqual(float(np.abs(p_bm[:3]).sum()), 0.0)
        self.assertIsNotNone(p_min_t)
        db0_new_idx = int(np.argmin(np.abs(p_grid.beat_times - bt16[5])))
        self.assertEqual(db0_new_idx % 4, 0)


if __name__ == "__main__":
    unittest.main()
