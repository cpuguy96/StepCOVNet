import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import keras
import numpy as np

from stepcovnet import (
    config,
    generator,
    models,  # Required to ensure registration of custom Keras layers/functions for model loading.
)

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "testdata"


def _keras_predict_stub(*, predict_side_effect, inputs=None):
    model = mock.create_autospec(keras.Model, instance=True)
    model.predict.side_effect = predict_side_effect
    if inputs is not None:
        model.inputs = inputs
    return model


def _mock_arrow_input(name: str):
    """Minimal mock input with .name for generator's input-name dispatch."""
    return SimpleNamespace(name=name)


class GeneratorTest(unittest.TestCase):
    def test_generate_output_data_with_mock_models(self):

        def _onset_pred_mock(x):
            self.assertEqual(x.shape, (1, 11726, 128))
            return np.random.random((1, 11726, 1)).astype(np.float32)

        mock_onset_model = _keras_predict_stub(predict_side_effect=_onset_pred_mock)

        def _arrow_pred_mock(x):
            if isinstance(x, list):
                x = x[0]
            num_arrows = x.shape[1]
            self.assertEqual(x.shape, (1, num_arrows, 1))
            return np.random.random((1, num_arrows, 256)).astype(np.float32)

        mock_arrow_model = _keras_predict_stub(
            predict_side_effect=_arrow_pred_mock,
            inputs=[_mock_arrow_input("timing_input")],
        )

        for use_post_processing in [True, False]:
            with self.subTest(f"use_post_processing={use_post_processing}"):
                output_data = generator.generate_output_data(
                    audio_path=TEST_DATA_DIR / "mayu.ogg",
                    song_title="Test Song",
                    bpm=120,
                    onset_model=mock_onset_model,
                    arrow_model=mock_arrow_model,
                    use_post_processing=use_post_processing,
                )
                self.assertEqual(output_data.title, "Test Song")
                self.assertEqual(output_data.bpm, 120)
                self.assertTrue("Challenge" in output_data.notes)
                self.assertLessEqual(len(output_data.notes["Challenge"]), 11726)
                for onset, arrow in output_data.notes["Challenge"]:
                    self.assertNotIn("4", arrow)
                    # 0 is used as padding for training datasets. So there should be none present.
                    self.assertNotEqual(arrow, "0000")
                    self.assertGreaterEqual(float(onset), 0)
                    self.assertLessEqual(float(onset), 129)

    def test_generate_output_data_with_two_input_arrow_model(self):
        """generate_output_data works when arrow model has two inputs (snippets)."""
        onset_model = keras.models.load_model(
            TEST_DATA_DIR / "stepcovnet_ONSET-mayu_overfit.keras",
            compile=False,
        )
        arrow_model = models.build_arrow_model(
            models.ArrowInputOptions(snippet_half_frames=5),
            models.ArrowOutputOptions(),
            config.TransformerArrowParams(),
        )
        output_data = generator.generate_output_data(
            audio_path=TEST_DATA_DIR / "mayu.ogg",
            song_title="M.A.Y.U",
            bpm=128,
            onset_model=onset_model,  # type: ignore[arg-type]
            arrow_model=arrow_model,  # type: ignore[arg-type]
        )
        self.assertEqual(output_data.title, "M.A.Y.U")
        self.assertEqual(output_data.bpm, 128)
        self.assertIn("Challenge", output_data.notes)
        self.assertGreater(len(output_data.notes["Challenge"]), 0)

    def test_two_input_arrow_model_uses_n_frames_not_n_mels_for_snippets(self):
        """Snippet n_frames is taken from input shape [2], not [3] (n_mels).

        Snippet input is always (batch, steps, n_frames, n_mels); see models.build_arrow_model.
        """
        n_frames_window = 11  # 2 * 5 + 1 for half_frames=5
        n_mels = 128
        snippet_shape = (None, None, n_frames_window, n_mels)

        def _onset_pred_mock(x):
            return np.random.random((1, x.shape[1], 1)).astype(np.float32)

        mock_onset = _keras_predict_stub(predict_side_effect=_onset_pred_mock)

        call_args = []

        def _arrow_pred_mock(x):
            call_args.append(x)
            num_steps = x[0].shape[1] if isinstance(x, list) else x.shape[1]
            return np.random.random((1, num_steps, 256)).astype(np.float32)

        mock_arrow = _keras_predict_stub(predict_side_effect=_arrow_pred_mock)
        mock_arrow.inputs = [
            _mock_arrow_input("timing_input"),
            SimpleNamespace(name="snippet_input", shape=snippet_shape),
        ]
        mock_arrow.input_shape = [(None, None, 1), snippet_shape]

        generator.generate_output_data(
            audio_path=TEST_DATA_DIR / "mayu.ogg",
            song_title="Snippet shape test",
            bpm=120,
            onset_model=mock_onset,
            arrow_model=mock_arrow,
        )
        self.assertEqual(len(call_args), 1)
        args = call_args[0]
        self.assertIsInstance(args, list)
        snippets_batch = args[1]
        self.assertEqual(
            snippets_batch.shape[2],
            n_frames_window,
            "Snippet n_frames must come from input_shape[1][2], not [3] (n_mels)",
        )
        self.assertEqual(snippets_batch.shape[3], n_mels)

    def test_generate_output_data_matches_input_names_with_keras_suffix(self):
        """Input names with ':0' suffix (Keras/TF tensor names) are matched by base name."""

        def _onset_pred_mock(x):
            return np.random.random((1, x.shape[1], 1)).astype(np.float32)

        mock_onset = _keras_predict_stub(predict_side_effect=_onset_pred_mock)

        call_args = []

        def _arrow_pred_mock(x):
            call_args.append(x)
            num_steps = x[0].shape[1] if isinstance(x, list) else x.shape[1]
            return np.random.random((1, num_steps, 256)).astype(np.float32)

        mock_arrow = _keras_predict_stub(predict_side_effect=_arrow_pred_mock)
        mock_arrow.inputs = [
            _mock_arrow_input("timing_input:0"),
            _mock_arrow_input("interval_input:0"),
        ]

        generator.generate_output_data(
            audio_path=TEST_DATA_DIR / "mayu.ogg",
            song_title="Keras suffix test",
            bpm=120,
            onset_model=mock_onset,
            arrow_model=mock_arrow,
        )
        self.assertEqual(len(call_args), 1)
        args = call_args[0]
        self.assertIsInstance(args, list)
        self.assertEqual(len(args), 2, "Both timing and interval inputs must be passed")

    def test_generate_output_data_with_interval_input(self):
        """When arrow model has timing_input and interval_input, generator passes both."""

        def _onset_pred_mock(x):
            return np.random.random((1, x.shape[1], 1)).astype(np.float32)

        mock_onset = _keras_predict_stub(predict_side_effect=_onset_pred_mock)

        call_args = []

        def _arrow_pred_mock(x):
            call_args.append(x)
            num_steps = x[0].shape[1] if isinstance(x, list) else x.shape[1]
            return np.random.random((1, num_steps, 256)).astype(np.float32)

        mock_arrow = _keras_predict_stub(predict_side_effect=_arrow_pred_mock)
        mock_arrow.inputs = [
            _mock_arrow_input("timing_input"),
            _mock_arrow_input("interval_input"),
        ]

        generator.generate_output_data(
            audio_path=TEST_DATA_DIR / "mayu.ogg",
            song_title="Interval test",
            bpm=120,
            onset_model=mock_onset,
            arrow_model=mock_arrow,
        )
        self.assertEqual(len(call_args), 1)
        args = call_args[0]
        self.assertIsInstance(args, list)
        self.assertEqual(len(args), 2)
        timing_batch, interval_batch = args[0], args[1]
        self.assertEqual(timing_batch.shape[2], 1)
        self.assertEqual(interval_batch.shape, timing_batch.shape)
        self.assertGreaterEqual(interval_batch.min(), 0.0)
        self.assertLessEqual(interval_batch.max(), 1.0)

    def _run_generate_with_extra_input_and_capture(
        self, extra_input_name: str, bpm: int = 120
    ):
        """Run generate_output_data with timing_input + one extra input; return predict args."""

        def _onset_pred_mock(x):
            return np.random.random((1, x.shape[1], 1)).astype(np.float32)

        mock_onset = _keras_predict_stub(predict_side_effect=_onset_pred_mock)

        call_args = []

        def _arrow_pred_mock(x):
            call_args.append(x)
            num_steps = x[0].shape[1] if isinstance(x, list) else x.shape[1]
            return np.random.random((1, num_steps, 256)).astype(np.float32)

        mock_arrow = _keras_predict_stub(predict_side_effect=_arrow_pred_mock)
        mock_arrow.inputs = [
            _mock_arrow_input("timing_input"),
            _mock_arrow_input(extra_input_name),
        ]

        generator.generate_output_data(
            audio_path=TEST_DATA_DIR / "mayu.ogg",
            song_title="Extra input test",
            bpm=bpm,
            onset_model=mock_onset,
            arrow_model=mock_arrow,
        )
        self.assertEqual(len(call_args), 1)
        args = call_args[0]
        self.assertIsInstance(args, list)
        self.assertEqual(len(args), 2)
        return args[0], args[1]

    def test_generate_output_data_with_interval_log_input(self):
        """When arrow model has interval_log_input, generator passes log-normalized intervals."""
        timing_batch, interval_log_batch = (
            self._run_generate_with_extra_input_and_capture("interval_log_input")
        )
        self.assertEqual(timing_batch.shape[2], 1)
        self.assertEqual(interval_log_batch.shape, timing_batch.shape)
        self.assertGreaterEqual(interval_log_batch.min(), 0.0)
        self.assertLessEqual(interval_log_batch.max(), 1.0)

    def test_generate_output_data_with_interval_next_input(self):
        """When arrow model has interval_next_input, generator passes next-interval normalized."""
        timing_batch, interval_next_batch = (
            self._run_generate_with_extra_input_and_capture("interval_next_input")
        )
        self.assertEqual(timing_batch.shape[2], 1)
        self.assertEqual(interval_next_batch.shape, timing_batch.shape)
        self.assertGreaterEqual(interval_next_batch.min(), 0.0)
        self.assertLessEqual(interval_next_batch.max(), 1.0)

    def test_generate_output_data_with_step_index_input(self):
        """When arrow model has step_index_input, generator passes normalized step indices."""
        timing_batch, step_index_batch = (
            self._run_generate_with_extra_input_and_capture("step_index_input")
        )
        self.assertEqual(timing_batch.shape[2], 1)
        self.assertEqual(step_index_batch.shape, timing_batch.shape)
        self.assertGreaterEqual(step_index_batch.min(), 0.0)
        self.assertLessEqual(step_index_batch.max(), 1.0)

    def test_generate_output_data_with_beat_phase_input(self):
        """When arrow model has beat_phase_input, generator passes beat phase using bpm."""
        timing_batch, beat_phase_batch = (
            self._run_generate_with_extra_input_and_capture("beat_phase_input", bpm=128)
        )
        self.assertEqual(timing_batch.shape[2], 1)
        self.assertEqual(beat_phase_batch.shape, timing_batch.shape)
        self.assertGreaterEqual(beat_phase_batch.min(), 0.0)
        self.assertLess(beat_phase_batch.max(), 1.0)

    def test_generate_output_data(self):
        onset_model = keras.models.load_model(
            TEST_DATA_DIR / "stepcovnet_ONSET-mayu_overfit.keras",
            compile=False,
        )
        arrow_model = keras.models.load_model(
            TEST_DATA_DIR / "stepcovnet_ARROW-mayu_overfit.keras",
            compile=False,
        )

        for use_post_processing in [True, False]:
            with self.subTest(f"use_post_processing={use_post_processing}"):
                output_data = generator.generate_output_data(
                    audio_path=TEST_DATA_DIR / "mayu.ogg",
                    song_title="M.A.Y.U",
                    bpm=128,
                    onset_model=onset_model,  # type: ignore
                    arrow_model=arrow_model,  # type: ignore
                )
                self.assertEqual(output_data.title, "M.A.Y.U")
                self.assertEqual(output_data.bpm, 128)
                self.assertTrue("Challenge" in output_data.notes)
                self.assertEqual(len(output_data.notes["Challenge"]), 384)
                self.assertEqual(("7.48", "2000"), output_data.notes["Challenge"][0])
                for onset, arrow in output_data.notes["Challenge"]:
                    self.assertNotIn("4", arrow)
                    # 0 is used as padding for training datasets. So there
                    # should be none present.
                    self.assertNotEqual(arrow, "0000")
                    self.assertGreaterEqual(float(onset), 7)
                    self.assertLessEqual(float(onset), 109)

    def test_output_data_generate_txt_output(self):
        output_data = generator.OutputData(
            title="Test Song", bpm=120, notes={"Challenge": [("3103", "1.04")]}
        )
        expected_output = (
            "TITLE Test Song\nBPM 120\nNOTES\nDIFFICULTY Challenge\n1.04 3103\n"
        )
        self.assertEqual(output_data.generate_txt_output(), expected_output)

    def test_generate_output_data_bpm_none_estimates_from_audio(self):
        """When bpm is None, BPM is estimated from audio and used in output."""

        def _onset_pred_mock(x):
            return np.random.random((1, x.shape[1], 1)).astype(np.float32)

        mock_onset = _keras_predict_stub(predict_side_effect=_onset_pred_mock)

        def _arrow_pred_mock(x):
            x_in = x[0] if isinstance(x, list) else x
            return np.random.random((1, x_in.shape[1], 256)).astype(np.float32)

        mock_arrow = _keras_predict_stub(
            predict_side_effect=_arrow_pred_mock,
            inputs=[_mock_arrow_input("timing_input")],
        )

        audio_path = TEST_DATA_DIR / "mayu.ogg"
        with mock.patch.object(
            generator,
            "_estimate_bpm_from_audio",
            return_value=120,
            autospec=True,
        ) as mock_estimate:
            output_data = generator.generate_output_data(
                audio_path=audio_path,
                song_title="Test",
                bpm=None,
                onset_model=mock_onset,
                arrow_model=mock_arrow,
            )
        self.assertEqual(output_data.bpm, 120)
        mock_estimate.assert_called_once_with(audio_path)

    def test_estimate_bpm_from_audio_success_returns_bpm(self):
        """_estimate_bpm_from_audio returns integer BPM when librosa returns valid tempo."""
        with (
            mock.patch.object(generator.librosa, "load", autospec=True) as mock_load,
            mock.patch.object(
                generator.librosa.beat, "beat_track", autospec=True
            ) as mock_beat,
        ):
            mock_load.return_value = (np.zeros(44100), 44100)
            mock_beat.return_value = (120.0, np.array([0, 1, 2]))
            result = generator._estimate_bpm_from_audio("/fake/path.ogg")
        self.assertEqual(result, 120)
        mock_load.assert_called_once_with("/fake/path.ogg", sr=44100)
        mock_beat.assert_called_once()

    def test_estimate_bpm_from_audio_success_clamps_to_valid_range(self):
        """_estimate_bpm_from_audio clamps BPM to [1, 9999]."""
        with (
            mock.patch.object(generator.librosa, "load", autospec=True) as mock_load,
            mock.patch.object(
                generator.librosa.beat,
                "beat_track",
                side_effect=[(20000.0, np.array([])), (0.4, np.array([]))],
                autospec=True,
            ),
        ):
            mock_load.return_value = (np.zeros(44100), 44100)
            high = generator._estimate_bpm_from_audio("/fake.ogg")
            low = generator._estimate_bpm_from_audio("/fake2.ogg")
        self.assertEqual(high, 9999)
        self.assertEqual(low, 1)

    def test_estimate_bpm_from_audio_failure_raises(self):
        """_estimate_bpm_from_audio raises ValueError when tempo is 0 or invalid."""
        with (
            mock.patch.object(generator.librosa, "load", autospec=True) as mock_load,
            mock.patch.object(
                generator.librosa.beat, "beat_track", autospec=True
            ) as mock_beat,
        ):
            mock_load.return_value = (np.zeros(44100), 44100)
            mock_beat.return_value = (0.0, np.array([]))
            with self.assertRaises(ValueError) as ctx:
                generator._estimate_bpm_from_audio("/fake/path.ogg")
        self.assertIn("Could not estimate BPM", str(ctx.exception))
        self.assertIn("--bpm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
