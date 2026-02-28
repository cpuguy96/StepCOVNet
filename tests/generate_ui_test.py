"""Unit tests for the generator UI script (scripts/generate_ui.py)."""

import os
import queue
import sys
import tempfile
import unittest
from unittest import mock

# Allow importing the script module (scripts/generate_ui.py)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import generate_ui  # noqa: E402

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class ValidateInputsTest(unittest.TestCase):
    """Tests for _validate_inputs."""

    def _valid_args(self):
        return (
            "/path/to/audio.mp3",
            "My Song",
            "120",
            "/path/to/onset.keras",
            "/path/to/arrow.keras",
            "/path/to/output.txt",
        )

    def test_validate_inputs_success(self):
        ok, result = generate_ui._validate_inputs(*self._valid_args())
        self.assertTrue(ok)
        self.assertIsInstance(result, tuple)
        audio_path, song_title, bpm, onset_path, arrow_path, output_path = result
        self.assertEqual(audio_path, "/path/to/audio.mp3")
        self.assertEqual(song_title, "My Song")
        self.assertEqual(bpm, 120)
        self.assertEqual(onset_path, "/path/to/onset.keras")
        self.assertEqual(arrow_path, "/path/to/arrow.keras")
        self.assertEqual(output_path, "/path/to/output.txt")

    def test_validate_inputs_success_strips_whitespace(self):
        ok, result = generate_ui._validate_inputs(
            "  /path/to/audio.mp3  ",
            "  My Song  ",
            "  120  ",
            "  /path/to/onset.keras  ",
            "  /path/to/arrow.keras  ",
            "  /path/to/output.txt  ",
        )
        self.assertTrue(ok)
        self.assertEqual(result[0], "/path/to/audio.mp3")
        self.assertEqual(result[1], "My Song")
        self.assertEqual(result[2], 120)

    def test_validate_inputs_missing_audio(self):
        args = list(self._valid_args())
        args[0] = ""
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "Please select an audio file.")

    def test_validate_inputs_missing_song_title(self):
        args = list(self._valid_args())
        args[1] = ""
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "Please enter a song title.")

    def test_validate_inputs_missing_bpm(self):
        args = list(self._valid_args())
        args[2] = ""
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "Please enter BPM.")

    def test_validate_inputs_bpm_not_integer(self):
        args = list(self._valid_args())
        args[2] = "not a number"
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "BPM must be an integer.")

    def test_validate_inputs_bpm_zero(self):
        args = list(self._valid_args())
        args[2] = "0"
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "BPM must be between 1 and 9999.")

    def test_validate_inputs_bpm_negative(self):
        args = list(self._valid_args())
        args[2] = "-1"
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "BPM must be between 1 and 9999.")

    def test_validate_inputs_bpm_too_high(self):
        args = list(self._valid_args())
        args[2] = "10000"
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "BPM must be between 1 and 9999.")

    def test_validate_inputs_bpm_boundary_valid(self):
        for bpm_str in ("1", "9999"):
            with self.subTest(bpm=bpm_str):
                args = list(self._valid_args())
                args[2] = bpm_str
                ok, result = generate_ui._validate_inputs(*args)
                self.assertTrue(ok)
                self.assertEqual(result[2], int(bpm_str))

    def test_validate_inputs_missing_onset_path(self):
        args = list(self._valid_args())
        args[3] = ""
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "Please select the onset model.")

    def test_validate_inputs_missing_arrow_path(self):
        args = list(self._valid_args())
        args[4] = ""
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "Please select the arrow model.")

    def test_validate_inputs_missing_output_path(self):
        args = list(self._valid_args())
        args[5] = ""
        ok, msg = generate_ui._validate_inputs(*args)
        self.assertFalse(ok)
        self.assertEqual(msg, "Please choose an output file.")


class RunGenerationTest(unittest.TestCase):
    """Tests for _run_generation."""

    def test_run_generation_success(self):
        """_run_generation loads models, runs generator, writes file and puts (True, path) in queue."""
        mock_output = "TITLE Test\nBPM 120\nNOTES\nDIFFICULTY Challenge\n1.0 0001\n"
        mock_output_data = mock.MagicMock()
        mock_output_data.generate_txt_output.return_value = mock_output

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            output_path = f.name
        self.addCleanup(lambda: os.path.exists(output_path) and os.unlink(output_path))

        result_queue = queue.Queue()
        with (
            mock.patch(
                "generate_ui.keras.models.load_model",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "generate_ui.generator.generate_output_data",
                return_value=mock_output_data,
            ),
        ):
            generate_ui._run_generation(
                audio_path="/nonexistent/audio.mp3",
                song_title="Test",
                bpm=120,
                onset_model_path="/p/onset.keras",
                arrow_model_path="/p/arrow.keras",
                output_path=output_path,
                use_post_processing=False,
                result_queue=result_queue,
            )

        success, value = result_queue.get_nowait()
        self.assertTrue(success)
        self.assertEqual(value, output_path)
        with open(output_path) as f:
            self.assertEqual(f.read(), mock_output)

    def test_run_generation_writes_txt_format(self):
        """Written file contains TITLE, BPM, NOTES, DIFFICULTY."""
        mock_output_data = mock.MagicMock()
        mock_output_data.generate_txt_output.return_value = (
            "TITLE My Song\nBPM 128\nNOTES\nDIFFICULTY Challenge\n0.5 1000\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            output_path = f.name
        self.addCleanup(lambda: os.path.exists(output_path) and os.unlink(output_path))

        result_queue = queue.Queue()
        with (
            mock.patch("generate_ui.keras.models.load_model", return_value=mock.MagicMock()),
            mock.patch(
                "generate_ui.generator.generate_output_data",
                return_value=mock_output_data,
            ),
        ):
            generate_ui._run_generation(
                audio_path="/a.mp3",
                song_title="My Song",
                bpm=128,
                onset_model_path="/o.keras",
                arrow_model_path="/ar.keras",
                output_path=output_path,
                use_post_processing=True,
                result_queue=result_queue,
            )

        success, _ = result_queue.get_nowait()
        self.assertTrue(success)
        with open(output_path) as f:
            content = f.read()
        self.assertIn("TITLE My Song", content)
        self.assertIn("BPM 128", content)
        self.assertIn("NOTES", content)
        self.assertIn("DIFFICULTY Challenge", content)

    def test_run_generation_load_model_raises(self):
        """When load_model raises, queue receives (False, error_message)."""
        result_queue = queue.Queue()
        with mock.patch(
            "generate_ui.keras.models.load_model",
            side_effect=OSError("No such file"),
        ):
            generate_ui._run_generation(
                audio_path="/a.mp3",
                song_title="Song",
                bpm=120,
                onset_model_path="/onset.keras",
                arrow_model_path="/arrow.keras",
                output_path="/out.txt",
                use_post_processing=False,
                result_queue=result_queue,
            )

        success, value = result_queue.get_nowait()
        self.assertFalse(success)
        self.assertIn("No such file", value)

    def test_run_generation_generator_raises(self):
        """When generate_output_data raises, queue receives (False, error_message)."""
        result_queue = queue.Queue()
        with (
            mock.patch("generate_ui.keras.models.load_model", return_value=mock.MagicMock()),
            mock.patch(
                "generate_ui.generator.generate_output_data",
                side_effect=ValueError("Failed to predict any onsets"),
            ),
        ):
            generate_ui._run_generation(
                audio_path="/a.mp3",
                song_title="Song",
                bpm=120,
                onset_model_path="/o.keras",
                arrow_model_path="/a.keras",
                output_path="/out.txt",
                use_post_processing=False,
                result_queue=result_queue,
            )

        success, value = result_queue.get_nowait()
        self.assertFalse(success)
        self.assertIn("Failed to predict any onsets", value)


class ConstantsTest(unittest.TestCase):
    """Tests for file type constants used by file dialogs."""

    def test_audio_types_defined(self):
        self.assertIsInstance(generate_ui.AUDIO_TYPES, list)
        self.assertGreater(len(generate_ui.AUDIO_TYPES), 0)

    def test_model_types_defined(self):
        self.assertIsInstance(generate_ui.MODEL_TYPES, list)
        self.assertGreater(len(generate_ui.MODEL_TYPES), 0)

    def test_txt_types_defined(self):
        self.assertIsInstance(generate_ui.TXT_TYPES, list)
        self.assertGreater(len(generate_ui.TXT_TYPES), 0)


if __name__ == "__main__":
    unittest.main()
