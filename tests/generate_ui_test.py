import io
import os
import queue
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import unittest
from unittest import mock

# Allow importing the script module (scripts/generate_ui.py)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import generate_ui  # noqa: E402

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _make_app():
    """Create a Tk root and _GeneratorApp; returns (root, app). Caller must root.destroy()."""
    try:
        root = tk.Tk()
    except tk.TclError as e:
        raise unittest.SkipTest(f"Tk unavailable: {e}") from e
    root.withdraw()
    app = generate_ui._GeneratorApp(root)
    return root, app


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

    def test_validate_inputs_missing_bpm_valid_returns_none(self):
        """Empty BPM is valid; result[2] is None (BPM will be estimated from audio)."""
        args = list(self._valid_args())
        args[2] = ""
        ok, result = generate_ui._validate_inputs(*args)
        self.assertTrue(ok)
        self.assertIsNone(result[2])

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

    def test_validate_inputs_empty_onset_path_valid_returns_none(self):
        """Empty onset path is valid; result[3] is None (will resolve/download in worker)."""
        args = list(self._valid_args())
        args[3] = ""
        ok, result = generate_ui._validate_inputs(*args)
        self.assertTrue(ok)
        self.assertIsNone(result[3])

    def test_validate_inputs_empty_arrow_path_valid_returns_none(self):
        """Empty arrow path is valid; result[4] is None (will resolve/download in worker)."""
        args = list(self._valid_args())
        args[4] = ""
        ok, result = generate_ui._validate_inputs(*args)
        self.assertTrue(ok)
        self.assertIsNone(result[4])

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

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            result_queue = queue.Queue()
            with (
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_onset_model_path",
                    side_effect=lambda p: p or os.path.join(tmpdir, "onset.keras"),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_arrow_model_path",
                    side_effect=lambda p: p or os.path.join(tmpdir, "arrow.keras"),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.keras.models,
                    "load_model",
                    return_value=mock.MagicMock(),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.generator,
                    "generate_output_data",
                    return_value=mock_output_data,
                    autospec=True,
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

            source, success, value = result_queue.get_nowait()
            self.assertEqual(source, "generation")
            self.assertTrue(success)
            self.assertEqual(value, output_path)
            with open(output_path) as f:
                self.assertEqual(f.read(), mock_output)

    def test_run_generation_with_none_bpm_calls_generator_with_none(self):
        """_run_generation with bpm=None calls generate_output_data with bpm=None."""
        mock_output_data = mock.MagicMock()
        mock_output_data.generate_txt_output.return_value = "TITLE X\nBPM 100\nNOTES\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            result_queue = queue.Queue()
            with (
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_onset_model_path",
                    side_effect=lambda p: p or os.path.join(tmpdir, "onset.keras"),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_arrow_model_path",
                    side_effect=lambda p: p or os.path.join(tmpdir, "arrow.keras"),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.keras.models,
                    "load_model",
                    return_value=mock.MagicMock(),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.generator,
                    "generate_output_data",
                    return_value=mock_output_data,
                    autospec=True,
                ) as mock_gen,
            ):
                generate_ui._run_generation(
                    audio_path="/a.mp3",
                    song_title="Song",
                    bpm=None,
                    onset_model_path="/o.keras",
                    arrow_model_path="/ar.keras",
                    output_path=output_path,
                    use_post_processing=False,
                    result_queue=result_queue,
                )

            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            self.assertIsNone(call_kwargs["bpm"])
            source, success, _ = result_queue.get_nowait()
            self.assertEqual(source, "generation")
            self.assertTrue(success)

    def test_run_generation_writes_txt_format(self):
        """Written file contains TITLE, BPM, NOTES, DIFFICULTY."""
        mock_output_data = mock.MagicMock()
        mock_output_data.generate_txt_output.return_value = (
            "TITLE My Song\nBPM 128\nNOTES\nDIFFICULTY Challenge\n0.5 1000\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            result_queue = queue.Queue()
            with (
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_onset_model_path",
                    side_effect=lambda p: p or os.path.join(tmpdir, "onset.keras"),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_arrow_model_path",
                    side_effect=lambda p: p or os.path.join(tmpdir, "arrow.keras"),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.keras.models,
                    "load_model",
                    return_value=mock.MagicMock(),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.generator,
                    "generate_output_data",
                    return_value=mock_output_data,
                    autospec=True,
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

            source, success, _ = result_queue.get_nowait()
            self.assertEqual(source, "generation")
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
        with (
            mock.patch.object(
                generate_ui.pretrained,
                "resolve_onset_model_path",
                side_effect=lambda p: p or "/resolved/onset.keras",
            ),
            mock.patch.object(
                generate_ui.pretrained,
                "resolve_arrow_model_path",
                side_effect=lambda p: p or "/resolved/arrow.keras",
            ),
            mock.patch.object(
                generate_ui.keras.models,
                "load_model",
                side_effect=OSError("No such file"),
            ),
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

        source, success, value = result_queue.get_nowait()
        self.assertEqual(source, "generation")
        self.assertFalse(success)
        self.assertIn("No such file", value)

    def test_run_generation_generator_raises(self):
        """When generate_output_data raises, queue receives (False, error_message)."""
        result_queue = queue.Queue()
        with (
            mock.patch.object(
                generate_ui.pretrained,
                "resolve_onset_model_path",
                side_effect=lambda p: p or "/resolved/onset.keras",
            ),
            mock.patch.object(
                generate_ui.pretrained,
                "resolve_arrow_model_path",
                side_effect=lambda p: p or "/resolved/arrow.keras",
            ),
            mock.patch.object(
                generate_ui.keras.models,
                "load_model",
                return_value=mock.MagicMock(),
                autospec=True,
            ),
            mock.patch.object(
                generate_ui.generator,
                "generate_output_data",
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

        source, success, value = result_queue.get_nowait()
        self.assertEqual(source, "generation")
        self.assertFalse(success)
        self.assertIn("Failed to predict any onsets", value)

    def test_run_generation_with_none_model_paths_calls_resolve(self):
        """When onset/arrow paths are None, resolve_onset_model_path and resolve_arrow_model_path are called."""
        mock_output_data = mock.MagicMock()
        mock_output_data.generate_txt_output.return_value = "TITLE X\nBPM 100\nNOTES\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            onset_resolved = os.path.join(tmpdir, "onset.keras")
            arrow_resolved = os.path.join(tmpdir, "arrow.keras")
            result_queue = queue.Queue()
            with (
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_onset_model_path",
                    return_value=onset_resolved,
                    autospec=True,
                ) as m_resolve_onset,
                mock.patch.object(
                    generate_ui.pretrained,
                    "resolve_arrow_model_path",
                    return_value=arrow_resolved,
                    autospec=True,
                ) as m_resolve_arrow,
                mock.patch.object(
                    generate_ui.keras.models,
                    "load_model",
                    return_value=mock.MagicMock(),
                    autospec=True,
                ),
                mock.patch.object(
                    generate_ui.generator,
                    "generate_output_data",
                    return_value=mock_output_data,
                    autospec=True,
                ),
            ):
                generate_ui._run_generation(
                    audio_path="/a.mp3",
                    song_title="Song",
                    bpm=100,
                    onset_model_path=None,
                    arrow_model_path=None,
                    output_path=output_path,
                    use_post_processing=False,
                    result_queue=result_queue,
                )

            m_resolve_onset.assert_called_once_with(None)
            m_resolve_arrow.assert_called_once_with(None)
            source, success, _ = result_queue.get_nowait()
            self.assertEqual(source, "generation")
            self.assertTrue(success)


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


class MainWindowTest(unittest.TestCase):
    """Tests for main window and _GeneratorApp setup."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_window_title(self):
        self.assertEqual(self.root.title(), "StepCOVNet Generator")

    def test_window_geometry_was_set(self):
        self.root.update_idletasks()
        geom = self.root.geometry()
        if "480" in geom and "540" in geom:
            return
        self.skipTest("geometry not applied when window withdrawn (platform-dependent)")

    def test_window_minsize_was_set(self):
        self.root.update_idletasks()
        try:
            w, h = self.root.winfo_minsize()
            self.assertEqual(w, 400)
            self.assertEqual(h, 320)
        except AttributeError:
            self.skipTest("winfo_minsize not available on this Tk")

    def test_vars_initialized(self):
        self.assertEqual(self.app.audio_path_var.get(), "")
        self.assertEqual(self.app.song_title_var.get(), "")
        self.assertEqual(self.app.bpm_var.get(), "")
        self.assertEqual(self.app.onset_model_var.get(), "")
        self.assertEqual(self.app.arrow_model_var.get(), "")
        self.assertEqual(self.app.output_path_var.get(), "")
        self.assertFalse(self.app.use_post_processing_var.get())
        self.assertEqual(self.app.status_var.get(), "")

    def test_run_btn_and_status_label_exist(self):
        self.assertIsNotNone(self.app.run_btn)
        self.assertIsNotNone(self.app.status_label)
        self.assertEqual(self.app.run_btn["text"], "Generate chart")

    def test_wm_delete_window_protocol_set(self):
        """WM_DELETE_WINDOW is handled so clicking X cleanly shuts down the app."""
        handler = self.root.protocol("WM_DELETE_WINDOW")
        self.assertIsNotNone(handler)
        # Tk may return the callable or an internal name; ensure we have a handler
        if callable(handler):
            handler_name = getattr(handler, "__name__", None)
            if handler_name is not None:
                self.assertEqual(handler_name, "_on_close")


class CloseBehaviorTest(unittest.TestCase):
    """Tests for _on_close and clean shutdown (WM_DELETE_WINDOW)."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_on_close_calls_quit_and_destroy(self):
        """_on_close sets _closing, calls root.quit() and root.destroy()."""
        with (
            mock.patch.object(self.root, "quit", autospec=True) as m_quit,
            mock.patch.object(self.root, "destroy", autospec=True) as m_destroy,
        ):
            self.app._on_close()
        self.assertTrue(self.app._closing)
        m_quit.assert_called_once()
        m_destroy.assert_called_once()

    def test_on_close_cancels_poll_after_id(self):
        """_on_close cancels pending after() callback so mainloop can exit."""
        self.app._poll_after_id = "fake_after_id"
        with (
            mock.patch.object(self.root, "after_cancel", autospec=True) as m_cancel,
            mock.patch.object(self.root, "quit", autospec=True),
            mock.patch.object(self.root, "destroy", autospec=True),
        ):
            self.app._on_close()
        m_cancel.assert_called_once_with("fake_after_id")
        self.assertIsNone(self.app._poll_after_id)

    def test_on_close_handles_tcl_error_from_after_cancel(self):
        """_on_close catches tk.TclError from after_cancel and still clears _poll_after_id and quits."""
        self.app._poll_after_id = "stale_after_id"
        with (
            mock.patch.object(
                self.root,
                "after_cancel",
                side_effect=tk.TclError("invalid command name"),
            ),
            mock.patch.object(self.root, "quit", autospec=True) as m_quit,
            mock.patch.object(self.root, "destroy", autospec=True) as m_destroy,
        ):
            self.app._on_close()
        self.assertIsNone(self.app._poll_after_id)
        m_quit.assert_called_once()
        m_destroy.assert_called_once()

    def test_on_close_idempotent(self):
        """_on_close does nothing if already closing (avoids double destroy)."""
        with (
            mock.patch.object(self.root, "quit", autospec=True) as m_quit,
            mock.patch.object(self.root, "destroy", autospec=True) as m_destroy,
        ):
            self.app._on_close()
            m_quit.reset_mock()
            m_destroy.reset_mock()
            self.app._on_close()
        m_quit.assert_not_called()
        m_destroy.assert_not_called()

    def test_poll_result_does_not_reschedule_when_closing(self):
        """_poll_result does not schedule another poll when _closing is True."""
        self.app._closing = True
        with mock.patch.object(self.root, "after", autospec=True) as m_after:
            self.app._poll_result()
        m_after.assert_not_called()


class BrowseCallbacksTest(unittest.TestCase):
    """Tests for browse_audio, browse_onset, browse_arrow, browse_output."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_browse_audio_sets_path_and_default_output_not_same_as_audio(self):
        """Default output is stepcovnet_chart_basename.txt when output was empty (prefix so stemming won't match)."""
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="C:/music/song.mp3",
            autospec=True,
        ):
            self.app.browse_audio()
        self.assertEqual(self.app.audio_path_var.get(), "C:/music/song.mp3")
        self.assertEqual(
            os.path.normpath(self.app.output_path_var.get()),
            os.path.normpath("C:/music/stepcovnet_chart_song.txt"),
        )

    def test_browse_audio_default_output_uses_song_title_when_set(self):
        """When song title is set, default output filename is stepcovnet_chart_ + song title (sanitized)."""
        self.app.song_title_var.set("My Song")
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="C:/music/anything.ogg",
            autospec=True,
        ):
            self.app.browse_audio()
        self.assertEqual(
            os.path.normpath(self.app.output_path_var.get()),
            os.path.normpath("C:/music/stepcovnet_chart_My Song.txt"),
        )

    def test_default_output_path_for_audio_sanitizes_unsafe_chars(self):
        """Song title with path-unsafe chars is sanitized to underscores; filename uses stepcovnet_chart_ prefix."""
        self.app.song_title_var.set("Title/with:bad*chars?")
        result = self.app._default_output_path_for_audio("C:/dir/file.ogg")
        self.assertEqual(
            os.path.normpath(result),
            os.path.normpath("C:/dir/stepcovnet_chart_Title_with_bad_chars_.txt"),
        )

    def test_browse_audio_does_not_overwrite_output_when_set(self):
        self.app.output_path_var.set("C:/out/custom.txt")
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="C:/music/song.mp3",
            autospec=True,
        ):
            self.app.browse_audio()
        self.assertEqual(self.app.output_path_var.get(), "C:/out/custom.txt")

    def test_browse_audio_cancelled_does_nothing(self):
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="",
            autospec=True,
        ):
            self.app.browse_audio()
        self.assertEqual(self.app.audio_path_var.get(), "")

    def test_browse_onset_sets_path(self):
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="/path/to/onset.keras",
            autospec=True,
        ):
            self.app.browse_onset()
        self.assertEqual(self.app.onset_model_var.get(), "/path/to/onset.keras")

    def test_browse_onset_cancelled_does_nothing(self):
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="",
            autospec=True,
        ):
            self.app.browse_onset()
        self.assertEqual(self.app.onset_model_var.get(), "")

    def test_browse_arrow_sets_path(self):
        with mock.patch.object(
            filedialog,
            "askopenfilename",
            return_value="/path/to/arrow.keras",
            autospec=True,
        ):
            self.app.browse_arrow()
        self.assertEqual(self.app.arrow_model_var.get(), "/path/to/arrow.keras")

    def test_browse_output_sets_path(self):
        with mock.patch.object(
            filedialog,
            "asksaveasfilename",
            return_value="/path/to/chart.txt",
            autospec=True,
        ):
            self.app.browse_output()
        self.assertEqual(self.app.output_path_var.get(), "/path/to/chart.txt")

    def test_browse_output_cancelled_does_nothing(self):
        with mock.patch.object(
            filedialog,
            "asksaveasfilename",
            return_value="",
            autospec=True,
        ):
            self.app.browse_output()
        self.assertEqual(self.app.output_path_var.get(), "")


class SetStatusTest(unittest.TestCase):
    """Tests for set_status."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_set_status_updates_var(self):
        self.app.set_status("Processing…")
        self.root.update_idletasks()
        self.assertEqual(self.app.status_var.get(), "Processing…")

    def test_set_status_calls_update_idletasks(self):
        with mock.patch.object(
            self.app.status_label, "update_idletasks", autospec=True
        ) as m_update:
            self.app.set_status("Done")
        m_update.assert_called_once()


class RunClickedTest(unittest.TestCase):
    """Tests for run_clicked validation and worker scheduling."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_run_clicked_validation_failure_shows_messagebox(self):
        self.app.audio_path_var.set("")
        with mock.patch.object(messagebox, "showerror", autospec=True) as m_showerror:
            self.app.run_clicked()
        m_showerror.assert_called_once()
        self.assertEqual(
            m_showerror.call_args[0], ("Validation", "Please select an audio file.")
        )

    def test_run_clicked_validation_success_disables_btn_sets_status_schedules_poll(
        self,
    ):
        self.app.audio_path_var.set("/a.mp3")
        self.app.song_title_var.set("Song")
        self.app.bpm_var.set("120")
        self.app.onset_model_var.set("/o.keras")
        self.app.arrow_model_var.set("/ar.keras")
        self.app.output_path_var.set("/out.txt")
        after_cbs = []

        def capture_after(ms, cb):
            after_cbs.append((ms, cb))

        with (
            mock.patch.object(
                self.root, "after", side_effect=capture_after, autospec=True
            ),
            mock.patch.object(
                generate_ui,
                "_run_generation",
                side_effect=lambda **kw: kw["result_queue"].put(
                    ("generation", True, "/out.txt")
                ),
                autospec=True,
            ),
        ):
            self.app.run_clicked()

        self.assertEqual(self.app.run_btn["state"], tk.DISABLED)
        self.assertEqual(self.app.status_var.get(), "Processing…")
        self.assertEqual(len(after_cbs), 1)
        self.assertEqual(after_cbs[0][0], 100)
        self.assertIs(after_cbs[0][1].__self__, self.app)
        self.assertEqual(after_cbs[0][1].__name__, "_poll_result")


class PollResultTest(unittest.TestCase):
    """Tests for _poll_result."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_poll_result_empty_reschedules(self):
        with mock.patch.object(self.root, "after", autospec=True) as m_after:
            self.app._poll_result()
        m_after.assert_called_once_with(100, self.app._poll_result)

    def test_poll_result_generation_success_enables_btn_sets_status_shows_info(self):
        self.app.result_queue.put(("generation", True, "C:/out/chart.txt"))
        with mock.patch.object(messagebox, "showinfo", autospec=True) as m_showinfo:
            self.app._poll_result()
        self.assertEqual(self.app.run_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.status_var.get(), "Saved to C:/out/chart.txt")
        m_showinfo.assert_called_once()
        self.assertIn("chart.txt", m_showinfo.call_args[0][1])

    def test_poll_result_generation_failure_sets_error_status_shows_showerror(self):
        self.app.result_queue.put(("generation", False, "Load error"))
        with mock.patch.object(messagebox, "showerror", autospec=True) as m_showerror:
            self.app._poll_result()
        self.assertEqual(self.app.run_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.status_var.get(), "Error")
        m_showerror.assert_called_once()
        self.assertEqual(m_showerror.call_args[0], ("Generation failed", "Load error"))

    def test_poll_result_cache_success_enables_cache_buttons_shows_info(self):
        self.app.refresh_cache_btn.config(state=tk.DISABLED)
        self.app.clear_cache_btn.config(state=tk.DISABLED)
        self.app.result_queue.put(("cache", True, "Cache cleared."))
        with mock.patch.object(messagebox, "showinfo", autospec=True) as m_showinfo:
            self.app._poll_result()
        self.assertEqual(self.app.refresh_cache_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.clear_cache_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.status_var.get(), "Cache cleared.")
        m_showinfo.assert_called_once()
        self.assertEqual(m_showinfo.call_args[0], ("Model cache", "Cache cleared."))

    def test_poll_result_cache_failure_enables_cache_buttons_shows_showerror(self):
        self.app.refresh_cache_btn.config(state=tk.DISABLED)
        self.app.clear_cache_btn.config(state=tk.DISABLED)
        self.app.result_queue.put(("cache", False, "Network error"))
        with mock.patch.object(messagebox, "showerror", autospec=True) as m_showerror:
            self.app._poll_result()
        self.assertEqual(self.app.refresh_cache_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.clear_cache_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.status_var.get(), "Cache operation failed")
        m_showerror.assert_called_once()
        self.assertEqual(m_showerror.call_args[0], ("Model cache", "Network error"))


class AddRowTest(unittest.TestCase):
    """Tests for _add_row layout helper."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_add_row_creates_label_and_entry_without_browse(self):
        var = tk.StringVar()
        initial_row = self.app._row
        ent = self.app._add_row("Test label:", var)
        self.assertIsInstance(ent, tk.Entry)
        self.assertEqual(ent.get(), "")
        self.assertEqual(self.app._row, initial_row + 2)
        children = [w.winfo_class() for w in self.app._main_frame.winfo_children()]
        self.assertIn("Label", children)
        self.assertIn("Frame", children)

    def test_add_row_with_browse_creates_button(self):
        var = tk.StringVar()
        with mock.patch.object(
            filedialog, "askopenfilename", return_value="", autospec=True
        ):
            ent = self.app._add_row("File:", var, self.app.browse_audio)
        self.assertEqual(ent["state"], "readonly")
        frames = [
            w
            for w in self.app._main_frame.winfo_children()
            if w.winfo_class() == "Frame"
        ]
        buttons = []
        for f in frames:
            for c in f.winfo_children():
                if c.winfo_class() == "Button":
                    buttons.append(c)
        self.assertGreater(len(buttons), 0)

    def test_app_has_refresh_and_clear_cache_buttons(self):
        """App has Refresh cache and Clear cache buttons for model cache."""
        self.assertIsNotNone(getattr(self.app, "refresh_cache_btn", None))
        self.assertIsNotNone(getattr(self.app, "clear_cache_btn", None))
        self.assertEqual(self.app.refresh_cache_btn["text"], "Refresh cache")
        self.assertEqual(self.app.clear_cache_btn["text"], "Clear cache")


class CacheButtonsTest(unittest.TestCase):
    """Tests for refresh_cache_clicked and clear_cache_clicked."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_refresh_cache_clicked_disables_buttons_schedules_poll(self):
        after_cbs = []

        def capture_after(ms, cb):
            after_cbs.append((ms, cb))

        with (
            mock.patch.object(
                generate_ui.pretrained, "refresh_model_cache", autospec=True
            ) as m_refresh,
            mock.patch.object(
                self.root, "after", side_effect=capture_after, autospec=True
            ),
        ):
            self.app.refresh_cache_clicked()
        self.assertEqual(self.app.refresh_cache_btn["state"], tk.DISABLED)
        self.assertEqual(self.app.clear_cache_btn["state"], tk.DISABLED)
        self.assertEqual(self.app.status_var.get(), "Refreshing model cache…")
        self.assertEqual(len(after_cbs), 1)
        self.assertEqual(after_cbs[0][0], 100)
        self.assertEqual(after_cbs[0][1].__name__, "_poll_result")
        m_refresh.assert_called_once()

    def test_clear_cache_clicked_disables_buttons_schedules_poll(self):
        after_cbs = []

        def capture_after(ms, cb):
            after_cbs.append((ms, cb))

        with (
            mock.patch.object(
                generate_ui.pretrained, "clear_model_cache", autospec=True
            ) as m_clear,
            mock.patch.object(
                self.root, "after", side_effect=capture_after, autospec=True
            ),
        ):
            self.app.clear_cache_clicked()
        self.assertEqual(self.app.refresh_cache_btn["state"], tk.DISABLED)
        self.assertEqual(self.app.clear_cache_btn["state"], tk.DISABLED)
        self.assertEqual(self.app.status_var.get(), "Clearing cache…")
        self.assertEqual(len(after_cbs), 1)
        m_clear.assert_called_once()

    def test_refresh_cache_worker_puts_success_on_queue(self):
        with (
            mock.patch.object(
                generate_ui.pretrained, "refresh_model_cache", autospec=True
            ),
            mock.patch.object(messagebox, "showinfo", autospec=True) as m_showinfo,
        ):
            self.app.refresh_cache_clicked()
            self.app.result_queue.put(("cache", True, "Models refreshed."))
            self.app._poll_result()
        m_showinfo.assert_called_once()
        self.assertIn("Models refreshed", m_showinfo.call_args[0][1])

    def test_clear_cache_worker_puts_success_on_queue(self):
        with (
            mock.patch.object(
                generate_ui.pretrained, "clear_model_cache", autospec=True
            ),
            mock.patch.object(messagebox, "showinfo", autospec=True) as m_showinfo,
        ):
            self.app.clear_cache_clicked()
            self.app.result_queue.put(("cache", True, "Cache cleared."))
            self.app._poll_result()
        m_showinfo.assert_called_once()
        self.assertIn("Cache cleared", m_showinfo.call_args[0][1])


class CanvasHandlersTest(unittest.TestCase):
    """Tests for _on_canvas_configure and _on_mousewheel."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_on_canvas_configure_updates_window_width(self):
        self.root.update_idletasks()
        with mock.patch.object(
            self.app.canvas, "itemconfig", autospec=True
        ) as m_itemconfig:
            self.app._on_canvas_configure(mock.MagicMock(width=300))
            m_itemconfig.assert_called()
            call_kw = m_itemconfig.call_args[1]
            self.assertEqual(call_kw.get("width"), 300)

    def test_on_mousewheel_scrolls(self):
        with mock.patch.object(
            self.app.canvas, "yview_scroll", autospec=True
        ) as m_scroll:
            self.app._on_mousewheel(mock.MagicMock(delta=120))
        m_scroll.assert_called_once_with(-1, "units")


class FrozenTkinterCompatTest(unittest.TestCase):
    """Test that the PyInstaller spec ensures tkinter submodules are bundled for frozen exe."""

    def test_spec_includes_tkinter_filedialog_messagebox_hiddenimports(self):
        """generate_ui.spec must list tkinter.filedialog and tkinter.messagebox in hiddenimports.

        PyInstaller often omits these tkinter submodules unless listed; without them
        the frozen exe raises AttributeError when opening file dialogs or message boxes.
        """
        spec_path = os.path.join(_SCRIPT_DIR, "generate_ui.spec")
        with open(spec_path, encoding="utf-8") as f:
            spec_source = f.read()
        self.assertIn(
            "tkinter.filedialog",
            spec_source,
            "generate_ui.spec hiddenimports must include tkinter.filedialog for frozen exe",
        )
        self.assertIn(
            "tkinter.messagebox",
            spec_source,
            "generate_ui.spec hiddenimports must include tkinter.messagebox for frozen exe",
        )


class MainEntryTest(unittest.TestCase):
    """Tests for main() entry point."""

    def test_main_creates_app_and_runs_mainloop(self):
        try:
            with mock.patch.object(tk.Tk, "mainloop", autospec=True):
                generate_ui.main()
            if tk._default_root is not None:
                tk._default_root.destroy()
                tk._default_root = None
        except tk.TclError as e:
            if "no display" in str(e).lower() or "DISPLAY" in str(e):
                raise unittest.SkipTest(f"Tk requires a display: {e}") from e
            raise


class EnsureStderrStdoutForFrozenTest(unittest.TestCase):
    """Tests for _ensure_stdout_stderr_for_frozen (avoids NoneType.write in frozen exe)."""

    def test_replaces_none_stdout_and_stderr_with_writeable_streams(self):
        """When stdout/stderr are None, they are replaced so .write() does not crash."""
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            sys.stdout = None
            sys.stderr = None
            generate_ui._ensure_stdout_stderr_for_frozen()
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            assert sys.stdout is not None
            assert sys.stderr is not None
            sys.stdout.write("out")
            sys.stderr.write("err")
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            generate_ui._close_stdout_stderr_fallbacks()

    def test_leaves_non_none_stdout_and_stderr_unchanged(self):
        """When stdout/stderr are already set, they are not replaced."""
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        try:
            sys.stdout = out_buf
            sys.stderr = err_buf
            generate_ui._ensure_stdout_stderr_for_frozen()
            self.assertIs(sys.stdout, out_buf)
            self.assertIs(sys.stderr, err_buf)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def test_replaces_only_stdout_when_stderr_is_set(self):
        """When only stdout is None, only stdout is replaced."""
        err_buf = io.StringIO()
        try:
            sys.stdout = None
            sys.stderr = err_buf
            generate_ui._ensure_stdout_stderr_for_frozen()
            self.assertIsNotNone(sys.stdout)
            self.assertIs(sys.stderr, err_buf)
            assert sys.stdout is not None
            sys.stdout.write("x")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            generate_ui._close_stdout_stderr_fallbacks()

    def test_replaces_only_stderr_when_stdout_is_set(self):
        """When only stderr is None, only stderr is replaced."""
        out_buf = io.StringIO()
        try:
            sys.stdout = out_buf
            sys.stderr = None
            generate_ui._ensure_stdout_stderr_for_frozen()
            self.assertIs(sys.stdout, out_buf)
            self.assertIsNotNone(sys.stderr)
            assert sys.stderr is not None
            sys.stderr.write("x")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            generate_ui._close_stdout_stderr_fallbacks()

    def test_registers_atexit_when_fallbacks_opened(self):
        """When stdout/stderr are replaced, atexit is registered to close the handles."""
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            sys.stdout = None
            sys.stderr = None
            with mock.patch.object(
                generate_ui.atexit, "register", autospec=True
            ) as m_register:
                generate_ui._ensure_stdout_stderr_for_frozen()
            m_register.assert_called_once_with(
                generate_ui._close_stdout_stderr_fallbacks
            )
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            generate_ui._close_stdout_stderr_fallbacks()

    def test_close_stdout_stderr_fallbacks_closes_files_and_clears_list(self):
        """_close_stdout_stderr_fallbacks closes opened devnull handles and clears the list."""
        saved_stdout = sys.stdout
        saved_stderr = sys.stderr
        try:
            sys.stdout = None
            sys.stderr = None
            generate_ui._ensure_stdout_stderr_for_frozen()
            fallbacks = list(generate_ui._stdout_stderr_fallbacks)
            self.assertEqual(len(fallbacks), 2)
            generate_ui._close_stdout_stderr_fallbacks()
            self.assertEqual(generate_ui._stdout_stderr_fallbacks, [])
            for f in fallbacks:
                self.assertTrue(f.closed)
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr


class SingleInstanceTest(unittest.TestCase):
    """Tests for _try_single_instance_win32 (frozen Windows exe single-instance)."""

    def test_returns_true_when_not_frozen(self):
        """When not frozen, _try_single_instance_win32 returns True without using mutex."""
        with mock.patch.object(sys, "frozen", False, create=True):
            result = generate_ui._try_single_instance_win32()
        self.assertTrue(result)

    def test_returns_true_when_frozen_win32_first_instance(self):
        """When frozen on win32 and mutex is new (first instance), returns True and does not exit."""
        mock_kernel = mock.MagicMock()
        mock_kernel.CreateMutexW.return_value = 12345
        mock_kernel.GetLastError.return_value = 0
        with (
            mock.patch.object(sys, "frozen", True, create=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(generate_ui.ctypes, "windll", create=True) as m_windll,
        ):
            m_windll.kernel32 = mock_kernel
            result = generate_ui._try_single_instance_win32()
        self.assertTrue(result)
        mock_kernel.CreateMutexW.assert_called_once()
        mock_kernel.GetLastError.assert_called_once()

    def test_exits_when_frozen_win32_second_instance(self):
        """When frozen on win32 and mutex already exists, shows message and sys.exit(0)."""
        mock_kernel = mock.MagicMock()
        mock_kernel.CreateMutexW.return_value = 12345
        mock_kernel.GetLastError.return_value = 183  # ERROR_ALREADY_EXISTS
        mock_user = mock.MagicMock()
        with (
            mock.patch.object(sys, "frozen", True, create=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(generate_ui.ctypes, "windll", create=True) as m_windll,
        ):
            m_windll.kernel32 = mock_kernel
            m_windll.user32 = mock_user
            with self.assertRaises(SystemExit) as cm:
                generate_ui._try_single_instance_win32()
        self.assertEqual(cm.exception.code, 0)
        mock_kernel.CloseHandle.assert_called_once_with(12345)
        mock_user.MessageBoxW.assert_called_once()
        call_args = mock_user.MessageBoxW.call_args[0]
        self.assertIn("already running", call_args[1].lower())
        self.assertEqual(call_args[2], "Already running")

    def test_returns_true_when_create_mutex_returns_none(self):
        """When frozen on win32 and CreateMutexW fails (returns None), returns True and does not exit."""
        mock_kernel = mock.MagicMock()
        mock_kernel.CreateMutexW.return_value = None
        mock_user = mock.MagicMock()
        with (
            mock.patch.object(sys, "frozen", True, create=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(generate_ui.ctypes, "windll", create=True) as m_windll,
        ):
            m_windll.kernel32 = mock_kernel
            m_windll.user32 = mock_user
            result = generate_ui._try_single_instance_win32()
        self.assertTrue(result)
        mock_kernel.CreateMutexW.assert_called_once()
        mock_kernel.GetLastError.assert_not_called()
        mock_kernel.CloseHandle.assert_not_called()
        mock_user.MessageBoxW.assert_not_called()


class MainBlockTest(unittest.TestCase):
    """Tests for the if __name__ == '__main__' block (frozen exe and main entry)."""

    def test_main_block_calls_freeze_support_and_single_instance_when_frozen_on_win32(
        self,
    ):
        """When run as frozen exe on Windows, freeze_support(), _ensure_stdout_stderr_for_frozen(), and _try_single_instance_win32() are called before main()."""
        with (
            mock.patch.object(sys, "frozen", True, create=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(
                generate_ui.multiprocessing, "freeze_support", autospec=True
            ) as m_freeze,
            mock.patch.object(
                generate_ui, "_ensure_stdout_stderr_for_frozen", autospec=True
            ) as m_ensure_io,
            mock.patch.object(
                generate_ui, "_try_single_instance_win32", autospec=True
            ) as m_single,
            mock.patch.object(generate_ui, "main", autospec=True) as m_main,
        ):
            if getattr(sys, "frozen", False) and sys.platform == "win32":
                generate_ui.multiprocessing.freeze_support()
                generate_ui._ensure_stdout_stderr_for_frozen()
                generate_ui._try_single_instance_win32()
            generate_ui.main()
        m_freeze.assert_called_once()
        m_ensure_io.assert_called_once()
        m_single.assert_called_once()
        m_main.assert_called_once()

    def test_main_block_does_not_call_freeze_support_when_not_frozen(self):
        """When not frozen, freeze_support() is not called."""
        with mock.patch.object(
            generate_ui.multiprocessing, "freeze_support", autospec=True
        ) as m_freeze:
            if getattr(sys, "frozen", False) and sys.platform == "win32":
                generate_ui.multiprocessing.freeze_support()
        m_freeze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
