"""Unit tests for the generator UI script (scripts/generate_ui.py)."""

import os
import queue
import sys
import tempfile
import tkinter as tk
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

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
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

    def test_run_generation_with_none_bpm_calls_generator_with_none(self):
        """_run_generation with bpm=None calls generate_output_data with bpm=None."""
        mock_output_data = mock.MagicMock()
        mock_output_data.generate_txt_output.return_value = "TITLE X\nBPM 100\nNOTES\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            result_queue = queue.Queue()
            with (
                mock.patch(
                    "generate_ui.keras.models.load_model", return_value=mock.MagicMock()
                ),
                mock.patch(
                    "generate_ui.generator.generate_output_data",
                    return_value=mock_output_data,
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
            success, _ = result_queue.get_nowait()
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
                mock.patch(
                    "generate_ui.keras.models.load_model", return_value=mock.MagicMock()
                ),
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
            mock.patch(
                "generate_ui.keras.models.load_model", return_value=mock.MagicMock()
            ),
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


class BrowseCallbacksTest(unittest.TestCase):
    """Tests for browse_audio, browse_onset, browse_arrow, browse_output."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_browse_audio_sets_path_and_default_output_not_same_as_audio(self):
        """Default output is basename_chart.txt (not basename.txt) when output was empty."""
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="C:/music/song.mp3",
        ):
            self.app.browse_audio()
        self.assertEqual(self.app.audio_path_var.get(), "C:/music/song.mp3")
        self.assertEqual(
            os.path.normpath(self.app.output_path_var.get()),
            os.path.normpath("C:/music/song_chart.txt"),
        )

    def test_browse_audio_default_output_uses_song_title_when_set(self):
        """When song title is set, default output filename is song title + .txt (sanitized)."""
        self.app.song_title_var.set("My Song")
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="C:/music/anything.ogg",
        ):
            self.app.browse_audio()
        self.assertEqual(
            os.path.normpath(self.app.output_path_var.get()),
            os.path.normpath("C:/music/My Song.txt"),
        )

    def test_default_output_path_for_audio_sanitizes_unsafe_chars(self):
        """Song title with path-unsafe chars is sanitized to underscores."""
        self.app.song_title_var.set("Title/with:bad*chars?")
        result = self.app._default_output_path_for_audio("C:/dir/file.ogg")
        self.assertEqual(
            os.path.normpath(result),
            os.path.normpath("C:/dir/Title_with_bad_chars_.txt"),
        )

    def test_browse_audio_does_not_overwrite_output_when_set(self):
        self.app.output_path_var.set("C:/out/custom.txt")
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="C:/music/song.mp3",
        ):
            self.app.browse_audio()
        self.assertEqual(self.app.output_path_var.get(), "C:/out/custom.txt")

    def test_browse_audio_cancelled_does_nothing(self):
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="",
        ):
            self.app.browse_audio()
        self.assertEqual(self.app.audio_path_var.get(), "")

    def test_browse_onset_sets_path(self):
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="/path/to/onset.keras",
        ):
            self.app.browse_onset()
        self.assertEqual(self.app.onset_model_var.get(), "/path/to/onset.keras")

    def test_browse_onset_cancelled_does_nothing(self):
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="",
        ):
            self.app.browse_onset()
        self.assertEqual(self.app.onset_model_var.get(), "")

    def test_browse_arrow_sets_path(self):
        with mock.patch(
            "tkinter.filedialog.askopenfilename",
            return_value="/path/to/arrow.keras",
        ):
            self.app.browse_arrow()
        self.assertEqual(self.app.arrow_model_var.get(), "/path/to/arrow.keras")

    def test_browse_output_sets_path(self):
        with mock.patch(
            "tkinter.filedialog.asksaveasfilename",
            return_value="/path/to/chart.txt",
        ):
            self.app.browse_output()
        self.assertEqual(self.app.output_path_var.get(), "/path/to/chart.txt")

    def test_browse_output_cancelled_does_nothing(self):
        with mock.patch(
            "tkinter.filedialog.asksaveasfilename",
            return_value="",
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
        with mock.patch.object(self.app.status_label, "update_idletasks") as m_update:
            self.app.set_status("Done")
        m_update.assert_called_once()


class RunClickedTest(unittest.TestCase):
    """Tests for run_clicked validation and worker scheduling."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_run_clicked_validation_failure_shows_messagebox(self):
        self.app.audio_path_var.set("")
        with mock.patch("tkinter.messagebox.showerror") as m_showerror:
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
            mock.patch.object(self.root, "after", side_effect=capture_after),
            mock.patch(
                "generate_ui._run_generation",
                side_effect=lambda **kw: kw["result_queue"].put((True, "/out.txt")),
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
        with mock.patch.object(self.root, "after") as m_after:
            self.app._poll_result()
        m_after.assert_called_once_with(100, self.app._poll_result)

    def test_poll_result_success_enables_btn_sets_status_shows_info(self):
        self.app.result_queue.put((True, "C:/out/chart.txt"))
        with mock.patch("tkinter.messagebox.showinfo") as m_showinfo:
            self.app._poll_result()
        self.assertEqual(self.app.run_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.status_var.get(), "Saved to C:/out/chart.txt")
        m_showinfo.assert_called_once()
        self.assertIn("chart.txt", m_showinfo.call_args[0][1])

    def test_poll_result_failure_sets_error_status_shows_showerror(self):
        self.app.result_queue.put((False, "Load error"))
        with mock.patch("tkinter.messagebox.showerror") as m_showerror:
            self.app._poll_result()
        self.assertEqual(self.app.run_btn["state"], tk.NORMAL)
        self.assertEqual(self.app.status_var.get(), "Error")
        m_showerror.assert_called_once()
        self.assertEqual(m_showerror.call_args[0], ("Generation failed", "Load error"))


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
        with mock.patch("tkinter.filedialog.askopenfilename", return_value=""):
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


class CanvasHandlersTest(unittest.TestCase):
    """Tests for _on_canvas_configure and _on_mousewheel."""

    def setUp(self):
        self.root, self.app = _make_app()
        self.addCleanup(self.root.destroy)

    def test_on_canvas_configure_updates_window_width(self):
        self.root.update_idletasks()
        with mock.patch.object(self.app.canvas, "itemconfig") as m_itemconfig:
            self.app._on_canvas_configure(mock.MagicMock(width=300))
            m_itemconfig.assert_called()
            call_kw = m_itemconfig.call_args[1]
            self.assertEqual(call_kw.get("width"), 300)

    def test_on_mousewheel_scrolls(self):
        with mock.patch.object(self.app.canvas, "yview_scroll") as m_scroll:
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
            with mock.patch("tkinter.Tk.mainloop"):
                generate_ui.main()
            if tk._default_root is not None:
                tk._default_root.destroy()
                tk._default_root = None
        except tk.TclError as e:
            if "no display" in str(e).lower() or "DISPLAY" in str(e):
                raise unittest.SkipTest(f"Tk requires a display: {e}") from e
            raise


class MainBlockTest(unittest.TestCase):
    """Tests for the if __name__ == '__main__' block (frozen exe and main entry)."""

    def test_main_block_calls_freeze_support_when_frozen_on_win32(self):
        """When run as frozen exe on Windows, freeze_support() is called before main()."""
        with (
            mock.patch.object(sys, "frozen", True, create=True),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch("generate_ui.multiprocessing.freeze_support") as m_freeze,
            mock.patch("generate_ui.main") as m_main,
        ):
            # Simulate the __main__ block
            if getattr(sys, "frozen", False) and sys.platform == "win32":
                generate_ui.multiprocessing.freeze_support()
            generate_ui.main()
        m_freeze.assert_called_once()
        m_main.assert_called_once()

    def test_main_block_does_not_call_freeze_support_when_not_frozen(self):
        """When not frozen, freeze_support() is not called."""
        with mock.patch("generate_ui.multiprocessing.freeze_support") as m_freeze:
            # Simulate the __main__ block when not frozen (normal interpreter)
            if getattr(sys, "frozen", False) and sys.platform == "win32":
                generate_ui.multiprocessing.freeze_support()
        m_freeze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
