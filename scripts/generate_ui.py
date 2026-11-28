r"""Simple UI for generating a StepMania chart from an audio file.

Launch with:
    python scripts/generate_ui.py

Onset and arrow models (.pt) are optional; leave blank to download from Google Drive and cache.
"""

import atexit
import contextlib
import ctypes
import multiprocessing
import os
import pathlib
import queue
import sys
import threading
import tkinter as tk
import typing
from tkinter import filedialog, messagebox

from stepcovnet import (
    beat_tracking,
    generator,
    pretrained,
    trainers,
)

AUDIO_TYPES = [
    ("Audio files", "*.mp3 *.wav *.ogg *.flac"),
    ("All files", "*.*"),
]
MODEL_TYPES = [
    ("PyTorch models", "*.pt"),
    ("All files", "*.*"),
]
TXT_TYPES = [
    ("Text files", "*.txt"),
    ("StepMania simfiles", "*.sm"),
    ("All files", "*.*"),
]


def _validate_inputs(
    audio_path: str,
    song_title: str,
    bpm_str: str,
    onset_path: str,
    arrow_path: str,
    output_path: str,
) -> tuple[bool, str | tuple[str, str, int | None, str | None, str | None, str]]:
    """Validate UI inputs. Returns (False, error_message) or (True, (audio_path, song_title, bpm, onset_path, arrow_path, output_path)). bpm/onset_path/arrow_path are None when left blank."""
    audio_path = audio_path.strip()
    song_title = song_title.strip()
    bpm_str = bpm_str.strip()
    onset_path = onset_path.strip()
    arrow_path = arrow_path.strip()
    output_path = output_path.strip()

    if not audio_path:
        return (False, "Please select an audio file.")
    if not song_title:
        return (False, "Please enter a song title.")
    if not bpm_str:
        bpm_val = None
    else:
        try:
            bpm_val = int(bpm_str)
        except ValueError:
            return (False, "BPM must be an integer.")
        if bpm_val < 1 or bpm_val > 9999:
            return (False, "BPM must be between 1 and 9999.")
    if not output_path:
        return (False, "Please choose an output file.")
    return (
        True,
        (
            audio_path,
            song_title,
            bpm_val,
            onset_path if onset_path else None,
            arrow_path if arrow_path else None,
            output_path,
        ),
    )


def _validate_advanced_options(
    beat_tracker: str,
    target_density_str: str,
    threshold_str: str,
) -> tuple[bool, str | tuple[str, float, float]]:
    """Validate beat tracker, ITGPT target density, and onset threshold inputs.

    Args:
        beat_tracker: Beat tracker backend name.
        target_density_str: Target average steps per beat string (blank -> 2.0).
        threshold_str: Onset probability threshold string (blank -> 0.5).

    Returns:
        ``(False, error_message)`` or ``(True, (beat_tracker, target_density, threshold))``.
    """
    tracker = beat_tracker.strip() or beat_tracking.TRACKER_AUTO
    if tracker not in beat_tracking.VALID_TRACKERS:
        return (
            False,
            f"Beat tracker must be one of {', '.join(beat_tracking.VALID_TRACKERS)}.",
        )

    density_s = target_density_str.strip()
    if not density_s:
        target_density = 2.0
    else:
        try:
            target_density = float(density_s)
        except ValueError:
            return (False, "Target density must be a positive number.")
        if not (target_density > 0.0 and target_density <= 32.0):
            return (False, "Target density must be greater than 0 and at most 32.")

    thr_s = threshold_str.strip()
    if not thr_s:
        threshold = 0.5
    else:
        try:
            threshold = float(thr_s)
        except ValueError:
            return (False, "Onset threshold must be a number between 0 and 1.")
        if not (0.0 < threshold < 1.0):
            return (False, "Onset threshold must be strictly between 0 and 1.")

    return (True, (tracker, target_density, threshold))


def _run_generation(
    audio_path: str,
    song_title: str,
    bpm: int | None,
    onset_model_path: str | None,
    arrow_model_path: str | None,
    output_path: str,
    use_post_processing: bool,
    result_queue: queue.Queue,
    *,
    beat_tracker: str = beat_tracking.TRACKER_AUTO,
    snap_beats: bool = False,
    target_density: float = 2.0,
    threshold: float = 0.5,
    all_difficulties: bool = False,
) -> None:
    """Load models, run generator, write output. Puts (True, output_path) or (False, error_msg) into result_queue."""
    try:
        onset_path = pretrained.resolve_onset_model_path(onset_model_path)
        arrow_path = pretrained.resolve_arrow_model_path(arrow_model_path)
        if beat_tracking.is_itgpt_checkpoint(onset_path):
            onset_model = beat_tracking.load_onset_model_auto(onset_path)
        else:
            onset_model = trainers.load_onset_model(onset_path)
        arrow_model = trainers.load_arrow_model(arrow_path)
        output_data = generator.generate_output_data(
            audio_path=audio_path,
            song_title=song_title,
            bpm=bpm,
            onset_model=onset_model,
            arrow_model=arrow_model,
            use_post_processing=use_post_processing,
            beat_tracker=beat_tracker,
            snap_beats=snap_beats,
            target_density=target_density,
            threshold=threshold,
            all_difficulties=all_difficulties,
        )
        out_path = pathlib.Path(output_path)
        if out_path.suffix.lower() == ".sm":
            content = output_data.generate_sm_output(
                music_filename=pathlib.Path(audio_path).name
            )
        else:
            content = output_data.generate_txt_output()
        with out_path.open("w") as f:
            f.write(content)
        result_queue.put(("generation", True, output_path))
    except Exception as e:  # noqa: BLE001
        result_queue.put(("generation", False, str(e)))


class _GeneratorApp:
    """Application UI and callbacks for the generator window. Built for testability.

    Attributes:
        root: The root Tk window.
        result_queue: Queue for passing generation results back (e.g. to tests).
        audio_path_var: StringVar for the audio file path.
        song_title_var: StringVar for the song title.
        bpm_var: StringVar for BPM (optional).
        onset_model_var: StringVar for onset model path.
        arrow_model_var: StringVar for arrow model path.
        output_path_var: StringVar for output .txt or .sm path.
        use_post_processing_var: BooleanVar for post-processing toggle.
        beat_tracker_var: StringVar for beat tracker backend (auto/beat_this/librosa).
        snap_beats_var: BooleanVar for onset-transient beat snapping toggle.
        all_difficulties_var: BooleanVar for generating a 5-difficulty pack.
        target_density_var: StringVar for ITGPT target density (steps/beat).
        threshold_var: StringVar for onset probability threshold.
        status_var: StringVar for status text.
        canvas: Canvas used for the scrollable form area.
        run_btn: Generate chart button.
        status_label: Label showing status_var.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("StepCOVNet Generator")
        root.minsize(400, 320)
        root.geometry("480x660")
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._closing = False
        self._poll_after_id = None
        self.result_queue = queue.Queue()

        self.audio_path_var = tk.StringVar()
        self.song_title_var = tk.StringVar()
        self.bpm_var = tk.StringVar()
        self.onset_model_var = tk.StringVar()
        self.arrow_model_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.use_post_processing_var = tk.BooleanVar(value=False)
        self.beat_tracker_var = tk.StringVar(value=beat_tracking.TRACKER_AUTO)
        self.snap_beats_var = tk.BooleanVar(value=False)
        self.all_difficulties_var = tk.BooleanVar(value=False)
        self.target_density_var = tk.StringVar(value="2.0")
        self.threshold_var = tk.StringVar(value="0.5")
        self.status_var = tk.StringVar(value="")

        outer = tk.Frame(root, padx=12, pady=12)
        outer.pack(fill=tk.BOTH, expand=True)

        bottom_frame = tk.Frame(outer)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))

        scroll_area = tk.Frame(outer)
        scroll_area.pack(fill=tk.BOTH, expand=True)
        scroll_area.rowconfigure(0, weight=1)
        scroll_area.columnconfigure(0, weight=1, minsize=80)
        scroll_area.columnconfigure(1, weight=0, minsize=16)

        self.canvas = tk.Canvas(scroll_area, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            scroll_area, orient=tk.VERTICAL, command=self.canvas.yview, width=16
        )
        scrollable_frame = tk.Frame(self.canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self._canvas_window = self.canvas.create_window(
            (0, 0), window=scrollable_frame, anchor=tk.NW
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 12))
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        main_frame = scrollable_frame
        self._main_frame = main_frame
        self._bottom_frame = bottom_frame
        self._row = 0

        self.run_btn = tk.Button(
            bottom_frame, text="Generate chart", command=self.run_clicked
        )
        self.status_label = tk.Label(
            bottom_frame, textvariable=self.status_var, anchor=tk.W, fg="gray"
        )

        self._add_row("Audio file:", self.audio_path_var, self.browse_audio)
        self._add_row("Song title:", self.song_title_var)
        self._add_row("BPM (optional; leave blank to detect from audio):", self.bpm_var)
        self._add_row(
            "Onset model (.pt, optional):", self.onset_model_var, self.browse_onset
        )
        self._add_row(
            "Arrow model (.pt, optional):", self.arrow_model_var, self.browse_arrow
        )
        cache_row = tk.Frame(main_frame)
        cache_row.grid(row=self._row, column=0, sticky=tk.W, pady=(6, 8))
        self._row += 1
        tk.Label(cache_row, text="Model cache (for default models):", anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W
        )
        cache_btns_frame = tk.Frame(cache_row)
        cache_btns_frame.grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        self.refresh_cache_btn = tk.Button(
            cache_btns_frame,
            text="Refresh cache",
            command=self.refresh_cache_clicked,
        )
        self.refresh_cache_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.clear_cache_btn = tk.Button(
            cache_btns_frame,
            text="Clear cache",
            command=self.clear_cache_clicked,
        )
        self.clear_cache_btn.pack(side=tk.LEFT)
        self._row += 1
        self._add_row(
            "Output file (.txt or .sm):", self.output_path_var, self.browse_output
        )

        tracker_row = tk.Frame(main_frame)
        tracker_row.grid(row=self._row, column=0, sticky=tk.W, pady=(4, 6))
        self._row += 1
        tk.Label(tracker_row, text="Beat tracker:", anchor=tk.W).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        tk.OptionMenu(
            tracker_row,
            self.beat_tracker_var,
            *beat_tracking.VALID_TRACKERS,
        ).pack(side=tk.LEFT)

        self._add_row(
            "Target density (steps/beat, ITGPT models; default 2.0):",
            self.target_density_var,
        )
        self._add_row(
            "Onset threshold (0–1; default 0.5):",
            self.threshold_var,
        )

        cb_frame = tk.Frame(main_frame)
        cb_frame.grid(row=self._row, column=0, sticky=tk.W, pady=(4, 12))
        self._row += 1
        tk.Checkbutton(
            cb_frame,
            text="Generate full 5-difficulty pack (Beginner–Challenge)",
            variable=self.all_difficulties_var,
        ).pack(anchor=tk.W)
        tk.Checkbutton(
            cb_frame,
            text="Use post-processing (peak-picking for legacy onset timings)",
            variable=self.use_post_processing_var,
        ).pack(anchor=tk.W)
        tk.Checkbutton(
            cb_frame,
            text="Snap beats to audio transients (constant phase alignment)",
            variable=self.snap_beats_var,
        ).pack(anchor=tk.W)

        self.run_btn.pack(side=tk.LEFT)
        self.status_label.pack(side=tk.LEFT, padx=(16, 0), fill=tk.X, expand=True)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfig(self._canvas_window, width=event.width)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_close(self) -> None:
        """Handle WM_DELETE_WINDOW: stop polling, quit mainloop, destroy root so process exits."""
        if self._closing:
            return
        self._closing = True
        if self._poll_after_id is not None:
            with contextlib.suppress(tk.TclError):
                self.root.after_cancel(self._poll_after_id)
            self._poll_after_id = None
        self.root.quit()
        self.root.destroy()

    def _default_output_path_for_audio(self, audio_path: str) -> str:
        """Suggested output path when user selects an audio file: stepcovnet_chart_ + song title or basename."""
        audio = pathlib.Path(audio_path)
        prefix = "stepcovnet_chart_"
        title = self.song_title_var.get().strip()
        if title:
            # Sanitize for filesystem: replace path separators and other unsafe chars
            safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in title).strip()
            name = prefix + (safe if safe else audio.stem)
        else:
            name = prefix + audio.stem
        return str(audio.parent / f"{name}.txt")

    def browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=AUDIO_TYPES, title="Select audio file"
        )
        if path:
            self.audio_path_var.set(path)
            if not self.output_path_var.get().strip():
                self.output_path_var.set(self._default_output_path_for_audio(path))

    def browse_onset(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=MODEL_TYPES, title="Select onset model"
        )
        if path:
            self.onset_model_var.set(path)

    def browse_arrow(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=MODEL_TYPES, title="Select arrow model"
        )
        if path:
            self.arrow_model_var.set(path)

    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=TXT_TYPES,
            title="Save stepcovnet chart as",
        )
        if path:
            self.output_path_var.set(path)

    def set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.status_label.update_idletasks()

    def run_clicked(self) -> None:
        ok, result = _validate_inputs(
            self.audio_path_var.get(),
            self.song_title_var.get(),
            self.bpm_var.get(),
            self.onset_model_var.get(),
            self.arrow_model_var.get(),
            self.output_path_var.get(),
        )
        if not ok:
            assert isinstance(result, str)
            messagebox.showerror("Validation", result)
            return
        assert isinstance(result, tuple)
        audio_path, song_title, bpm_val, onset_path, arrow_path, output_path = result

        adv_ok, adv_result = _validate_advanced_options(
            self.beat_tracker_var.get(),
            self.target_density_var.get(),
            self.threshold_var.get(),
        )
        if not adv_ok:
            assert isinstance(adv_result, str)
            messagebox.showerror("Validation", adv_result)
            return
        assert isinstance(adv_result, tuple)
        beat_tracker, target_density, threshold = adv_result
        snap_beats = bool(self.snap_beats_var.get())
        all_difficulties = bool(self.all_difficulties_var.get())
        use_post_processing = bool(self.use_post_processing_var.get())

        self.run_btn.config(state=tk.DISABLED)
        self.set_status("Processing…")

        def worker() -> None:
            _run_generation(
                audio_path=audio_path,
                song_title=song_title,
                bpm=bpm_val,
                onset_model_path=onset_path,
                arrow_model_path=arrow_path,
                output_path=output_path,
                use_post_processing=use_post_processing,
                result_queue=self.result_queue,
                beat_tracker=beat_tracker,
                snap_beats=snap_beats,
                target_density=target_density,
                threshold=threshold,
                all_difficulties=all_difficulties,
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._poll_after_id = self.root.after(100, self._poll_result)

    def _poll_result(self) -> None:
        self._poll_after_id = None
        if self._closing:
            return
        try:
            item = self.result_queue.get_nowait()
        except queue.Empty:
            if not self._closing:
                self._poll_after_id = self.root.after(100, self._poll_result)
            return
        source, success, value = item
        if source == "generation":
            self.run_btn.config(state=tk.NORMAL)
            if success:
                self.set_status(f"Saved to {value}")
                messagebox.showinfo("Success", f"Chart saved to:\n{value}")
            else:
                self.set_status("Error")
                messagebox.showerror("Generation failed", value)
        else:
            assert source == "cache"
            self.refresh_cache_btn.config(state=tk.NORMAL)
            self.clear_cache_btn.config(state=tk.NORMAL)
            if success:
                self.set_status(value)
                messagebox.showinfo("Model cache", value)
            else:
                self.set_status("Cache operation failed")
                messagebox.showerror("Model cache", value)

    def refresh_cache_clicked(self) -> None:
        """Re-download default models into the cache (runs in background thread)."""
        self.refresh_cache_btn.config(state=tk.DISABLED)
        self.clear_cache_btn.config(state=tk.DISABLED)
        self.set_status("Refreshing model cache…")

        def worker() -> None:
            try:
                pretrained.refresh_model_cache()
                self.result_queue.put(("cache", True, "Models refreshed."))
            except Exception as e:  # noqa: BLE001
                self.result_queue.put(("cache", False, str(e)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._poll_after_id = self.root.after(100, self._poll_result)

    def clear_cache_clicked(self) -> None:
        """Remove cached default models (runs in background thread)."""
        self.refresh_cache_btn.config(state=tk.DISABLED)
        self.clear_cache_btn.config(state=tk.DISABLED)
        self.set_status("Clearing cache…")

        def worker() -> None:
            try:
                pretrained.clear_model_cache()
                self.result_queue.put(("cache", True, "Cache cleared."))
            except Exception as e:  # noqa: BLE001
                self.result_queue.put(("cache", False, str(e)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self._poll_after_id = self.root.after(100, self._poll_result)

    def _add_row(
        self,
        label_text: str,
        entry_var: tk.StringVar,
        browse_callback: typing.Callable[[], None] | None = None,
    ) -> tk.Entry:
        lbl = tk.Label(self._main_frame, text=label_text, anchor=tk.W)
        lbl.grid(row=self._row, column=0, sticky=tk.W, pady=(6, 0))
        self._row += 1
        fr = tk.Frame(self._main_frame)
        fr.grid(row=self._row, column=0, sticky=tk.EW, pady=(0, 8))
        self._main_frame.columnconfigure(0, weight=1)
        ent = tk.Entry(
            fr,
            textvariable=entry_var,
            state="readonly" if browse_callback else tk.NORMAL,
        )
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        if browse_callback:
            tk.Button(fr, text="Browse…", command=browse_callback).pack(side=tk.RIGHT)
        self._row += 1
        return ent


# File objects we open for stdout/stderr when frozen; closed at process exit to avoid handle leak.
_stdout_stderr_fallbacks: list = []


def _close_stdout_stderr_fallbacks() -> None:
    """Close file handles opened by _ensure_stdout_stderr_for_frozen (called at exit)."""
    for f in _stdout_stderr_fallbacks:
        with contextlib.suppress(OSError):
            f.close()
    _stdout_stderr_fallbacks.clear()


def _ensure_stdout_stderr_for_frozen() -> None:
    """When running as a frozen Windows exe without console, sys.stdout/stderr can be None.

    Replace them with a safe fallback so code in the app (e.g. gdown during refresh cache)
    does not crash with 'NoneType' object has no attribute 'write'. Opened handles are
    closed at process exit via atexit to avoid leaking file descriptors.
    """
    if sys.stdout is None:
        f = pathlib.Path(os.devnull).open("w")  # noqa: SIM115
        _stdout_stderr_fallbacks.append(f)
        sys.stdout = f
    if sys.stderr is None:
        f = pathlib.Path(os.devnull).open("w")  # noqa: SIM115
        _stdout_stderr_fallbacks.append(f)
        sys.stderr = f
    if _stdout_stderr_fallbacks:
        atexit.register(_close_stdout_stderr_fallbacks)


def _try_single_instance_win32() -> bool:
    """When running as a frozen exe on Windows, ensure only one instance runs.

    Uses a named mutex; if another instance already holds it, shows a message and exits.
    Returns True if this is the only instance (or not frozen/win32), False never returned
    (second instance exits the process).
    """
    if not (getattr(sys, "frozen", False) and sys.platform == "win32"):
        return True
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    ERROR_ALREADY_EXISTS = 183
    MB_OK = 0x00
    MB_ICONWARNING = 0x30
    mutex_name = "StepCOVNet_GenerateUI_SingleInstance"
    handle = kernel32.CreateMutexW(None, True, mutex_name)
    if handle is None:
        return True
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        if handle:
            kernel32.CloseHandle(handle)
        user32.MessageBoxW(
            None,
            "Another instance of StepCOVNet Generator is already running.",
            "Already running",
            MB_OK | MB_ICONWARNING,
        )
        sys.exit(0)
    return True


def main() -> None:
    root = tk.Tk()
    _GeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        multiprocessing.freeze_support()
        _ensure_stdout_stderr_for_frozen()
        _try_single_instance_win32()
    main()
