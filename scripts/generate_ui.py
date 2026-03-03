r"""Simple UI for generating a StepMania chart from an audio file.

Launch with:
    python scripts/generate_ui.py

Requires trained onset and arrow models (.keras).
"""

import multiprocessing
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import keras

from stepcovnet import (
    generator,
    models,  # noqa: F401 - required for custom Keras layer registration
)

AUDIO_TYPES = [
    ("Audio files", "*.mp3 *.wav *.ogg *.flac"),
    ("All files", "*.*"),
]
MODEL_TYPES = [
    ("Keras models", "*.keras"),
    ("All files", "*.*"),
]
TXT_TYPES = [
    ("Text files", "*.txt"),
    ("All files", "*.*"),
]


def _validate_inputs(
    audio_path: str,
    song_title: str,
    bpm_str: str,
    onset_path: str,
    arrow_path: str,
    output_path: str,
) -> tuple[bool, str | tuple[str, str, int | None, str, str, str]]:
    """Validate UI inputs. Returns (False, error_message) or (True, (audio_path, song_title, bpm, onset_path, arrow_path, output_path)). bpm is None when left blank (estimated from audio)."""
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
    if not onset_path:
        return (False, "Please select the onset model.")
    if not arrow_path:
        return (False, "Please select the arrow model.")
    if not output_path:
        return (False, "Please choose an output file.")
    return (
        True,
        (audio_path, song_title, bpm_val, onset_path, arrow_path, output_path),
    )


def _run_generation(
    audio_path: str,
    song_title: str,
    bpm: int | None,
    onset_model_path: str,
    arrow_model_path: str,
    output_path: str,
    use_post_processing: bool,
    result_queue: queue.Queue,
) -> None:
    """Load models, run generator, write output. Puts (True, output_path) or (False, error_msg) into result_queue."""
    try:
        onset_model = keras.models.load_model(filepath=onset_model_path, compile=False)
        arrow_model = keras.models.load_model(filepath=arrow_model_path, compile=False)
        output_data = generator.generate_output_data(
            audio_path=audio_path,
            song_title=song_title,
            bpm=bpm,
            onset_model=onset_model,  # type: ignore[arg-type]
            arrow_model=arrow_model,  # type: ignore[arg-type]
            use_post_processing=use_post_processing,
        )
        with open(output_path, "w") as f:
            f.write(output_data.generate_txt_output())
        result_queue.put((True, output_path))
    except Exception as e:  # noqa: BLE001
        result_queue.put((False, str(e)))


class _GeneratorApp:
    """Application UI and callbacks for the generator window. Built for testability."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("StepCOVNet Generator")
        root.minsize(400, 320)
        root.geometry("480x540")
        root.resizable(True, True)

        self.result_queue = queue.Queue()

        self.audio_path_var = tk.StringVar()
        self.song_title_var = tk.StringVar()
        self.bpm_var = tk.StringVar()
        self.onset_model_var = tk.StringVar()
        self.arrow_model_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.use_post_processing_var = tk.BooleanVar(value=False)
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
        self._add_row("Onset model (.keras):", self.onset_model_var, self.browse_onset)
        self._add_row("Arrow model (.keras):", self.arrow_model_var, self.browse_arrow)
        self._add_row("Output file (.txt):", self.output_path_var, self.browse_output)

        cb_frame = tk.Frame(main_frame)
        cb_frame.grid(row=self._row, column=0, sticky=tk.W, pady=(4, 12))
        self._row += 1
        tk.Checkbutton(
            cb_frame,
            text="Use post-processing (peak-picking for onset timings)",
            variable=self.use_post_processing_var,
        ).pack(anchor=tk.W)

        self.run_btn.pack(side=tk.LEFT)
        self.status_label.pack(side=tk.LEFT, padx=(16, 0), fill=tk.X, expand=True)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfig(self._canvas_window, width=event.width)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _default_output_path_for_audio(self, audio_path: str) -> str:
        """Suggested output path when user selects an audio file: song title or basename_chart.txt, not basename.txt."""
        out_dir = os.path.dirname(audio_path)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        title = self.song_title_var.get().strip()
        if title:
            # Sanitize for filesystem: replace path separators and other unsafe chars
            safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in title).strip()
            name = safe if safe else base + "_chart"
        else:
            name = base + "_chart"
        return os.path.join(out_dir, name + ".txt")

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
            title="Save chart as",
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
            messagebox.showerror("Validation", result)
            return
        audio_path, song_title, bpm_val, onset_path, arrow_path, output_path = result

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
                use_post_processing=self.use_post_processing_var.get(),
                result_queue=self.result_queue,
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.root.after(100, self._poll_result)

    def _poll_result(self) -> None:
        try:
            success, value = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_result)
            return
        self.run_btn.config(state=tk.NORMAL)
        if success:
            self.set_status(f"Saved to {value}")
            messagebox.showinfo("Success", f"Chart saved to:\n{value}")
        else:
            self.set_status("Error")
            messagebox.showerror("Generation failed", value)

    def _add_row(
        self,
        label_text: str,
        entry_var: tk.StringVar,
        browse_callback: object = None,
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


def main() -> None:
    root = tk.Tk()
    _GeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        multiprocessing.freeze_support()
    main()
