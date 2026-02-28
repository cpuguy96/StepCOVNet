r"""Simple UI for generating a StepMania chart from an audio file.

Launch with:
    python scripts/generate_ui.py

Requires trained onset and arrow models (.keras).
"""

import os
import queue
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
) -> tuple[bool, str | tuple[str, str, int, str, str, str]]:
    """Validate UI inputs. Returns (False, error_message) or (True, (audio_path, song_title, bpm, onset_path, arrow_path, output_path))."""
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
        return (False, "Please enter BPM.")
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
    bpm: int,
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


def main() -> None:
    root = tk.Tk()
    root.title("StepCOVNet Generator")
    root.minsize(400, 320)
    root.geometry("480x540")
    root.resizable(True, True)

    result_queue = queue.Queue()

    # Variables
    audio_path_var = tk.StringVar()
    song_title_var = tk.StringVar()
    bpm_var = tk.StringVar()
    onset_model_var = tk.StringVar()
    arrow_model_var = tk.StringVar()
    output_path_var = tk.StringVar()
    use_post_processing_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="")

    outer = tk.Frame(root, padx=12, pady=12)
    outer.pack(fill=tk.BOTH, expand=True)

    # Bottom bar (button + status) stays fixed so it is never cut off
    bottom_frame = tk.Frame(outer)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))

    # Scrollable form area above the bottom bar (grid so scrollbar keeps fixed width when narrow)
    scroll_area = tk.Frame(outer)
    scroll_area.pack(fill=tk.BOTH, expand=True)
    scroll_area.rowconfigure(0, weight=1)
    scroll_area.columnconfigure(0, weight=1, minsize=80)
    scroll_area.columnconfigure(1, weight=0, minsize=16)

    canvas = tk.Canvas(scroll_area, highlightthickness=0)
    scrollbar = tk.Scrollbar(
        scroll_area, orient=tk.VERTICAL, command=canvas.yview, width=16
    )
    scrollable_frame = tk.Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
    canvas.configure(yscrollcommand=scrollbar.set)

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.bind("<MouseWheel>", _on_mousewheel)

    canvas.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 12))
    scrollbar.grid(row=0, column=1, sticky=tk.NS)

    main_frame = scrollable_frame

    def browse_audio() -> None:
        path = filedialog.askopenfilename(
            filetypes=AUDIO_TYPES, title="Select audio file"
        )
        if path:
            audio_path_var.set(path)
            if not output_path_var.get().strip():
                base = os.path.splitext(os.path.basename(path))[0]
                out_dir = os.path.dirname(path)
                output_path_var.set(os.path.join(out_dir, base + ".txt"))

    def browse_onset() -> None:
        path = filedialog.askopenfilename(
            filetypes=MODEL_TYPES, title="Select onset model"
        )
        if path:
            onset_model_var.set(path)

    def browse_arrow() -> None:
        path = filedialog.askopenfilename(
            filetypes=MODEL_TYPES, title="Select arrow model"
        )
        if path:
            arrow_model_var.set(path)

    def browse_output() -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=TXT_TYPES,
            title="Save chart as",
        )
        if path:
            output_path_var.set(path)

    def set_status(msg: str) -> None:
        status_var.set(msg)
        status_label.update_idletasks()

    def run_clicked() -> None:
        ok, result = _validate_inputs(
            audio_path_var.get(),
            song_title_var.get(),
            bpm_var.get(),
            onset_model_var.get(),
            arrow_model_var.get(),
            output_path_var.get(),
        )
        if not ok:
            messagebox.showerror("Validation", result)
            return
        audio_path, song_title, bpm_val, onset_path, arrow_path, output_path = result

        run_btn.config(state=tk.DISABLED)
        set_status("Processing…")

        def worker() -> None:
            _run_generation(
                audio_path=audio_path,
                song_title=song_title,
                bpm=bpm_val,
                onset_model_path=onset_path,
                arrow_model_path=arrow_path,
                output_path=output_path,
                use_post_processing=use_post_processing_var.get(),
                result_queue=result_queue,
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        root.after(100, _poll_result)

    def _poll_result() -> None:
        try:
            success, value = result_queue.get_nowait()
        except queue.Empty:
            root.after(100, _poll_result)
            return
        run_btn.config(state=tk.NORMAL)
        if success:
            set_status(f"Saved to {value}")
            messagebox.showinfo("Success", f"Chart saved to:\n{value}")
        else:
            set_status("Error")
            messagebox.showerror("Generation failed", value)

    # Layout
    row = 0

    def add_row(
        label_text: str, entry_var: tk.StringVar, browse_callback=None
    ) -> tk.Entry:
        nonlocal row
        lbl = tk.Label(main_frame, text=label_text, anchor=tk.W)
        lbl.grid(row=row, column=0, sticky=tk.W, pady=(6, 0))
        row += 1
        fr = tk.Frame(main_frame)
        fr.grid(row=row, column=0, sticky=tk.EW, pady=(0, 8))
        main_frame.columnconfigure(0, weight=1)
        ent = tk.Entry(
            fr,
            textvariable=entry_var,
            state="readonly" if browse_callback else tk.NORMAL,
        )
        ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        if browse_callback:
            tk.Button(fr, text="Browse…", command=browse_callback).pack(side=tk.RIGHT)
        row += 1
        return ent

    add_row("Audio file:", audio_path_var, browse_audio)
    add_row("Song title:", song_title_var)
    add_row("BPM:", bpm_var)
    add_row("Onset model (.keras):", onset_model_var, browse_onset)
    add_row("Arrow model (.keras):", arrow_model_var, browse_arrow)
    add_row("Output file (.txt):", output_path_var, browse_output)

    cb_frame = tk.Frame(main_frame)
    cb_frame.grid(row=row, column=0, sticky=tk.W, pady=(4, 12))
    row += 1
    tk.Checkbutton(
        cb_frame,
        text="Use post-processing (peak-picking for onset timings)",
        variable=use_post_processing_var,
    ).pack(anchor=tk.W)

    # Generate button and status live in bottom bar so they are never cut off
    run_btn = tk.Button(bottom_frame, text="Generate chart", command=run_clicked)
    run_btn.pack(side=tk.LEFT)
    status_label = tk.Label(
        bottom_frame, textvariable=status_var, anchor=tk.W, fg="gray"
    )
    status_label.pack(side=tk.LEFT, padx=(16, 0), fill=tk.X, expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
