r"""Script for generating a StepMania chart from an audio file using PyTorch models.

Usage:
    python scripts/generate.py --audio_path=/path/to/song.mp3 --song_title="My Song" --output_file=/path/to/output.txt

    python scripts/generate.py ... --onset_model_path=/path/to/onset.pt --arrow_model_path=/path/to/arrow.pt --bpm=120 ...
"""

import argparse
import pathlib

import torch

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/generate.py")

from stepcovnet import beat_tracking, generator, pretrained, trainers
from stepcovnet.itgpt import trainers as itgpt_trainers
from stepcovnet.pytorch import runtime

PARSER = argparse.ArgumentParser(description="Generate step chart from audio.")
PARSER.add_argument(
    "--audio_path",
    type=str,
    help="Path to the input audio file.",
    required=True,
)
PARSER.add_argument(
    "--song_title",
    type=str,
    help="Title of the song.",
    required=True,
)
PARSER.add_argument(
    "--bpm",
    type=int,
    default=None,
    help="BPM of the song. Optional; if omitted, estimated from the audio.",
)
PARSER.add_argument(
    "--onset_model_path",
    type=str,
    default=None,
    help=(
        "Path to the trained PyTorch onset model (.pt); supports both "
        "ITGPT-MERT 48-slot checkpoints and legacy dense onset checkpoints."
    ),
)
PARSER.add_argument(
    "--arrow_model_path",
    type=str,
    default=None,
    help="Path to the trained PyTorch arrow prediction model (.pt).",
)
PARSER.add_argument(
    "--output_file",
    type=str,
    help="Path where the generated chart will be saved.",
    required=True,
)
PARSER.add_argument(
    "--use_post_processing",
    type=bool,
    help="Use peak-picking post-processing to refine legacy dense onset timings.",
    default=False,
    required=False,
)
PARSER.add_argument(
    "--beat_tracker",
    type=str,
    choices=list(beat_tracking.VALID_TRACKERS),
    default=beat_tracking.TRACKER_AUTO,
    help=(
        "Beat tracker backend for raw audio ('auto' prefers beat_this when "
        "installed and falls back to librosa)."
    ),
)
PARSER.add_argument(
    "--snap_beats",
    action="store_true",
    help=(
        "Apply a song-level constant phase shift aligning estimated beats "
        "to audio onset-strength peaks."
    ),
)
PARSER.add_argument(
    "--target_density",
    type=float,
    default=2.0,
    help=(
        "Target average steps per beat for density-conditioned ITGPT-MERT "
        "models (e.g. 0.5 Beginner, 1.5 Medium, 2.0 Hard, 3.0 Challenge)."
    ),
)
PARSER.add_argument(
    "--meter",
    type=int,
    default=10,
    help="Difficulty meter for meter-conditioned ITGPT checkpoints.",
)
PARSER.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Onset probability threshold (default: 0.5).",
)
PARSER.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Target compute device ('auto', 'cuda', 'mps', 'cpu').",
)


def _is_itgpt_checkpoint(path: str) -> bool:
    """Return True if ``path`` holds a saved PyTorch ITGPT placement checkpoint."""
    payload = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return False
    model_cfg = payload.get("model_config")
    return isinstance(model_cfg, dict) and "n_enc_layers" in model_cfg


def _load_onset_model_auto(onset_path: str, *, device: object) -> object:
    """Load either an ITGPT-MERT placement model or a legacy dense onset model."""
    if _is_itgpt_checkpoint(onset_path):
        return itgpt_trainers.load_placement_model(onset_path, device=device)  # type: ignore[arg-type]
    return trainers.load_onset_model(onset_path, device=device)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> None:
    args = PARSER.parse_args(argv)
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    device = runtime.resolve_device(args.device)
    onset_path = pretrained.resolve_onset_model_path(args.onset_model_path)
    arrow_path = pretrained.resolve_arrow_model_path(args.arrow_model_path)
    onset_model = _load_onset_model_auto(onset_path, device=device)
    arrow_model = trainers.load_arrow_model(arrow_path, device=device)

    output_data = generator.generate_output_data(
        audio_path=args.audio_path,
        song_title=args.song_title,
        bpm=args.bpm,
        onset_model=onset_model,
        arrow_model=arrow_model,
        use_post_processing=args.use_post_processing,
        beat_tracker=args.beat_tracker,
        snap_beats=args.snap_beats,
        target_density=args.target_density,
        meter=args.meter,
        threshold=args.threshold,
    )

    with pathlib.Path(args.output_file).open("w") as f:
        f.write(output_data.generate_txt_output())
    print(f"Successfully generated step chart at {args.output_file}")


if __name__ == "__main__":
    main()
