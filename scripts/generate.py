r"""Script for generating a StepMania chart from an audio file using PyTorch models.

Usage:
    python scripts/generate.py --audio_path=/path/to/song.mp3 --song_title="My Song" --output_file=/path/to/output.txt

    python scripts/generate.py ... --onset_model_path=/path/to/onset.pt --arrow_model_path=/path/to/arrow.pt --bpm=120 ...
"""

import argparse
import pathlib

from stepcovnet import wsl_gpu

wsl_gpu.bootstrap_gpu_script("scripts/generate.py")

from stepcovnet import generator, pretrained, trainers
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
    help="Path to the trained PyTorch onset detection model (.pt).",
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
    help="Use peak-picking post-processing to refine onset timings.",
    default=False,
    required=False,
)
PARSER.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Target compute device ('auto', 'cuda', 'mps', 'cpu').",
)


def main(argv: list[str] | None = None) -> None:
    args = PARSER.parse_args(argv)
    wsl_gpu.guard_pytorch_gpu_job(__file__)
    device = runtime.resolve_device(args.device)
    onset_path = pretrained.resolve_onset_model_path(args.onset_model_path)
    arrow_path = pretrained.resolve_arrow_model_path(args.arrow_model_path)
    onset_model = trainers.load_onset_model(onset_path, device=device)
    arrow_model = trainers.load_arrow_model(arrow_path, device=device)

    output_data = generator.generate_output_data(
        audio_path=args.audio_path,
        song_title=args.song_title,
        bpm=args.bpm,
        onset_model=onset_model,
        arrow_model=arrow_model,
        use_post_processing=args.use_post_processing,
    )

    with pathlib.Path(args.output_file).open("w") as f:
        f.write(output_data.generate_txt_output())
    print(f"Successfully generated step chart at {args.output_file}")


if __name__ == "__main__":
    main()
