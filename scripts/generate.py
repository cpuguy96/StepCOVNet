r"""Script for generating a StepMania chart from an audio file.

Usage:
    python scripts/generate.py --audio_path=/path/to/song.mp3 --song_title="My Song" --bpm=120 --onset_model_path=/path/to/onset.keras --arrow_model_path=/path/to/arrow.keras --output_file=/path/to/output.txt
"""
import argparse

import keras

from stepcovnet import generator
from stepcovnet import models

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
    help="BPM of the song.",
    required=True,
)
PARSER.add_argument(
    "--onset_model_path",
    type=str,
    help="Path to the trained onset detection model (.keras).",
    required=True,
)
PARSER.add_argument(
    "--arrow_model_path",
    type=str,
    help="Path to the trained arrow prediction model (.keras).",
    required=True,
)
PARSER.add_argument(
    "--output_file",
    type=str,
    help="Path where the generated chart will be saved.",
    required=True,
)

ARGS = PARSER.parse_args()


def main():
    onset_model = keras.models.load_model(
        filepath=ARGS.onset_model_path,
        compile=False,
        custom_objects={"_crop_to_match": models._crop_to_match}
    )
    arrow_model = keras.models.load_model(
        filepath=ARGS.arrow_model_path,
        compile=False,
        custom_objects={"PositionalEncoding": models.PositionalEncoding}
    )

    output_data = generator.generate_output_data(
        audio_path=ARGS.audio_path,
        song_title=ARGS.song_title,
        bpm=ARGS.bpm,
        onset_model=onset_model,
        arrow_model=arrow_model
    )

    with open(ARGS.output_file, "w") as f:
        f.write(output_data.generate_txt_output())
    print(f"Successfully generated step chart at {ARGS.output_file}")


if __name__ == "__main__":
    main()
