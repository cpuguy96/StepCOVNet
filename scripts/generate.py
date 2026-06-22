r"""Script for generating a StepMania chart from an audio file.

Usage:
    python scripts/generate.py --audio_path=/path/to/song.mp3 --song_title="My Song" --output_file=/path/to/output.txt

    If onset/arrow model paths are omitted, the latest models are downloaded from Google Drive
    (see stepcovnet.pretrained) and cached locally. BPM is optional (estimated from audio when omitted).
    python scripts/generate.py ... --onset_model_path=/path/to/onset.keras --arrow_model_path=/path/to/arrow.keras --bpm=120 ...
"""

import argparse
import pathlib

import keras

from stepcovnet import (
    generator,
    models,  # noqa: F401 (required to ensure registration of custom Keras layers/functions for model loading)
    pretrained,
)

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
    help="Path to the trained onset detection model (.keras). Optional; if omitted, downloaded from Google Drive and cached.",
)
PARSER.add_argument(
    "--arrow_model_path",
    type=str,
    default=None,
    help="Path to the trained arrow prediction model (.keras). Optional; if omitted, downloaded from Google Drive and cached.",
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
    help="Use peak-picking post-processing to refine onset timings (recommended for cleaner charts).",
    default=False,
    required=False,
)

ARGS = PARSER.parse_args()


def main() -> None:
    onset_path = pretrained.resolve_onset_model_path(ARGS.onset_model_path)
    arrow_path = pretrained.resolve_arrow_model_path(ARGS.arrow_model_path)
    onset_model = keras.models.load_model(filepath=onset_path, compile=False)
    arrow_model = keras.models.load_model(filepath=arrow_path, compile=False)

    output_data = generator.generate_output_data(
        audio_path=ARGS.audio_path,
        song_title=ARGS.song_title,
        bpm=ARGS.bpm,
        onset_model=onset_model,  # type: ignore
        arrow_model=arrow_model,  # type: ignore
        use_post_processing=ARGS.use_post_processing,
    )

    with pathlib.Path(ARGS.output_file).open("w") as f:
        f.write(output_data.generate_txt_output())
    print(f"Successfully generated step chart at {ARGS.output_file}")


if __name__ == "__main__":
    main()
