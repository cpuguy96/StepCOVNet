import os
import tempfile
import time
import warnings
from os.path import join
from shutil import copyfile
from typing import Sequence

import joblib

from stepcovnet import config
from stepcovnet.common.utils import (
    get_bpm,
    get_filename,
    get_filenames_from_folder,
    standardize_filename,
    write_file,
)
from stepcovnet.executor.InferenceExecutor import InferenceExecutor
from stepcovnet.inputs.InferenceInput import InferenceInput
from stepcovnet.model.StepCOVNetModel import StepCOVNetModel
from wav_converter import wav_converter

warnings.filterwarnings("ignore")


def copy_to_tmp_dir(input_path: str, tmp_dir_name: str, batch: bool):
    if batch:
        for input_audio_name in get_filenames_from_folder(input_path):
            new_file_name = standardize_filename(get_filename(input_audio_name, False))
            copyfile(
                join(input_path, input_audio_name),
                join(tmp_dir_name, "input", new_file_name),
            )
    else:
        new_file_name = standardize_filename(get_filename(input_path, False))
        copyfile(join(input_path), join(tmp_dir_name, "input", new_file_name))


def build_tmp_dir(tmp_dir_name: str):
    os.makedirs(join(tmp_dir_name, "input"), exist_ok=True)
    os.makedirs(join(tmp_dir_name, "wav"), exist_ok=True)


def get_timings_arrow_mapping(pred_arrows: Sequence[str], hopsize: float) -> dict:
    timings_arrow_mapping = {}
    for i, pred_arrow in enumerate(pred_arrows):
        if pred_arrow != "0000":
            timings_arrow_mapping[str(i * hopsize)] = pred_arrow
    return timings_arrow_mapping


def save_pred_arrows(
    timings_arrows_mapping: dict, output_path: str, file_name: str, bpm: float
):
    header = "TITLE " + str(file_name) + "\n" + "BPM " + str(bpm) + "\n" + "NOTES \n"
    output_data = ""
    for timing, arrow in timings_arrows_mapping.items():
        output_data += str(arrow) + " " + str(timing) + "\n"
    write_file(
        join(output_path, "pred_" + file_name + ".txt"), output_data, header=header
    )


def generate_notes(
    output_path: str, tmp_dir: str, stepcovnet_model: StepCOVNetModel, verbose_int: int
):
    verbose = True if verbose_int == 1 else False

    dataset_config = stepcovnet_model.metadata["dataset_config"]
    lookback = stepcovnet_model.metadata["training_config"]["lookback"]
    difficulty = stepcovnet_model.metadata["training_config"]["difficulty"]
    sample_frequency = dataset_config["SAMPLE_RATE"]
    hopsize = dataset_config["STFT_HOP_LENGTH_SECONDS"]
    audio_files_path = join(tmp_dir, "wav/")
    scalers = joblib.load(
        open(
            os.path.join(
                stepcovnet_model.model_root_path,
                stepcovnet_model.metadata["model_name"] + "_scaler.pkl",
            ),
            "rb",
        )
    )

    # Convert audio clip into a wav before preprocessing
    wav_converter(
        input_path=join(tmp_dir, "input/"),
        output_path=audio_files_path,
        sample_frequency=sample_frequency,
        verbose_int=verbose_int,
    )

    audio_file_names = [
        get_filename(file_name, with_ext=False)
        for file_name in get_filenames_from_folder(audio_files_path)
    ]
    inference_executor = InferenceExecutor(
        stepcovnet_model=stepcovnet_model, verbose=verbose
    )

    for audio_file_name in audio_file_names:
        start_time = time.time()
        if verbose:
            print(
                "Generating notes for %s\n-----------------------------------------\n"
                % audio_file_name
            )
        inference_config = config.InferenceConfig(
            audio_path=audio_files_path,
            file_name=audio_file_name,
            dataset_config=dataset_config,
            lookback=lookback,
            difficulty=difficulty,
            scalers=scalers,
        )
        inference_input = InferenceInput(inference_config=inference_config)
        bpm = get_bpm(wav_file_path=join(audio_files_path, audio_file_name + ".wav"))
        pred_arrows = inference_executor.execute(input_data=inference_input)

        timings_arrows_mapping = get_timings_arrow_mapping(pred_arrows, hopsize=hopsize)
        save_pred_arrows(
            timings_arrows_mapping=timings_arrows_mapping,
            output_path=output_path,
            file_name=audio_file_name,
            bpm=bpm,
        )
        end_time = time.time()
        if verbose:
            print("Elapsed time was %g seconds\n" % (end_time - start_time))


def stepmania_note_generator(
    input_path: str, output_path: str, model_path: str, verbose_int: int = 0
):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError(
            "%s is not a valid verbose input. Choose 0 for none or 1 for full"
            % verbose_int
        )
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(output_path):
        print("Output path not found. Creating directory...")
        os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(model_path):
        raise NotADirectoryError(
            "StepCOVNet model %s is not found" % os.path.abspath(model_path)
        )

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        batch = False if os.path.isfile(input_path) else True
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_tmp_dir(tmp_dir)
            copy_to_tmp_dir(input_path, tmp_dir, batch)
            if verbose:
                print("Loading StepCOVNet retrained model")
            try:
                stepcovnet_model = StepCOVNetModel.load(
                    input_path=model_path, retrained=True
                )
            except OSError:
                if verbose:
                    print(
                        "Failed to retrieve retrained StepCOVNet model. Loading non-retrained model"
                    )
                stepcovnet_model = StepCOVNetModel.load(
                    input_path=model_path, retrained=False
                )
            if verbose:
                print(
                    "Starting audio to txt generation\n-----------------------------------------\n"
                )
            generate_notes(output_path, tmp_dir, stepcovnet_model, verbose_int)
    else:
        raise FileNotFoundError(
            "Audio file(s) path %s not found" % os.path.abspath(input_path)
        )
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate .txt (eventually .sm) files from audio tracks"
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Input audio file/directory path", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output .txt files path", required=True
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Input trained StepCOVNet model path",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="Verbosity: 0 - none, 1 - full",
    )
    args = parser.parse_args()

    stepmania_note_generator(args.input, args.output, args.model, args.verbose)
