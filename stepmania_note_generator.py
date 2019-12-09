from scripts_common.utilFunctions import get_file_names
from scripts_wrapper.wav_converter import wav_converter
from scripts_wrapper.arrow_prediction import arrow_prediction
from scripts_wrapper.timing_arrow_combiner import timing_arrow_combiner
from scripts_wrapper.timing_prediction import timing_prediction

from shutil import copyfile, rmtree
from os.path import join

import os
import re
import warnings

warnings.filterwarnings("ignore")


def cleanup():
    print()


def stepmania_note_generator(input_path,
                             output_path,
                             scalers_path,
                             timing_model,
                             arrow_model,
                             overwrite_int):
    if not os.path.isdir(input_path):
        raise OSError('Audio files path %s not found' % input_path)

    if not os.path.isdir(output_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(timing_model):
        raise OSError('Timing model %s is not found' % timing_model)

    if not os.path.isfile(arrow_model):
        raise OSError('Arrow model %s is not found' % arrow_model)

    if overwrite_int == 1:
        overwrite = True
    else:
        overwrite = False

    input_audio_names = get_file_names(input_path)
    existing_txt_names = get_file_names(output_path)

    tmp_folder_name = "_tmp"

    print("Starting audio to txt generation\n-----------------------------------------")

    for input_audio_name in input_audio_names:
        try:
            new_file_name = re.sub("[^a-z0-9-_]", "", "".join(input_audio_name.lower().split(".")[:-1]))
            if not overwrite and 'pred_txt_' + new_file_name + '.txt' in existing_txt_names:
                print("Skipping...", input_audio_name, "txt is already generated!")
                continue

            # create tmp folder
            os.makedirs(join(tmp_folder_name), exist_ok=True)
            # copy audio to tmp folder
            os.makedirs(join(tmp_folder_name, "input"), exist_ok=True)
            copyfile(join(input_path, input_audio_name), join(tmp_folder_name, "input", input_audio_name))

            # convert audio file to wav
            print()
            os.makedirs(join(tmp_folder_name, "wav"), exist_ok=True)
            wav_converter(audio_path=join(tmp_folder_name, "input"),
                          wav_path=join(tmp_folder_name, "wav/"))

            # generate timings for wav
            print()
            os.makedirs(join(tmp_folder_name, "timing"), exist_ok=True)
            timing_prediction(wav_path=join(tmp_folder_name, "wav/"),
                              out_path=join(tmp_folder_name, "timing"),
                              model_path=join(timing_model),
                              scaler_path=join(scalers_path),
                              overwrite_int=overwrite_int)

            # generate arrows for wav
            print()
            os.makedirs(join(tmp_folder_name, "arrows"), exist_ok=True)
            arrow_prediction(timings_path=join(tmp_folder_name, "timing/"),
                             out_path=join(tmp_folder_name, "arrows"),
                             model_path=join(arrow_model),
                             overwrite_int=overwrite_int)

            # combine timings and arrows
            print()
            timing_arrow_combiner(wavs_path=join(tmp_folder_name, "wav/"),
                                  timings_path=join(tmp_folder_name, "timing/"),
                                  arrows_path=join(tmp_folder_name, "arrows/"),
                                  out_path=join(output_path),
                                  overwrite_int=overwrite_int)

            # convert txt to .sm file
            print()
            # need to add sm file writer to this for that to work
            # doing nothing for now

        except Exception:
            print("Skipping... Failed to generate txt from", input_audio_name)
        finally:
            try:
                rmtree(tmp_folder_name)
            except Exception:
                pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate .txt (eventually .sm) files from audio tracks")
    parser.add_argument("--input",
                        type=str,
                        help="input audio files path")
    parser.add_argument("--output",
                        type=str,
                        help="output .txt file path")
    parser.add_argument("--scalers",
                        type=str,
                        default="training_data/",
                        help="scalers used in training path")
    parser.add_argument("--timing_model",
                        type=str,
                        default="models/retrained_timing_model.h5",
                        help="trained timing model path")
    parser.add_argument("--arrow_model",
                        type=str,
                        default="models/retrained_arrow_model.h5",
                        help="trained arrow model path")
    parser.add_argument("--overwrite",
                        type=int,
                        default=0,
                        help="overwrite already created files")
    args = parser.parse_args()

    stepmania_note_generator(args.input,
                             args.output,
                             args.scalers,
                             args.timing_model,
                             args.arrow_model,
                             args.overwrite)
