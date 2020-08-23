import os
import tempfile
import time
import warnings
from os.path import join
from shutil import copyfile

from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder
from stepcovnet.common.utils import standardize_filename
from stepcovnet.inferencing.timing_arrow_combiner import timing_arrow_combiner
from stepcovnet.modeling.StepCOVNetModel import StepCOVNetModel
from stepcovnet.wrapper import wav_converter

warnings.filterwarnings("ignore")


def copy_to_tmp_dir(input_path, tmp_dir_name, batch):
    if batch:
        for input_audio_name in get_filenames_from_folder(input_path):
            new_file_name = standardize_filename(get_filename(input_audio_name, False))
            copyfile(join(input_path, input_audio_name), join(tmp_dir_name, "input", new_file_name))
    else:
        new_file_name = standardize_filename(get_filename(input_path, False))
        copyfile(join(input_path), join(tmp_dir_name, "input", new_file_name))


def build_tmp_dir(tmp_dir_name):
    os.makedirs(join(tmp_dir_name, "input"), exist_ok=True)
    os.makedirs(join(tmp_dir_name, "wav"), exist_ok=True)


def generate_notes(output_path,
                   tmp_dir,
                   stepcovnet_model,
                   verbose_int):
    # Convert audio clip into a wav before preprocessing
    sample_frequency = stepcovnet_model.metadata["dataset_config"]["SAMPLE_RATE"]
    wav_converter(input_path=join(tmp_dir, "input/"),
                  output_path=join(tmp_dir, "wav"),
                  sample_frequency=sample_frequency,
                  verbose_int=verbose_int)

    # combine timings and arrows
    timing_arrow_combiner(wavs_path=join(tmp_folder_name, "wav/"),
                          timings_path=join(tmp_folder_name, "timing/"),
                          arrows_path=join(tmp_folder_name, "arrows/"),
                          output_path=join(output_path),
                          verbose_int=verbose_int)


def stepmania_note_generator(input_path,
                             output_path,
                             model_path,
                             verbose_int=0):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError('%s is not a valid verbose input. Choose 0 for none or 1 for full' % verbose_int)
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(output_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(model_path):
        raise NotADirectoryError('StepCOVNet model %s is not found' % model_path)

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        batch = False if os.path.isfile(input_path) else True
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_tmp_dir(tmp_dir)
            copy_to_tmp_dir(input_path, tmp_dir, batch)
            if verbose:
                print("Loading StepCOVNet model\n-----------------------------------------\n")
                stepcovnet_model = StepCOVNetModel.load(input_path=model_path, retrained=False)
            if verbose:
                print("Starting audio to txt generation\n-----------------------------------------\n")
            generate_notes(output_path, tmp_dir, stepcovnet_model, verbose_int)
    else:
        raise FileNotFoundError('Audio file(s) path %s not found' % input_path)
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate .txt (eventually .sm) files from audio tracks")
    parser.add_argument("-i", "--input",
                        type=str,
                        help="Input audio file/directory path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output .txt files path")
    parser.add_argument("-m", "--model",
                        type=str,
                        help="Input trained StepCOVNet model path")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    stepmania_note_generator(args.input,
                             args.output,
                             args.model,
                             args.verbose)
