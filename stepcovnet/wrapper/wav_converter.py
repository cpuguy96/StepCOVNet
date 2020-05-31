import multiprocessing
import os
import time
from functools import partial
from os.path import join

import psutil
import soundfile as sf

from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder
from stepcovnet.common.utils import standardize_filename


def convert_file(input_path,
                 output_path,
                 sample_frequency,
                 verbose,
                 file_name):
    try:
        new_file_name = standardize_filename(get_filename(file_name, False))
        if verbose:
            print("Converting " + file_name)
        file_input_path = join(input_path, file_name)
        file_output_path = join(output_path, new_file_name + '.wav')
        input_audio_data, _ = sf.read(file_input_path)
        sf.write(file_output_path, input_audio_data, sample_frequency)
    except Exception as ex:
        if verbose:
            print("Failed to convert %s: %r" % (file_name, ex))


def run_process(input_path, output_path, sample_frequency, verbose):
    if os.path.isfile(input_path):
        convert_file(os.path.dirname(input_path), output_path, sample_frequency, verbose, get_filename(input_path))
    else:
        file_names = get_filenames_from_folder(input_path)
        func = partial(convert_file, input_path, output_path, sample_frequency, verbose)
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            pool.map_async(func, file_names).get()


def wav_converter(input_path, output_path, sample_frequency, verbose_int=0):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError('%s is not a valid verbose input. Choose 0 for none or 1 for full' % verbose_int)
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(output_path):
        print("Wavs output path not found. Creating directory...")
        os.makedirs(output_path, exist_ok=True)

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        if verbose:
            print("Starting .wav conversion\n-----------------------------------------")
        run_process(input_path, output_path, sample_frequency, verbose)
    else:
        raise FileNotFoundError('Audio file(s) path %s not found' % input_path)
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio files to .wav format")
    parser.add_argument("-i", "--input",
                        type=str,
                        required=True,
                        help="Input audio file/directory path")
    parser.add_argument("-o", "--output",
                        type=str,
                        required=True,
                        help="Output wavs path")
    parser.add_argument("-sf", "--sample_frequency",
                        type=int,
                        default=44100,
                        help="Sampling frequency to create wavs")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    wav_converter(args.input,
                  args.output,
                  args.sample_frequency,
                  args.verbose)
