import multiprocessing
import os
import time
from functools import partial
from os.path import join

import numpy as np
import psutil
import resampy
import soundfile as sf

from stepcovnet import utils


def convert_file(
    input_path: str,
    output_path: str,
    sample_frequency: int,
    verbose: bool,
    file_name: str,
):
    try:
        new_file_name = utils.standardize_filename(utils.get_filename(file_name, False))
        if verbose:
            print("Converting " + file_name)
        file_input_path = join(input_path, file_name)
        file_output_path = join(output_path, new_file_name + ".wav")
        input_audio_data, input_audio_sample_rate = sf.read(file_input_path)
        if input_audio_data.shape[1] > 1:
            input_audio_data = np.mean(input_audio_data, axis=1)
        else:
            input_audio_data = np.squeeze(input_audio_data)
        if input_audio_sample_rate != sample_frequency:
            input_audio_data = resampy.resample(
                input_audio_data,
                sr_orig=input_audio_sample_rate,
                sr_new=sample_frequency,
            )
        sf.write(file_output_path, input_audio_data, sample_frequency)
    except Exception as ex:
        if verbose:
            print("Failed to convert %s: %r" % (file_name, ex))


def run_process(
    input_path: str, output_path: str, sample_frequency: int, cores: int, verbose: bool
):
    if os.path.isfile(input_path):
        convert_file(
            os.path.dirname(input_path),
            output_path,
            sample_frequency,
            verbose,
            utils.get_filename(input_path),
        )
    else:
        file_names = utils.get_filenames_from_folder(input_path)
        func = partial(convert_file, input_path, output_path, sample_frequency, verbose)
        with multiprocessing.Pool(cores) as pool:
            pool.map_async(func, file_names).get()


def wav_converter(
    input_path: str,
    output_path: str,
    sample_frequency: int = 16000,
    cores: int = 1,
    verbose_int: int = 0,
):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError(
            "%s is not a valid verbose input. Choose 0 for none or 1 for full"
            % verbose_int
        )
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(output_path):
        print("Wavs output path not found. Creating directory...")
        os.makedirs(output_path, exist_ok=True)

    if cores > os.cpu_count() or cores == 0:
        raise ValueError(
            "Number of cores selected must not be 0 and must be less than the number cpu cores (%d)"
            % os.cpu_count()
        )

    cores = psutil.cpu_count(logical=False) if cores < 0 else cores

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        if verbose:
            print("Starting .wav conversion\n-----------------------------------------")
        run_process(input_path, output_path, sample_frequency, cores, verbose)
    else:
        raise FileNotFoundError(
            "Audio file(s) path %s not found" % os.path.abspath(input_path)
        )
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio files to .wav format")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input audio file/directory path"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output wavs path"
    )
    parser.add_argument(
        "-sf",
        "--sample_frequency",
        type=int,
        default=16000,
        help="Sampling frequency to create wavs",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of processor cores to use for parallel processing: -1 max number of physical cores",
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

    wav_converter(
        args.input, args.output, args.sample_frequency, args.cores, args.verbose
    )
