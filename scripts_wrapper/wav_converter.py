from scripts_common.utilFunctions import get_filenames_from_folder, get_filename

from os.path import join

import os
import subprocess
import re


def __convert_file(input_path, file_name, output_path, verbose):
    try:
        new_file_name = re.sub("[^a-z0-9-_]", "", get_filename(file_name, False))
        if verbose:
            print("Converting " + file_name)
        subprocess.call(
            ['ffmpeg', '-y', '-loglevel', 'quiet', '-i',
             join(input_path, file_name), '-ar', '44100', join(output_path, new_file_name + '.wav')])
    except Exception:
        if verbose:
            print("Failed to convert", file_name)


def wav_converter(input_path, output_path, verbose_int=0):
    if verbose_int not in [0, 1]:
        raise ValueError('%s is not a valid verbose input. Choose 0 for none or 1 for full' % verbose_int)
    verbose = True if verbose_int == 1 else False

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        if not os.path.isdir(output_path):
            print("Wavs output path not found. Creating directory...")
            os.makedirs(output_path, exist_ok=True)

        if verbose:
            print("Starting .wav conversion\n-----------------------------------------")

        if os.path.isfile(input_path):
            __convert_file(os.path.dirname(input_path), get_filename(input_path), output_path, verbose)
        else:
            file_names = get_filenames_from_folder(input_path)
            for file_name in file_names:
                __convert_file(input_path, file_name, output_path, verbose)
    else:
        raise RuntimeError('Audio file(s) path %s not found' % input_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio files to .wav format.")
    parser.add_argument("-i", "--input",
                        type=str,
                        help="input audio file/path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output wav path")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        help="verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    wav_converter(args.input,
                  args.output,
                  args.verbose)
