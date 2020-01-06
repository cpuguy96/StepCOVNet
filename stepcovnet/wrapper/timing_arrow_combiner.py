import multiprocessing
import os
import time
from functools import partial
from os.path import join

import librosa
import numpy as np
import psutil

from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder


def get_bpm(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    return librosa.beat.beat_track(y=y, sr=sr)


def combine_data(timings_path,
                 arrows_path,
                 wavs_path,
                 output_path,
                 verbose,
                 wav_name):
    if not wav_name.endswith(".wav"):
        print("%s is not a wav file! Skipping..." % wav_name)
        return

    song_name = get_filename(wav_name, False)

    try:
        with open(join(timings_path, song_name + ".timings"), "r") as timings_file:
            timings = np.asarray([line.replace("\n", "") for line in timings_file.readlines()]).astype("float32")
    except FileNotFoundError:
        print("%s timings not found! Skipping..." % song_name)
        return

    try:
        with open(join(arrows_path, song_name + ".arrows"), "r") as arrows_file:
            arrows = np.asarray([line.replace("\n", "") for line in arrows_file.readlines()])
    except FileNotFoundError:
        print("%s arrows not found! Skipping..." % song_name)
        return

    try:
        bpm, _ = get_bpm(join(wavs_path, wav_name))
    except FileNotFoundError:
        print("%s not found! Skipping..." % wav_name)
        return

    if verbose:
        print("Creating combined txt file for %s" % wav_name)

    with open(join(output_path, "generated_" + song_name + ".txt"), "w") as comb_file:
        comb_file.write("TITLE " + str(song_name) + "\n")
        comb_file.write("BPM " + str(bpm) + "\n")
        comb_file.write("NOTES \n")
        for timing, arrow in zip(timings, arrows):
            comb_file.write(str(arrow) + " " + str(timing) + "\n")


def run_process(timings_path,
                arrows_path,
                wavs_path,
                output_path,
                verbose):
    if os.path.isfile(timings_path):
        combine_data(os.path.dirname(timings_path), arrows_path, wavs_path, output_path, verbose,
                     get_filename(wavs_path))
    else:
        wav_names = get_filenames_from_folder(wavs_path)
        func = partial(combine_data, timings_path, arrows_path, wavs_path, output_path, verbose)
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            pool.map_async(func, wav_names).get()


def timing_arrow_combiner(wavs_path,
                          timings_path,
                          arrows_path,
                          output_path,
                          verbose_int=0):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError('%s is not a valid verbose input. Choose 0 for none or 1 for full' % verbose_int)
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(wavs_path) and not os.path.isfile(wavs_path):
        raise FileNotFoundError('Input path %s not found' % wavs_path)

    if not os.path.isdir(timings_path) and not os.path.isfile(timings_path):
        raise FileNotFoundError('Timing files path %s not found' % timings_path)

    if not os.path.isdir(arrows_path) and not os.path.isfile(arrows_path):
        raise FileNotFoundError('Arrow files path %s not found' % arrows_path)

    if not os.path.isdir(output_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if verbose:
        print("Starting combined txt generation\n-----------------------------------------")

    run_process(timings_path, arrows_path, wavs_path, output_path, verbose)
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Combine predicted timings and predicted arrows")
    parser.add_argument("-w", "--wav",
                        type=str,
                        help="Input wav file/directory path")
    parser.add_argument("-t", "--timing",
                        type=str,
                        help="Input timings path")
    parser.add_argument("-a", "--arrow",
                        type=str,
                        help="Input arrows path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output txts path")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    timing_arrow_combiner(args.wav,
                          args.timing,
                          args.arrow,
                          args.output,
                          args.verbose)
