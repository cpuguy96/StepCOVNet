from __future__ import print_function
from scripts_common.utilFunctions import get_filenames_from_folder

from os.path import join

import os
import numpy as np
import librosa


def get_bpm(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    return librosa.beat.beat_track(y=y, sr=sr)


def timing_arrow_combiner(wavs_path,
                          timings_path,
                          arrows_path,
                          out_path,
                          overwrite_int):
    if not os.path.isdir(wavs_path):
        raise NotADirectoryError('Input path %s not found' % wavs_path)

    if not os.path.isdir(timings_path):
        raise NotADirectoryError('Timing files path %s not found' % timings_path)

    if not os.path.isdir(arrows_path):
        raise NotADirectoryError('Arrow files path %s not found' % arrows_path)

    if not os.path.isdir(out_path):
        print('Output path not found. Creating directory...')
        os.makedirs(out_path, exist_ok=True)

    if overwrite_int == 1:
        overwrite = True
    else:
        overwrite = False

    wav_names = get_filenames_from_folder(wavs_path)
    existing_txts = get_filenames_from_folder(out_path)

    print("Starting combined txt generation\n-----------------------------------------")

    for wav_name in wav_names:
        if not wav_name.endswith(".wav"):
            print(wav_name, "is not a wav file! Skipping...")
            continue

        if "pred_txt_" + wav_name[:-4] + ".txt" in existing_txts and not overwrite:
            print(wav_name[:-4] + " txt already generated! Skipping...")
            continue

        song_name = wav_name[:-4]

        try:
            with open(join(timings_path, "pred_timings_" + song_name + ".txt"), "r") as timings_file:
                timings = np.asarray([line.replace("\n", "") for line in timings_file.readlines()]).astype("float32")
        except Exception:
            print(song_name + " timings not found! Skipping...")
            continue

        try:
            with open(join(arrows_path, "pred_arrows_" + song_name + ".txt"), "r") as arrows_file:
                arrows = np.asarray([line.replace("\n", "") for line in arrows_file.readlines()])
        except Exception:
            print(song_name + " arrows not found! Skipping...")
            continue

        try:
            bpm, _ = get_bpm(join(wavs_path, wav_name))
        except Exception:
            print(wav_name + " not found! Skipping...")
            continue

        print("Creating combined txt file for " + wav_name[:-4])

        with open(join(out_path, "pred_txt_" + song_name + ".txt"), "w") as comb_file:
            comb_file.write("TITLE " + str(song_name) + "\n")
            comb_file.write("BPM " + str(bpm) + "\n")
            comb_file.write("NOTES \n")
            for timing, arrow in zip(timings, arrows):
                comb_file.write(str(arrow) + " " + str(timing) + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Combine")
    parser.add_argument("--wav",
                        type=str,
                        help="input wavs path")
    parser.add_argument("--timing",
                        type=str,
                        help="input timings path")
    parser.add_argument("--arrow",
                        type=str,
                        help="input arrows path")
    parser.add_argument("--output",
                        type=str,
                        help="output txt path")
    parser.add_argument("--overwrite",
                        type=int,
                        default=0,
                        help="overwrite already created files")
    args = parser.parse_args()

    timing_arrow_combiner(args.wav,
                          args.timing,
                          args.arrow,
                          args.output,
                          args.overwrite)
