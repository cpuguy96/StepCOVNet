from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa


def get_file_names(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_bpm(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    return librosa.beat.beat_track(y=y, sr=sr)


def main():
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

    if not os.path.isdir(args.wav):
        raise OSError('Input path %s not found' % args.wav)

    if not os.path.isdir(args.timing):
        raise OSError('Timing files path %s not found' % args.timing)

    if not os.path.isdir(args.arrow):
        raise OSError('Arrow files path %s not found' % args.arrow)

    if not os.path.isdir(args.output):
        print('Output path not found. Creating directory...')
        os.makedirs(args.output, exist_ok=True)

    wavs_path = args.wav
    timings_path = args.timing
    arrows_path = args.arrow
    out_path = args.output

    if args.overwrite == 1:
        overwrite = True
    else:
        overwrite = False

    wav_names = get_file_names(wavs_path)
    existing_txts = get_file_names(out_path)

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
    main()