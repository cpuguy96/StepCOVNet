from __future__ import print_function
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa


def get_file_names(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_bpm(wav_file_path):
    y, sr = librosa.load(wav_file_path)
    return librosa.beat.beat_track(y=y, sr=sr)


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
    args = parser.parse_args()

    wavs_path = args.wav
    timings_path = args.timing
    arrows_path = args.arrow
    out_path = args.output

    wavs_file_names = get_file_names(wavs_path)
    existing_txts = get_file_names(out_path)

    print("Starting combined txt generation\n-----------------------------------------")

    for wavs_file_name in wavs_file_names:
        if "pred_txt_" + wavs_file_name[:-4] + ".txt" in existing_txts:
            print(wavs_file_name[:-4] + " txt already generated! Skipping...")
            continue

        song_name = wavs_file_name[:-4]

        bpm = -1
        try:
            bpm, _ = get_bpm(wavs_path + wavs_file_name)
        except Exception:
            print(wavs_file_name + " not found! Skipping...")
            continue

        timings = []
        try:
            with open(timings_path + "pred_timings_" + song_name + ".txt", "r") as timings_file:
                timings = np.asarray([line.replace("\n", "") for line in timings_file.readlines()]).astype("float32")
        except Exception:
            print(song_name + " timings not found! Skipping...")
            continue

        arrows = []
        try:
            with open(arrows_path + "pred_arrows_" + song_name + ".txt", "r") as arrows_file:
                arrows = np.asarray([line.replace("\n", "") for line in arrows_file.readlines()])
        except Exception:
            print(song_name + " arrows not found! Skipping...")
            continue

        print("Creating combined txt file for " + wavs_file_name[:-4])

        with open(out_path + "pred_txt_" + song_name + ".txt", "w") as comb_file:
            comb_file.write("TITLE " + str(song_name) + "\n")
            comb_file.write("BPM " + str(bpm) + "\n")
            comb_file.write("NOTES \n")
            for timing, arrow in zip(timings, arrows):
                comb_file.write(str(arrow) + " " + str(timing) + "\n")
