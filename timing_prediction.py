import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import pickle

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

sys.path.append(join(os.path.dirname('__file__'), "./src/"))

from utilFunctions import smooth_obs
from audio_preprocessing import getMFCCBands2DMadmom
from madmom.features.onsets import OnsetPeakPickingProcessor


def get_file_names(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def boundary_decoding(obs_i,
                      threshold,
                      hopsize_t,
                      OnsetPeakPickingProcessor):

    """decode boundary"""
    arg_pp = {'threshold': threshold,
              'smooth': 0,
              'fps': 1. / hopsize_t,
              'pre_max': hopsize_t,
              'post_max': hopsize_t}

    peak_picking = OnsetPeakPickingProcessor(**arg_pp)
    i_boundary = peak_picking.process(obs_i)
    i_boundary = np.append(i_boundary, (len(obs_i) - 1) * hopsize_t)
    i_boundary /= hopsize_t
    return i_boundary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate arrow timings from .wav files.")
    parser.add_argument("--wav",
                        type=str,
                        help="input wav path")
    parser.add_argument("--output",
                        type=str,
                        help="output txt path")
    parser.add_argument("--model",
                        type=str,
                        help="trained model path")
    args = parser.parse_args()

    wav_path = args.wav
    out_path = args.output
    model_path = args.model

    wav_names = get_file_names(wav_path)
    existing_pred_timings = get_file_names(out_path)
    model = load_model(join(model_path), custom_objects={'auc': auc})
    with open("training_data/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    print("Starting timings prediction\n-----------------------------------------")

    for wav_name in wav_names:
        if "pred_timings_" + wav_name[:-4] + ".txt" in existing_pred_timings:
            print(wav_name[:-4] + " timings already generated! Skipping...")
            continue

        print("Generating timings for " + wav_name[:-4])

        log_mel = scaler.transform(getMFCCBands2DMadmom(wav_path + wav_name, 44100, 0.01, channel=1))
        log_mel = log_mel.reshape(log_mel.shape[0], 80, 15)
        log_mel = np.expand_dims(log_mel, axis=1)

        pdf = model.predict(log_mel, batch_size=256)
        pdf = np.squeeze(pdf)
        pdf = smooth_obs(pdf)

        timings = boundary_decoding(  obs_i=pdf,
                                      threshold=0.5,
                                      hopsize_t=0.01,
                                      OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

        with open(out_path + "pred_timings_" + wav_name[:-4] + ".txt", "w") as timings_file:
            for timing in timings:
                timings_file.write(str(timing / 100) + "\n")