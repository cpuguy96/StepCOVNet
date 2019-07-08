import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

#from keras import backend as K
#from keras.models import load_model
from sklearn.externals import joblib




sys.path.append(join(os.path.dirname('__file__'), "./src/"))

from utilFunctions import smooth_obs
from audio_preprocessing import getMFCCBands2DMadmom
from madmom.features.onsets import OnsetPeakPickingProcessor
from training_scripts.data_preparation import featureReshape


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

    custom_objects = {'GlorotNormal': tf.keras.initializers.glorot_normal,
                      'GlorotUniform': tf.keras.initializers.glorot_uniform}
    model = load_model(join(model_path), custom_objects=custom_objects)

    if model.layers[0].input_shape[0][1] != 1:
        multi = True
    else:
        multi = False

    scaler = []

    if multi:
        with open(join("training_data", "scaler_low.pkl"), "rb") as file:
            scaler.append(joblib.load(file))
        with open(join("training_data", "scaler_mid.pkl"), "rb") as file:
            scaler.append(joblib.load(file))
        with open(join("training_data", "scaler_high.pkl"), "rb") as file:
            scaler.append(joblib.load(file))
    else:
        with open(join("training_data", "scaler.pkl"), "rb") as file:
            scaler.append(joblib.load(file))

    print("Starting timings prediction\n-----------------------------------------")

    for wav_name in wav_names:
        if not wav_name.endswith(".wav"):
            print(wav_name, "is not a wav file! Skipping...")
            continue
        if "pred_timings_" + wav_name[:-4] + ".txt" in existing_pred_timings:
            print(wav_name[:-4] + " timings already generated! Skipping...")
            continue

        print("Generating timings for " + wav_name[:-4])

        if multi:
            log_mel = getMFCCBands2DMadmom(join(wav_path + wav_name), 44100, 0.01, channel=3)

            log_mel[:, :, 0] = scaler[0].transform(log_mel[:, :, 0])
            log_mel[:, :, 1] = scaler[1].transform(log_mel[:, :, 1])
            log_mel[:, :, 2] = scaler[2].transform(log_mel[:, :, 2])
        else:
            log_mel = getMFCCBands2DMadmom(join(wav_path + wav_name), 44100, 0.01, channel=1)
            log_mel = scaler[0].transform(log_mel)

        log_mel_re = featureReshape(log_mel, multi, 7)
        if not multi:
            log_mel_re = np.expand_dims(log_mel_re, axis=1)
        pdf = model.predict(log_mel_re)
        pdf = np.squeeze(pdf)
        pdf = smooth_obs(pdf)

        timings = boundary_decoding(obs_i=pdf,
                                    threshold=0.5,
                                    hopsize_t=0.01,
                                    OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

        with open(join(out_path, "pred_timings_" + wav_name[:-4] + ".txt"), "w") as timings_file:
            for timing in timings:
                timings_file.write(str(timing / 100) + "\n")