from common.audio_preprocessing import getMFCCBands2DMadmom
from common.utilFunctions import get_file_names
from madmom.features.onsets import OnsetPeakPickingProcessor
from training_scripts.data_preparation import featureReshape

from os.path import join
from tensorflow.keras.models import load_model

import os
import numpy as np
import xgboost as xgb
import joblib


def smooth_obs(obs):
    """using moving average hanning window for smoothing"""
    hann = np.hanning(5)
    hann /= np.sum(hann)
    obs = np.convolve(hann, obs, mode='same')
    return obs


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


def main():
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
    parser.add_argument("--pca",
                        type=str,
                        help="trained pca path")
    parser.add_argument("--model_type",
                        type=int,
                        default=0,
                        help="type of model: 0 - CNN; 1 - XGB; 2 - multi XGB")
    parser.add_argument("--overwrite",
                        type=int,
                        default=0,
                        help="overwrite already created files")
    args = parser.parse_args()

    if not os.path.isdir(args.wav):
        raise OSError('Wavs path %s not found' % args.wav)

    if not os.path.isdir(args.output):
        print('Output path not found. Creating directory...')
        os.makedirs(args.output, exist_ok=True)

    if not os.path.isfile(args.model):
        raise OSError('Model %s is not found' % args.model)

    if args.model_type not in [0, 1, 2]:
        raise OSError('Model type %s is not a valid model' % args.model_type)

    if args.model_type in [1, 2] and not os.path.isfile(args.pca):
        raise OSError('PCA %s is not found' % args.pca)

    wav_path = args.wav
    out_path = args.output
    model_path = args.model

    if args.overwrite == 1:
        overwrite = True
    else:
        overwrite = False

    wav_names = get_file_names(wav_path)
    existing_pred_timings = get_file_names(out_path)

    model_type = args.model_type

    if model_type == 0:
        custom_objects = {}

        model = load_model(join(model_path), custom_objects=custom_objects, compile=False)

        if model.layers[0].input_shape[0][1] != 1:
            multi = True
        else:
            multi = False
    else:
        model = xgb.Booster({'nthread': -1})
        model.load_model(join(model_path))
        pca = joblib.load(join(args.pca))
        if model_type == 1:
            multi = False
        else:
            multi = True

    scaler = []

    if multi:
        with open(join("training_data", "multi_scaler_low.pkl"), "rb") as file:
            scaler.append(joblib.load(file))
        with open(join("training_data", "multi_scaler_mid.pkl"), "rb") as file:
            scaler.append(joblib.load(file))
        with open(join("training_data", "multi_scaler_high.pkl"), "rb") as file:
            scaler.append(joblib.load(file))
    else:
        with open(join("training_data", "scaler.pkl"), "rb") as file:
            scaler.append(joblib.load(file))

    print("Starting timings prediction\n-----------------------------------------")

    for wav_name in wav_names:
        if not wav_name.endswith(".wav"):
            print(wav_name, "is not a wav file! Skipping...")
            continue
        if "pred_timings_" + wav_name[:-4] + ".txt" in existing_pred_timings and not overwrite:
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

        if model_type == 0:
            log_mel_re = featureReshape(log_mel, multi, 7)
            if not multi:
                log_mel_re = np.expand_dims(log_mel_re, axis=1)
            pdf = model.predict(log_mel_re)
        else:
            if model_type == 1:
                log_mel_pca = pca.transform(log_mel)
            else:
                log_mel_pca = pca.transform(log_mel.reshape(log_mel.shape[0], log_mel.shape[1] * log_mel.shape[2]))
            pdf = model.predict(xgb.DMatrix(log_mel_pca))

        pdf = np.squeeze(pdf)
        pdf = smooth_obs(pdf)

        timings = boundary_decoding(obs_i=pdf,
                                    threshold=0.5,
                                    hopsize_t=0.01,
                                    OnsetPeakPickingProcessor=OnsetPeakPickingProcessor)

        with open(join(out_path, "pred_timings_" + wav_name[:-4] + ".txt"), "w") as timings_file:
            for timing in timings:
                timings_file.write(str(timing / 100) + "\n")


if __name__ == '__main__':
    main()