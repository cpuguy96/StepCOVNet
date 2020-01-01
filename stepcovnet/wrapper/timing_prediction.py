import multiprocessing
import os
import time
from functools import partial
from os.path import join

import joblib
import numpy as np
import psutil
from madmom.features.onsets import OnsetPeakPickingProcessor

from stepcovnet.common.audio_preprocessing import get_madmom_log_mels, get_madmom_librosa_features
from stepcovnet.common.utils import feature_reshape, pre_process, get_filenames_from_folder, get_filename
from stepcovnet.configuration.parameters import SAMPLE_RATE, HOPSIZE_T, THRESHOLDS


def smooth_obs(obs):
    """using moving average hanning window for smoothing"""
    hann = np.hanning(5)
    hann /= np.sum(hann)
    obs = np.convolve(hann, obs, mode='same')
    return obs


def boundary_decoding(obs_i, threshold):
    """decode boundary"""
    arg_pp = {'threshold': threshold,
              'smooth': 0,
              'fps': 1. / HOPSIZE_T,
              'pre_max': HOPSIZE_T,
              'post_max': HOPSIZE_T}

    peak_picking = OnsetPeakPickingProcessor(**arg_pp)
    i_boundary = peak_picking.process(obs_i)
    i_boundary = np.append(i_boundary, (len(obs_i) - 1) * HOPSIZE_T)
    i_boundary /= HOPSIZE_T
    return i_boundary


def get_file_scalers(scaler_path, multi):
    if scaler_path is not None:
        scaler = []
        if multi:
            with open(join(scaler_path, "multi_scaler_low.pkl"), "rb") as file:
                scaler.append(joblib.load(file))
            with open(join(scaler_path, "multi_scaler_mid.pkl"), "rb") as file:
                scaler.append(joblib.load(file))
            with open(join(scaler_path, "multi_scaler_high.pkl"), "rb") as file:
                scaler.append(joblib.load(file))
        else:
            with open(join(scaler_path, "scaler.pkl"), "rb") as file:
                scaler.append(joblib.load(file))
        return scaler
    else:
        return None


def get_model(model_path,
              model_type,
              pca_path):
    extra = False
    pca = None

    if model_type == 0:
        from tensorflow.keras.models import load_model

        custom_objects = {}

        model = load_model(join(model_path), custom_objects=custom_objects, compile=False)
        if model.layers[0].input_shape[0][1] != 1:
            multi = True
        else:
            multi = False
        try:
            # try to find second input which indicates extra features
            if model.get_layer('extra_input'):
                extra = True
            else:
                extra = False
        except ValueError:
            # if not, then there is no extra features
            extra = False
    else:
        import xgboost
        model = xgboost.Booster({'nthread': -1})
        model.load_model(join(model_path))
        pca = joblib.load(join(pca_path))
        if model_type == 1:
            multi = False
        else:
            multi = True
    return model, multi, extra, pca


def generate_features(input_path,
                      multi,
                      extra,
                      model_type,
                      scaler,
                      pca,
                      verbose,
                      wav_name):
    if not wav_name.endswith(".wav"):
        if verbose:
            print("%s is not a wav file! Skipping..." % wav_name)
        return None, None
    try:
        if verbose:
            print("Generating features for %s" % get_filename(wav_name, False))

        log_mel = get_madmom_log_mels(join(input_path, wav_name), SAMPLE_RATE, HOPSIZE_T, multi)

        if model_type == 0:
            log_mel = feature_reshape(log_mel, multi)
            if not multi:
                log_mel = np.expand_dims(log_mel, axis=1)
        else:
            import xgboost
            if model_type == 1:
                log_mel = pca.transform(log_mel)
            else:
                log_mel = pca.transform(log_mel.reshape(log_mel.shape[0], log_mel.shape[1] * log_mel.shape[2]))

        log_mel, _ = pre_process(log_mel, multi, scalers=scaler)

        if extra:
            if verbose:
                print("Generating extra features for %s" % get_filename(wav_name, False))
            extra_features = get_madmom_librosa_features(join(input_path, wav_name), SAMPLE_RATE, HOPSIZE_T,
                                                         len(log_mel))
        else:
            extra_features = None

        return [log_mel, extra_features], wav_name
    except Exception as ex:
        if verbose:
            print("Error generating timings for %s: %r" % (wav_name, ex))
        return None, None


def generate_timings(model,
                     model_type,
                     verbose,
                     features_and_wav_names):
    pdfs = []
    if model_type == 0:
        for feature, wav_name in features_and_wav_names:
            if verbose:
                print("Generating timings for %s" % wav_name)
            pdfs.append(model.predict(feature, batch_size=2048))
    else:
        import xgboost
        for feature, wav_name in features_and_wav_names:
            if verbose:
                print("Generating timings for %s" % wav_name)
            pdfs.append(model.predict(xgboost.DMatrix(feature)))

    timings = [boundary_decoding(obs_i=smooth_obs(np.squeeze(pdf)), threshold=THRESHOLDS['expert'])
               for pdf in pdfs]
    return timings


def write_predictions(output_path,
                      timing_and_wav_name):
    timings = timing_and_wav_name[0]
    wav_name = timing_and_wav_name[1]
    with open(join(output_path, get_filename(wav_name, False) + ".timings"), "w") as timings_file:
        for timing in timings:
            timings_file.write(str(timing / 100) + "\n")


def run_process(input_path,
                output_path,
                model,
                model_type,
                multi,
                scaler,
                extra,
                pca,
                verbose):
    if os.path.isfile(input_path):
        features_and_wav_name = generate_features(os.path.dirname(input_path), multi, extra, model_type, scaler, pca,
                                                  verbose, get_filename(input_path))
        if features_and_wav_name[0] is None:
            return
        timing = generate_timings(model, model_type, verbose, [features_and_wav_name])
        write_predictions(output_path, (timing, features_and_wav_name[1]))
    else:
        wav_names = get_filenames_from_folder(input_path)
        func = partial(generate_features, input_path, multi, extra, model_type, scaler, pca, verbose)
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            features_and_wav_names = pool.map_async(func, wav_names).get()
        features, used_wav_names = [], []
        for feature, wav_name in features_and_wav_names:
            if feature is not None:
                features.append(feature)
                used_wav_names.append(get_filename(wav_name))
        timings = generate_timings(model, model_type, verbose, zip(features, used_wav_names))
        timings_and_wav_names = [(timing, wav_name) for timing, wav_name in zip(timings, used_wav_names)]
        func = partial(write_predictions, output_path)
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            pool.map_async(func, timings_and_wav_names).get()


def timing_prediction(input_path,
                      output_path,
                      model_path,
                      scaler_path=None,
                      pca_path=None,
                      model_type=0,
                      verbose_int=0):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError('%s is not a valid verbose input. Choose 0 for none or 1 for full' % verbose_int)
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(input_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(model_path):
        raise FileNotFoundError('Model %s is not found' % model_path)

    if model_type not in [0, 1, 2]:
        raise ValueError('Model type %s is not a valid model' % model_type)

    if model_type in [1, 2] and not os.path.isfile(pca_path):
        raise FileNotFoundError('PCA %s is not found' % pca_path)

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        if verbose:
            print("Starting timings prediction\n-----------------------------------------")
        model, multi, extra, pca = get_model(model_path, model_type, pca_path)
        scaler = get_file_scalers(scaler_path, multi)
        run_process(input_path, output_path, model, model_type, multi, scaler, extra, pca, verbose)
    else:
        raise FileNotFoundError('Wav file(s) path %s not found' % input_path)
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate arrow timings from .wav files.")
    parser.add_argument("-i" "--input",
                        type=str,
                        help="input wav path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output txt path")
    parser.add_argument("--model",
                        type=str,
                        help="trained model path")
    parser.add_argument("--scaler",
                        type=str,
                        help="scaler path")
    parser.add_argument("--pca",
                        type=str,
                        help="trained pca path")
    parser.add_argument("-mt" "--model_type",
                        type=int,
                        default=0,
                        help="type of model: 0 - CNN; 1 - XGB; 2 - multi XGB")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        help="verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    timing_prediction(args.input,
                      args.output,
                      args.model,
                      args.scaler,
                      args.pca,
                      args.model_type,
                      args.verbose)
