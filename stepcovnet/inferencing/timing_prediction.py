import multiprocessing
import os
import time
from functools import partial
from os.path import join

import joblib
import numpy as np
import psutil
from madmom.features.onsets import OnsetPeakPickingProcessor

from stepcovnet.common.audio_preprocessing import get_madmom_log_mels
from stepcovnet.common.constants import HOPSIZE_T
from stepcovnet.common.constants import SAMPLE_RATE
from stepcovnet.common.constants import THRESHOLDS
from stepcovnet.common.utils import feature_reshape_up
from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder
from stepcovnet.common.utils import pre_process
from stepcovnet.common.utils import write_file


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
                scaler.extend(joblib.load(file))
        return scaler
    else:
        return None


def get_model(model_path):
    from tensorflow.keras.models import load_model

    custom_objects = {}

    model = load_model(join(model_path), custom_objects=custom_objects, compile=False)
    if model.layers[0].input_shape[0][1] != 1:
        multi = True
    else:
        multi = False
    return model, multi


def generate_features(input_path,
                      multi,
                      scaler,
                      verbose,
                      wav_name):
    if not wav_name.endswith(".wav"):
        if verbose:
            print("%s is not a wav file! Skipping..." % wav_name)
        return None, None
    try:
        if verbose:
            print("Generating features for %s" % get_filename(wav_name, False))

        # TODO: Fix this. Need to find a way to persist config
        log_mel = get_madmom_log_mels(join(input_path, wav_name), SAMPLE_RATE, HOPSIZE_T, multi)

        # TODO: Fix this.
        log_mel = feature_reshape_up(log_mel, )

        features = pre_process(log_mel, multi=multi, scalers=scaler)

        return features, wav_name
    except Exception as ex:
        if verbose:
            print("Error generating timings for %s: %r" % (wav_name, ex))
        return None, None


def generate_timings(model,
                     verbose,
                     features_and_wav_names):
    pdfs = []
    for feature, wav_name in features_and_wav_names:
        if verbose:
            print("Generating timings for %s" % wav_name)
        pdfs.append(model.predict(feature, batch_size=256))

    timings = [boundary_decoding(obs_i=smooth_obs(np.squeeze(pdf)), threshold=THRESHOLDS['expert'])
               for pdf in pdfs]
    return timings


def write_predictions(output_path,
                      timing_and_wav_name):
    timings = timing_and_wav_name[0]
    wav_name = timing_and_wav_name[1]
    output_file = join(output_path, get_filename(wav_name, False) + ".timings")
    output_timings = '\n'.join([str(timing / 100) for timing in timings])
    write_file(output_path=output_file, output_data=output_timings)


def run_process(input_path,
                output_path,
                model,
                multi,
                scaler,
                verbose):
    if os.path.isfile(input_path):
        features_and_wav_name = generate_features(os.path.dirname(input_path), multi, scaler, verbose,
                                                  get_filename(input_path))
        if features_and_wav_name[0] is None:
            return
        timing = generate_timings(model, verbose, [features_and_wav_name])
        write_predictions(output_path, (timing, features_and_wav_name[1]))
    else:
        wav_names = get_filenames_from_folder(input_path)
        func = partial(generate_features, input_path, multi, scaler, verbose)
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            features_and_wav_names = pool.map_async(func, wav_names).get()
        features, used_wav_names = [], []
        for feature, wav_name in features_and_wav_names:
            if feature is not None:
                features.append(feature)
                used_wav_names.append(get_filename(wav_name))
        timings = generate_timings(model, verbose, zip(features, used_wav_names))
        timings_and_wav_names = [(timing, wav_name) for timing, wav_name in zip(timings, used_wav_names)]
        func = partial(write_predictions, output_path)
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            pool.map_async(func, timings_and_wav_names).get()


def timing_prediction(input_path,
                      output_path,
                      model_path,
                      scaler_path=None,
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

    if os.path.isfile(input_path) or os.path.isdir(input_path):
        if verbose:
            print("Starting timings prediction\n-----------------------------------------")
        model, multi = get_model(model_path)
        scaler = get_file_scalers(scaler_path, multi)
        run_process(input_path, output_path, model, multi, scaler, verbose)
    else:
        raise FileNotFoundError('Wav file(s) path %s not found' % input_path)
    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate arrow timings from .wav files")
    parser.add_argument("-i" "--input",
                        type=str,
                        help="Input wav file/directory path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output txts path")
    parser.add_argument("--stepcovnet_model",
                        type=str,
                        help="Input trained stepcovnet_model path")
    parser.add_argument("--scaler",
                        type=str,
                        help="Input scalers path")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    timing_prediction(args.input,
                      args.output,
                      args.model,
                      args.scaler,
                      args.verbose)
