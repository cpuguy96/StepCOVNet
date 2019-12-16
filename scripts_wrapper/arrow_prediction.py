from scripts_common.utilFunctions import get_filenames_from_folder, get_filename

from os.path import join
from nltk.util import ngrams

import os
import time
import numpy as np


def __get_binary_rep(arrow_values):
    return ((np.asarray(arrow_values).astype(int)[:,None] & (1 << np.arange(4))) > 0).astype(int)


def __get_extended_binary_rep(arrow_combs):
    extended_binary_rep = []
    for i, arrow_comb in enumerate(arrow_combs):
        binary_rep = np.zeros((4, 4))
        for j, num in enumerate(list(arrow_comb)):
            binary_rep[int(num), j] = 1
        extended_binary_rep.append(binary_rep.ravel())
    return np.asarray(extended_binary_rep)


def __get_all_note_combs():
    all_note_combs = []

    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    all_note_combs.append(str(i) + str(j) + str(k) + str(l))

    all_note_combs = all_note_combs[1:]

    return all_note_combs


def __create_tokens(timings):
    timings = timings.astype("float32")
    tokens = np.zeros((timings.shape[0], 3))
    tokens[0][0] = 1 # set start token
    next_note_token = np.append(timings[1:] - timings[:-1], np.asarray([0]))
    prev_note_token = np.append(np.asarray([0]),  next_note_token[: -1])
    tokens[:, 1] = prev_note_token.reshape(1, -1)
    tokens[:, 2] = next_note_token.reshape(1, -1)
    return tokens.astype("float32")


def __get_notes_ngram(binary_notes, lookback):
    padding = np.zeros((lookback, binary_notes.shape[1]))
    data_w_padding = np.append(padding, binary_notes, axis=0)
    return np.asarray(list(ngrams(data_w_padding, lookback)))


def __get_arrows(timings, model, encoder):
    pred_notes = []
    lookback = model.layers[0].input_shape[0][1]
    classes = model.layers[-1].output_shape[1]
    tokens = np.expand_dims(np.expand_dims(__create_tokens(timings), axis=1), axis=1)
    notes_ngram = np.expand_dims(__get_notes_ngram(np.zeros((1, 16)), lookback)[-1], axis=0)
    for i, token in enumerate(tokens):
        pred = model.predict([notes_ngram, token])
        pred_arrow = np.random.choice(classes, 1, p=pred[0])[0]
        binary_rep = encoder.categories_[0][pred_arrow]
        pred_notes.append(binary_rep)
        binary_note = __get_extended_binary_rep([binary_rep])
        notes_ngram = np.roll(notes_ngram, -1, axis=0)
        notes_ngram[0][-1] = binary_note
    return pred_notes


def __generate_arrows(input_path, output_path, model, encoder, verbose, timing_name):
    song_name = get_filename(timing_name, False)
    with open(input_path + timing_name, "r") as timings_file:
        timings = np.asarray([line.replace("\n", "") for line in timings_file.readlines()]).astype("float32")

    if verbose:
        print("Generating arrows for " + song_name)

    arrows = __get_arrows(timings, model, encoder)

    with open(join(output_path, song_name + ".arrows"), "w") as arrows_file:
        for arrow in arrows:
            arrows_file.write(str(arrow) + "\n")


def __run_process(input_path, output_path, model, encoder, verbose):
    if os.path.isfile(input_path):
        __generate_arrows(os.path.dirname(input_path), output_path, model, encoder, verbose, get_filename(input_path))
    else:
        timings_names = get_filenames_from_folder(input_path)
        for timing_name in timings_names:
            __generate_arrows(input_path, output_path, model, encoder, verbose, timing_name)


def arrow_prediction(input_path,
                     output_path,
                     model_path,
                     verbose_int=0):
    start_time = time.time()
    if verbose_int not in [0, 1]:
        raise ValueError('%s is not a valid verbose input. Choose 0 for none or 1 for full' % verbose_int)
    verbose = True if verbose_int == 1 else False

    if not os.path.isdir(output_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(model_path):
        raise FileNotFoundError('Model %s is not found' % model_path)

    if os.path.isdir(input_path) or os.path.isfile(input_path):
        if verbose:
            print("Starting arrows prediction\n-----------------------------------------")
        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import OneHotEncoder
        model = load_model(join(model_path), compile=False)
        encoder = OneHotEncoder(categories='auto', sparse=False).fit(np.asarray(__get_all_note_combs()).reshape(-1, 1))
        __run_process(input_path, output_path, model, encoder, verbose)
    else:
        raise FileNotFoundError('Timing files path %s not found' % input_path)

    end_time = time.time()
    if verbose:
        print("Elapsed time was %g seconds\n" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate arrow types from .wav files.")
    parser.add_argument("-i", "--input",
                        type=str,
                        help="input timings path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output arrows path")
    parser.add_argument("--model",
                        type=str,
                        help="trained model path")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        default=0,
                        help="verbosity: 0 - none, 1 - full")
    args = parser.parse_args()

    arrow_prediction(args.input,
                     args.output,
                     args.model,
                     args.verbose)
