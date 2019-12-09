from scripts_common.utilFunctions import get_file_names

from os.path import join
from nltk.util import ngrams
from tensorflow.keras.models import load_model

import os
import numpy as np


def get_binary_rep(arrow_values):
    return ((np.asarray(arrow_values).astype(int)[:,None] & (1 << np.arange(4))) > 0).astype(int)


def get_extended_binary_rep(arrow_combs):
    extended_binary_rep = []
    for i, arrow_comb in enumerate(arrow_combs):
        binary_rep = np.zeros((4, 4))
        for j, num in enumerate(list(arrow_comb)):
            binary_rep[int(num), j] = 1
        extended_binary_rep.append(binary_rep.ravel())
    return np.asarray(extended_binary_rep)


def get_all_note_combs():
    all_note_combs = []

    for i in range(0, 4):
        for j in range(0, 4):
            for k in range(0, 4):
                for l in range(0, 4):
                    all_note_combs.append(str(i) + str(j) + str(k) + str(l))

    all_note_combs = all_note_combs[1:]

    return all_note_combs


def create_tokens(timings):
    timings = timings.astype("float32")
    tokens = np.zeros((timings.shape[0], 3))
    tokens[0][0] = 1 # set start token
    next_note_token = np.append(timings[1:] - timings[:-1], np.asarray([0]))
    prev_note_token = np.append(np.asarray([0]),  next_note_token[: -1])
    tokens[:, 1] = prev_note_token.reshape(1, -1)
    tokens[:, 2] = next_note_token.reshape(1, -1)
    return tokens.astype("float32")


def get_notes_ngram(binary_notes, lookback):
    padding = np.zeros((lookback, binary_notes.shape[1]))
    data_w_padding = np.append(padding, binary_notes, axis=0)
    return np.asarray(list(ngrams(data_w_padding, lookback)))


def get_arrows(timings, model, encoder):
    pred_notes = []
    lookback = model.layers[0].input_shape[0][1]
    classes = model.layers[-1].output_shape[1]
    tokens = np.expand_dims(np.expand_dims(create_tokens(timings), axis=1), axis=1)
    notes_ngram = np.expand_dims(get_notes_ngram(np.zeros((1, 16)), lookback)[-1], axis=0)
    for i, token in enumerate(tokens):
        pred = model.predict([notes_ngram, token])
        pred_arrow = np.random.choice(classes, 1, p=pred[0])[0]
        binary_rep = encoder.categories_[0][pred_arrow]
        pred_notes.append(binary_rep)
        binary_note = get_extended_binary_rep([binary_rep])
        notes_ngram = np.roll(notes_ngram, -1, axis=0)
        notes_ngram[0][-1] = binary_note
    return pred_notes


def arrow_prediction(timings_path,
                     out_path,
                     model_path,
                     overwrite_int):
    if not os.path.isdir(timings_path):
        raise NotADirectoryError('Timing files path %s not found' % timings_path)

    if not os.path.isdir(out_path):
        print('Output path not found. Creating directory...')
        os.makedirs(out_path, exist_ok=True)

    if not os.path.isfile(model_path):
        raise FileNotFoundError('Model %s is not found' % model_path)

    if overwrite_int == 1:
        overwrite = True
    else:
        overwrite = False

    timings_names = get_file_names(timings_path)
    existing_pred_arrows = get_file_names(out_path)
    model = load_model(join(model_path), compile=False)

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(categories='auto', sparse=False).fit(np.asarray(get_all_note_combs()).reshape(-1, 1))

    print("Starting arrows prediction\n-----------------------------------------")

    for timings_name in timings_names:
        if not timings_name.endswith(".txt"):
            print(timings_name, "is not a timings file! Skipping...")
            continue

        song_name = timings_name[:-4]

        if song_name.startswith("pred_timings_"):
            song_name = song_name[13:]

        if "pred_arrows_" + song_name + ".txt" in existing_pred_arrows and not overwrite:
            print(timings_name[:-4] + " arrows already generated! Skipping...")
            continue

        with open(timings_path + timings_name, "r") as timings_file:
            timings = np.asarray([line.replace("\n", "") for line in timings_file.readlines()]).astype("float32")

        print("Generating arrows for " + song_name)

        arrows = get_arrows(timings, model, encoder)

        with open(join(out_path, "pred_arrows_" + song_name + ".txt"), "w") as arrows_file:
            for arrow in arrows:
                arrows_file.write(str(arrow) + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate arrow types from .wav files.")
    parser.add_argument("--timing",
                        type=str,
                        help="input timings path")
    parser.add_argument("--output",
                        type=str,
                        help="output arrows path")
    parser.add_argument("--model",
                        type=str,
                        help="trained model path")
    parser.add_argument("--overwrite",
                        type=int,
                        default=0,
                        help="overwrite already created files")
    args = parser.parse_args()

    arrow_prediction(args.timing,
                     args.output,
                     args.model,
                     args.overwrite)
