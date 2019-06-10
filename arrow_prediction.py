from os import listdir
from os.path import isfile, join
import numpy as np
from nltk.util import ngrams

from keras.models import load_model


def get_file_names(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


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

'''
def get_arrows(timings, model):
    pred_notes = []

    lookback = model.layers[0].input_shape[1]
    tokens = np.expand_dims(create_tokens(timings), axis=1)
    notes_ngram = np.expand_dims(get_notes_ngram(np.zeros((1, 4)), lookback)[-1], axis=0)

    for token in tokens:
        pred_arrow = np.argmax(model.predict([notes_ngram, token])) + 1
        pred_notes.append(pred_arrow)
        binary_note = get_binary_rep([pred_arrow])
        notes_ngram = np.roll(notes_ngram, -1, axis=0)
        notes_ngram[0][-1] = binary_note

    return pred_notes
'''

def get_arrows(timings, model):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(categories='auto', sparse=False).fit(np.asarray(get_all_note_combs()).reshape(-1, 1))

    pred_notes = []
    lookback = model.layers[0].input_shape[1]
    tokens = np.expand_dims(create_tokens(timings), axis=1)
    notes_ngram = np.expand_dims(get_notes_ngram(np.zeros((1, 16)), lookback)[-1], axis=0)
    for i, token in enumerate(tokens):
        pred_arrow = np.argmax(model.predict([notes_ngram, token]))
        binary_rep = encoder.categories_[0][pred_arrow]
        pred_notes.append(binary_rep)
        binary_note = get_extended_binary_rep([binary_rep])
        notes_ngram = np.roll(notes_ngram, -1, axis=0)
        notes_ngram[0][-1] = binary_note

    return pred_notes

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
    args = parser.parse_args()

    timings_path = args.timing
    out_path = args.output
    model_path = args.model

    timings_names = get_file_names(timings_path)
    existing_pred_arrows = get_file_names(out_path)
    model = load_model(join(model_path))

    print("Starting arrows prediction\n-----------------------------------------")

    for timings_name in timings_names:
        song_name = timings_name[:-4]

        if song_name.startswith("pred_timings_"):
            song_name = song_name[13:]

        if "pred_arrows_" + song_name + ".txt" in existing_pred_arrows:
            print(timings_name[:-4] + " arrows already generated! Skipping...")
            continue

        with open(timings_path + timings_name, "r") as timings_file:
            timings = np.asarray([line.replace("\n", "") for line in timings_file.readlines()]).astype("float32")

        print("Generating timings for " + song_name)

        arrows = get_arrows(timings, model)

        with open(out_path + "pred_arrows_" + song_name + ".txt", "w") as arrows_file:
            for arrow in arrows:
                arrows_file.write(str(arrow) + "\n")
