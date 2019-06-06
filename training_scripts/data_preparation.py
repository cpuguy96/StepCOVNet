import pickle
from sklearn.utils import compute_class_weight

def load_data(filename_labels_train_validation_set,
              filename_sample_weights,
              filename_scaler):

    # load training and validation data

    with open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = pickle.load(f)

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = pickle.load(f)

    with open(filename_scaler, 'rb') as f:
        scaler = pickle.load(f)

    # this is the filename indices
    indices_train_validation = range(len(Y_train_validation))

    class_weights = compute_class_weight('balanced', [0, 1], Y_train_validation)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    return indices_train_validation, Y_train_validation, sample_weights, class_weights, scaler
