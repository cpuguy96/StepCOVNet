import joblib
from sklearn.utils import compute_class_weight


def load_data(filename_features,
              filename_extra_features,
              filename_labels,
              filename_sample_weights,
              filename_scalers,
              filename_pretrained_model):
    # load training and validation data
    with open(filename_features, 'rb') as f:
        features = joblib.load(f)

    if filename_extra_features is not None:
        with open(filename_extra_features, 'rb') as f:
            extra_features = joblib.load(f)
    else:
        extra_features = None

    with open(filename_labels, 'rb') as f:
        labels = joblib.load(f)

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = joblib.load(f)

    scaler = []
    for filename_scaler in filename_scalers:
        with open(filename_scaler, 'rb') as f:
            scaler.append(joblib.load(f))

    if filename_pretrained_model is not None:
        from tensorflow.keras.models import load_model
        pretrained_model = load_model(filename_pretrained_model, compile=False)
    else:
        pretrained_model = None

    class_weights = compute_class_weight('balanced', [0, 1], labels)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    return features, extra_features, labels, sample_weights, class_weights, scaler, pretrained_model
