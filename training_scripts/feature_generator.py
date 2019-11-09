from training_scripts.data_preparation import featureReshape

import numpy as np


def generator(path_feature_data,
              indices,
              file_size,
              labels=None,
              sample_weights=None,
              multi_inputs=False,
              channel=1,
              scaler=None,
              path_extra_features=None,
              extra=False):

    indices_copy = np.array(indices[:], np.int64)

    if labels is not None:
        labels_copy = np.copy(labels)
    else:
        labels_copy = np.zeros((len(indices_copy), ))

    if sample_weights is not None:
        sample_weights_copy = np.copy(sample_weights)
    else:
        sample_weights_copy = np.ones((len(indices_copy), ))

    counter = 0

    with open(path_feature_data, 'rb') as f:
        f_used = np.load(f)['features']

    if extra:
        with open(path_extra_features, 'rb') as f:
            extra_features = np.load(f)['extra_features']

    if scaler:
        if multi_inputs:
            f_used[:, :, 0] = scaler[0].transform(np.asarray(f_used[:, :, 0]))
            f_used[:, :, 1] = scaler[1].transform(np.asarray(f_used[:, :, 1]))
            f_used[:, :, 2] = scaler[2].transform(np.asarray(f_used[:, :, 2]))
        else:
            f_used = scaler[0].transform(np.asarray(f_used))

    if multi_inputs:
        f_used = featureReshape(f_used, True, 7)
    else:
        f_used = featureReshape(f_used, False, 7)

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        batch_indices = indices_copy[idx_start:idx_end]
        index_sort = np.argsort(batch_indices)

        y_batch_tensor = labels_copy[idx_start:idx_end][index_sort]
        sample_weights_batch_tensor = sample_weights_copy[idx_start:idx_end][index_sort]

        if extra:
            extra_features_tensor = extra_features[idx_start:idx_end][index_sort]
        else:
            extra_features_tensor = None

        if channel == 1:
            X_batch_tensor = f_used[batch_indices[index_sort], :, :]
        else:
            X_batch_tensor = f_used[batch_indices[index_sort], :, :, :]
        if channel == 1:
            X_batch_tensor = np.squeeze(X_batch_tensor)
            X_batch_tensor = np.expand_dims(X_batch_tensor, axis=1)

        counter += 1

        if sample_weights is not None:
            if extra:
                yield [X_batch_tensor, extra_features_tensor], y_batch_tensor, sample_weights_batch_tensor
            else:
                yield X_batch_tensor, y_batch_tensor, sample_weights_batch_tensor
        else:
            if extra:
                yield [X_batch_tensor, extra_features_tensor], y_batch_tensor
            else:
                yield X_batch_tensor, y_batch_tensor
