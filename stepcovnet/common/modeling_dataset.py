import os

import h5py


class ModelDataset(object):
    def __init__(self, dataset_path, overwrite=False, mode='a'):
        self.dataset_path = dataset_path
        self.overwrite = overwrite
        self.mode = mode
        if self.overwrite and self.mode not in {"a", "r+", "w", "w+", "x", "w-"}:
            raise ValueError("Mode must be a, r+, w, or w+ while in overwrite mode!")
        if self.overwrite and os.path.isfile(self.dataset_path):
            os.remove(self.dataset_path)
        if not self.overwrite:
            self.mode = "r"
        self.h5py_file = h5py.File(self.dataset_path, self.mode)
        self.dataset_names = ["features", "labels", "sample_weights", "extra_features"]
        self.dataset_attr = {"num_samples", "pos_samples", "neg_samples"}

    def __getitem__(self, item):
        data = [self.features[item], self.labels[item], self.sample_weights[item]]
        try:
            data.append(self.extra_features[item])
        except Exception as ex:
            print("Dataset doesn't have extra features. Defaulting extra features to None.")
            data.append(None)

        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def dump(self, features, labels, sample_weights, extra_features):
        try:
            all_data = [features, labels, sample_weights, extra_features]
            for dataset_name, data in zip(self.dataset_names, all_data):
                if data is None:
                    continue
                if not self.h5py_file.get(dataset_name):
                    if len(data.shape) > 1:
                        data_shape = (None,) + data.shape[1:]
                    else:
                        data_shape = (None,)
                    self.h5py_file.create_dataset(dataset_name, data=data, chunks=True, compression="lzf",
                                                  maxshape=data_shape)
                    if dataset_name == "labels":
                        for dataset_attr in self.dataset_attr:
                            self.h5py_file["labels"].attrs[dataset_attr] = 0
                else:
                    self.h5py_file[dataset_name].resize((self.h5py_file[dataset_name].shape[0] + data.shape[0]), axis=0)
                    self.h5py_file[dataset_name][-data.shape[0]:] = data

            self.h5py_file["labels"].attrs["num_samples"] += len(labels)
            self.h5py_file["labels"].attrs["pos_samples"] += labels.sum()
            self.h5py_file["labels"].attrs["neg_samples"] += len(labels) - labels.sum()

            self.h5py_file.flush()
        except Exception as ex:
            self.close()
            raise ex

    def close(self):
        self.h5py_file.flush()
        self.h5py_file.close()

    @property
    def num_samples(self):
        return self.h5py_file["labels"].attrs["num_samples"]

    @property
    def pos_samples(self):
        return self.h5py_file["labels"].attrs["pos_samples"]

    @property
    def neg_samples(self):
        return self.h5py_file["labels"].attrs["neg_samples"]

    @property
    def labels(self):
        return self.h5py_file["labels"]

    @property
    def sample_weights(self):
        return self.h5py_file["sample_weights"]

    @property
    def extra_features(self):
        if self.h5py_file.get("extra_features"):
            return self.h5py_file["extra_features"]
        else:
            return None

    @property
    def features(self):
        return self.h5py_file["features"]
