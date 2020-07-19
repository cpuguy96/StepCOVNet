import os

import h5py

from stepcovnet.dataset.ModelDataset import ModelDataset


class DistributedModelDataset(ModelDataset):
    def __init__(self, *args, **kwargs):
        super(DistributedModelDataset, self).__init__(*args, **kwargs)
        self.sub_datasets = []
        self.virtual_dataset = None

    def __enter__(self):
        if self.virtual_dataset is None:
            self.virtual_dataset = h5py.File(self.dataset_path, self.mode, libver='latest')
        return self

    def close(self):
        self.virtual_dataset.close()

    def dump(self, *args, **kwargs):
        file_name = kwargs["file_name"]
        self.sub_datasets.append(file_name)
        sub_dataset_path = self.format_sub_dataset_name(file_name)
        if os.path.isfile(sub_dataset_path):
            os.remove(sub_dataset_path)
        self.h5py_file = h5py.File(sub_dataset_path, self.mode)
        super(DistributedModelDataset, self).dump(*args, **kwargs)
        self.build_virtual_dataset()

    def format_sub_dataset_name(self, file_name):
        return self.append_file_type('%s_%s' % (self.dataset_name, file_name))

    def build_virtual_dataset(self):
        if not self.sub_datasets:
            raise ValueError('Cannot build virtual dataset until data is dumped')
        if self.virtual_dataset is None:
            self.virtual_dataset = h5py.File(self.dataset_path, self.mode, libver='latest')
        for dataset_name in self.dataset_names:
            if dataset_name in self.difficulty_dataset_names:
                dataset_name = self.append_difficulty(dataset_name, self.difficulty)
            virtual_sources, virtual_source_shape, virtual_dtype = self.build_virtual_sources(
                dataset_name)
            virtual_layout = self.build_virtual_layout(virtual_sources, virtual_source_shape, virtual_dtype)
            saved_attributes = self.save_attributes(self.virtual_dataset, dataset_name)
            if dataset_name in self.virtual_dataset:
                del self.virtual_dataset[dataset_name]
            self.virtual_dataset.create_virtual_dataset(dataset_name, virtual_layout, fillvalue=-1)
            self.set_dataset_attrs(self.virtual_dataset, dataset_name, saved_attributes)
            if dataset_name is "features":
                self.update_dataset_attrs(self.virtual_dataset, dataset_name,
                                          self.h5py_file[dataset_name].attrs["file_names"])
            else:
                self.update_dataset_attrs(self.virtual_dataset, dataset_name,
                                          self.h5py_file[dataset_name][:])

    def build_virtual_sources(self, dataset_name):
        sources = []
        dtype = None
        source_shape = None
        for parent_dataset_name in self.sub_datasets:
            with h5py.File(self.format_sub_dataset_name(parent_dataset_name), 'r') as parent_dataset:
                vsource = h5py.VirtualSource(parent_dataset[dataset_name])
                if source_shape is None:
                    source_shape = vsource.shape
                else:
                    source_shape = (source_shape[0] + vsource.shape[0],)
                    if len(vsource.shape) > 1:
                        source_shape = source_shape + vsource.shape[1:]
                # should be fine to keep overriding this
                dtype = parent_dataset[dataset_name].dtype
                sources.append(vsource)

        return sources, source_shape, dtype

    @staticmethod
    def build_virtual_layout(sources, source_shapes, dtype):
        virtual_layout = h5py.VirtualLayout(shape=source_shapes, dtype=dtype)
        offset = 0
        for source in sources:
            length = source.shape[0]
            virtual_layout[offset: offset + length] = source
            offset += length

        return virtual_layout

    @property
    def file_names(self):
        return [file_name.decode("ascii") for file_name in self.virtual_dataset["features"].attrs["file_names"]]

    @property
    def num_samples(self):
        return self.virtual_dataset[self.append_difficulty("labels", self.difficulty)].attrs["num_samples"]

    @property
    def num_valid_samples(self):
        return self.virtual_dataset[self.append_difficulty("labels", self.difficulty)].attrs["num_valid_samples"]

    @property
    def pos_samples(self):
        return self.virtual_dataset[self.append_difficulty("labels", self.difficulty)].attrs["pos_samples"]

    @property
    def neg_samples(self):
        return self.virtual_dataset[self.append_difficulty("labels", self.difficulty)].attrs["neg_samples"]

    @property
    def labels(self):
        return self.virtual_dataset[self.append_difficulty("labels", self.difficulty)]

    @property
    def sample_weights(self):
        return self.virtual_dataset[self.append_difficulty("sample_weights", self.difficulty)]

    @property
    def arrows(self):
        return self.virtual_dataset[self.append_difficulty("arrows", self.difficulty)]

    @property
    def encoded_arrows(self):
        return self.virtual_dataset[self.append_difficulty("encoded_arrows", self.difficulty)]

    @property
    def features(self):
        return self.virtual_dataset["features"]
