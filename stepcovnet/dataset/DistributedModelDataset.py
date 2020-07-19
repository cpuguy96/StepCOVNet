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

    def dump(self, file_name, *args, **kwargs):
        self.sub_datasets.append(file_name)
        parent_dataset_path = self.format_sub_dataset_name(len(self.sub_datasets) - 1)
        if os.path.isfile(parent_dataset_path):
            os.remove(parent_dataset_path)
        self.h5py_file = h5py.File(parent_dataset_path, self.mode)
        super(DistributedModelDataset, self).dump(*args, **kwargs)
        self.build_virtual_dataset()

    def format_sub_dataset_name(self, index):
        return self.append_file_type(self.dataset_name + '_%s' % self.sub_datasets[index])

    def build_virtual_dataset(self):
        if not self.sub_datasets:
            raise ValueError('Cannot build virtual dataset until data is dumped')
        if self.virtual_dataset is None:
            self.virtual_dataset = h5py.File(self.dataset_path, self.mode, libver='latest')
        for dataset_name in self.dataset_names:
            if dataset_name is not "features":
                difficulty_dataset_name = self.append_difficulty(dataset_name, self.difficulty)
            else:
                difficulty_dataset_name = dataset_name
            virtual_sources, virtual_source_shape, virtual_dtype = self.build_virtual_sources(
                difficulty_dataset_name)
            virtual_layout = self.build_virtual_layout(virtual_sources, virtual_source_shape, virtual_dtype)
            saved_attributes = self.save_attributes(self.virtual_dataset, difficulty_dataset_name)
            if difficulty_dataset_name in self.virtual_dataset:
                del self.virtual_dataset[difficulty_dataset_name]
            self.virtual_dataset.create_virtual_dataset(difficulty_dataset_name, virtual_layout, fillvalue=-1)
            self.set_dataset_attrs(self.virtual_dataset, difficulty_dataset_name, saved_attributes)
            self.update_dataset_attrs(self.virtual_dataset, difficulty_dataset_name,
                                      self.h5py_file[difficulty_dataset_name][:])

    def build_virtual_sources(self, dataset_name):
        sources = []
        dtype = None
        source_shape = None
        for i, parent_dataset_name in enumerate(self.sub_datasets):
            with h5py.File(self.format_sub_dataset_name(index=i), 'r') as parent_dataset:
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
