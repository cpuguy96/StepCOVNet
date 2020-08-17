from stepcovnet.dataset.AbstractModelDataset import AbstractModelDataset


class ModelDataset(AbstractModelDataset):
    def __init__(self, *args, **kwargs):
        super(ModelDataset, self).__init__(*args, **kwargs)

    @property
    def file_names(self):
        return [file_name.decode('ascii') for file_name in self.h5py_file["file_names"]]
