from os import listdir
from os.path import join, isfile, splitext, basename


def get_filenames_from_folder(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_filename(file_path, with_ext=True):
    if with_ext:
        return basename(file_path)
    else:
        return splitext(basename(file_path))[0]
