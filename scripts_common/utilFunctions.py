from os import listdir
from os.path import join, isfile


def get_file_names(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]
