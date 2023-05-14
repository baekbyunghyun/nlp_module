import os


def get_file_name_with_extension(path=None):
    if path is None:
        raise Exception('Unknown file path')

    return os.path.basename(path)


def get_directory_path(path=None):
    if path is None:
        raise Exception('Unknown file path')

    return os.path.dirname(path)
