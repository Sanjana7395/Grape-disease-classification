import os


def make_folder(path):
    """Check if the folder exists, if it doesn't exist create one in the given path.

    Args:
        path [str]: path where the folder needs to be created.

    """
    if not os.path.exists(os.path.join(path)):
        print('[INFO] Creating new folder...')

        os.makedirs(os.path.join(path))
