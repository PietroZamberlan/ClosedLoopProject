import os


def get_path():

    path = '/tmp/pystim/datasets'

    if not os.path.isdir(path):
        os.makedirs(path)

    return path


class Bunch(dict):
    """Container object for datasets.

    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def __setattr__(self, key, value):

        self[key] = value

    def __dir__(self):

        return self.keys()

    def __getattr__(self, key):

        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
