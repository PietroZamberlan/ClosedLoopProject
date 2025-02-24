import numpy as np
import PIL.Image

from .base import Image


class PNGImage(Image):

    @classmethod
    def load(cls, path):

        image = PIL.Image.open(path, mode='r')
        data = np.array(image)
        assert np.issubdtype(data.dtype, np.uint8), data.dtype
        assert data.ndim == 2
        image = cls(data)

        return image

    def __init__(self, data):

        super().__init__()

        self._data = data

    @property
    def data(self):

        return self._data

    @property
    def min(self):

        return np.min(self._data)

    @property
    def max(self):

        return np.max(self._data)

    @property
    def dtype(self):

        return str(self._data.dtype)

    @property
    def inf(self):

        return np.iinfo(self.dtype).min

    @property
    def sup(self):

        return np.iinfo(self.dtype).max

    def save(self, path):

        if self.dtype == 'uint8':
            mode = 'L'  # i.e. 8-bit pixels, black and white
            image = PIL.Image.fromarray(self._data, mode=mode)
            image.save(path)
        else:
            # image = PIL.Image.fromarray(self._data)
            # image.save(path)
            raise ValueError("unsupported dtype value: {}".format(self.dtype))

        return


create = PNGImage
load = PNGImage.load
