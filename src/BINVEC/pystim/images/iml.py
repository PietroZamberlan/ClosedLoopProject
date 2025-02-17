import array
import numpy as np
import os

from .base import Image
from .png import PNGImage


class IMLImage(Image):

    @classmethod
    def load(cls, path, dtype, width, height):

        assert os.path.isfile(path), path

        with open(path, mode='rb') as handle:
            data_bytes = handle.read()
        data = array.array('H', data_bytes)
        data.byteswap()
        data = np.array(data, dtype=dtype)
        try:
            data = data.reshape(height, width)
        except ValueError as error:
            print(path)
            raise error

        iml = cls(data)

        return iml

    def __init__(self, data):

        super().__init__()

        self._data = data

    @property
    def data(self):

        return self._data

    @property
    def dtype(self):

        return str(self._data.dtype)

    @property
    def shape(self):

        return self._data.shape

    @property
    def width(self):

        return self._data.shape[1]

    @property
    def height(self):

        return self._data.shape[0]

    @property
    def min(self):

        return np.min(self._data)

    @property
    def max(self):

        return np.max(self._data)

    @property
    def mean(self):

        return np.mean(self._data)

    @property
    def std(self):

        return np.std(self._data)

    # def get_data(self, dtype='uint8'):
    #
    #     if dtype is None:
    #         data = self._data
    #     elif str(dtype) == 'uint8':
    #         data = self._data.astype('float')
    #         data[data < float(self._inf)] = float(self._inf)
    #         data[float(self._sup) < data] = float(self._sup)
    #         data = data - float(self._inf)
    #         data = data / float(self._sup - self._inf + 1)  # TODO check +1.
    #         v_min = np.iinfo(dtype).min
    #         v_max = np.iinfo(dtype).max
    #         data = data * float(v_max - v_min + 1)
    #         data = data + float(v_min)
    #         data = data.astype(dtype)
    #     else:
    #         raise ValueError("unknown dtype value: {}".format(dtype))
    #
    #     return data

    # def get_normalized_data(self):
    #
    #     data = self._data.astype('float')
    #     data[data < float(self._inf)] = float(self._inf)
    #     data[float(self._sup) < data] = float(self._sup)
    #     data = data - float(self._inf)
    #     data = data / float(self._inf - self._sup)
    #
    #     return data

    # def to_png(self, dtype='uint8'):
    #
    #     data = self.get_data(dtype=dtype)
    #     image = PNGImage(data)
    #
    #     return image

    def save(self, path):

        raise NotImplementedError


load = IMLImage.load
