import array
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def load(input_path):

    assert os.path.isfile(input_path), input_path

    dtype = np.uint16
    height = 1024
    width = 1536

    with open(input_path, mode='rb') as handle:
        data_bytes = handle.read()
    data = array.array('H', data_bytes)
    data.byteswap()
    data = np.array(data, dtype=dtype)
    data = data.reshape(height, width)

    iml = IML(data)

    return iml


class IML:

    def __init__(self, data):

        self._data = data

    @property
    def height(self):

        return self._data.shape[0]

    @property
    def width(self):

        return self._data.shape[1]

    def min(self):

        return np.min(self._data)

    def max(self):

        return np.max(self._data)

    def get_uint16(self):

        data = self._data

        return data

    def get_uint8(self):

        uint12_max = 2.0 ** 12
        uint8_max = 2.0 ** 8

        data = self._data.astype(np.float)
        data = data / uint12_max

        if not np.max(data) < 1.0:
            print("not np.max(data) ({}) <= 1.0".format(np.max(data)))
            data[data >= 1.0] = 1.0 - sys.float_info.epsilon

        data = data * uint8_max
        data = data.astype(np.uint8)

        return data

    def plot_histograms(self):

        fig, axes = plt.subplots(nrows=2)

        x = self.get_uint16().astype(np.float).flatten()
        bin_min = 0
        bin_max = 2 ** 12
        nb_bins = bin_max - bin_min
        bins_range = (bin_min, bin_max)

        if not np.max(x) <= bin_max:
            print("not np.max(x) ({}) <= bin_max ({})".format(np.max(x), bin_max))

        ax = axes[0]
        ax.hist(x, bins=nb_bins, range=bins_range)
        ax.set_xlim(bin_min, bin_max)
        ax.set_xlabel("grey level")
        ax.set_ylabel("number of pixels")
        ax.set_title("Histogram (12 bit)")

        x = self.get_uint8().astype(np.float).flatten()
        bin_min = 0
        bin_max = 2 ** 8
        nb_bins = bin_max - bin_min
        bins_range = (bin_min, bin_max)

        ax = axes[1]
        ax.hist(x, bins=nb_bins, range=bins_range)
        ax.set_xlim(bin_min, bin_max)
        ax.set_xlabel("grey level")
        ax.set_ylabel("number of pixels")
        ax.set_title("Histogram (8 bit)")

        fig.tight_layout()

        plt.show()

        return
