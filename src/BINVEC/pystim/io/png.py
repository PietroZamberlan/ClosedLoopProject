import matplotlib.pyplot as plt
import numpy as np

from PIL.Image import open as open_image


def load(input_path):

    image = open_image(input_path)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    width, height = image.size
    data = data.reshape(height, width)

    png = PNG(data)

    return png


class PNG:

    def __init__(self, data):

        self._data = data

    def plot_histogram(self):

        fig, ax = plt.subplots()

        x = self._data.astype(np.float).flatten()
        bin_min = 0
        bin_max = 2 ** 8
        nb_bins = bin_max - bin_min
        bins_range = (bin_min, bin_max)

        if not np.max(x) <= bin_max:
            print("not np.max(x) ({}) <= bin_max ({})".format(np.max(x), bin_max))

        ax.hist(x, bins=nb_bins, range=bins_range)
        ax.set_xlim(bin_min, bin_max)
        ax.set_xlabel("grey level")
        ax.set_ylabel("number of pixels")
        ax.set_title("Histogram (8 bit)")

        plt.show()

        return
