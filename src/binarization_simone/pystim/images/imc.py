import numpy as np

from .base import Image


class IMCImage(Image):

    @classmethod
    def load(cls, path, dtype, width, height, **kwargs):

        raise NotImplementedError

    def __init__(self, data, inf=None, sup=None):

        super().__init__()

        dinfo = np.iinfo(data.dtype)

        self._min_value = max(inf, dinfo.min) if inf is not None else dinfo.min
        self._max_value = min(sup, dinfo.max) if sup is not None else dinfo.max

    def to_png(self):

        raise NotImplementedError

    def save(self, path):

        raise NotImplementedError


load = IMCImage.load
