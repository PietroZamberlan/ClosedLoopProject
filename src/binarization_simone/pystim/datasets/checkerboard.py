import numpy as np
import os

from pystim.images.png import create as create_png_image
from pystim.images.png import load as load_png_image

from .base import get_path as get_base_path


_NAME = 'checkerboard'

_WIDTH = 56  # px
_HEIGHT = 56  # px

_ANGULAR_RESOLUTION = 15.0 / 60.0  # °/px


def get_reference_path():

    path = os.path.join(get_base_path(), _NAME)

    return path


def get_path(image_nb):

    ref_path = get_reference_path()
    filename = 'checkerboard_{:04d}.png'.format(image_nb)
    path = os.path.join(ref_path, filename)

    return path


def get_image_nbs(generated_only=False):

    image_nbs = np.arange(0, 2)  # TODO correct?
    if generated_only:
        image_nbs = np.array([
            image_nb
            for image_nb in image_nbs
            if os.path.isfile(get_path(image_nb))
        ])

    return image_nbs


def fetch(image_nbs=None, force=False):

    ref_path = get_reference_path()
    if not os.path.isdir(ref_path):
        os.makedirs(ref_path)

    if image_nbs is None:
        image_nbs = get_image_nbs()
    else:
        # TODO remove the 2 following lines?
        # for image_nb in image_nbs:
        #     assert image_nb in get_image_nbs()
        pass

    for image_nb in image_nbs:
        path = get_path(image_nb)
        if os.path.isfile(path) and not force:
            continue
        np.random.seed(seed=image_nb)
        a = np.array([0.0, 1.0])
        shape = (_HEIGHT, _WIDTH)
        pattern = np.random.choice(a=a, size=shape)
        data = np.array(254.0 * pattern, dtype=np.uint8)
        image = create_png_image(data)
        image.save(path)

    return


def load(image_nb):

    path = get_path(image_nb)
    image = load_png_image(path)

    return image


def load_data(image_nb, with_borders=0.5):

    image = load(image_nb)
    data = image.data
    data = np.flipud(data)
    data = np.transpose(data)
    data = data.astype(np.float)
    data = data / 254.0  # 0 -> 0.0 and 254 -> 1.0 such that 127 -> 0.5
    if with_borders:
        width, height = data.shape
        data_ = np.empty((1 + width + 1, 1 + height + 1))
        data_[1:-1, 1:-1] = data
        data_[0, :] = with_borders
        data_[-1, :] = with_borders
        data_[:, 0] = with_borders
        data_[:, -1] = with_borders
        data = data_

    return data


def get_horizontal_angles(with_borders=False):

    if not with_borders:
        x = np.arange(0, _WIDTH)
    else:
        x = np.arange(0, 1 + _WIDTH + 1)
    x = x.astype(np.float)
    x = x - np.mean(x)
    a_x = _ANGULAR_RESOLUTION * x

    return a_x


def get_vertical_angles(with_borders=False):

    if not with_borders:
        y = np.arange(0, _HEIGHT)
    else:
        y = np.arange(0, 1 + _HEIGHT + 1)
    y = y.astype(np.float)
    y = y - np.mean(y)
    a_y = _ANGULAR_RESOLUTION * y

    return a_y
