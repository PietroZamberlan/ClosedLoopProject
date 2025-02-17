import numpy as np
import os

from src.BINVEC.pystim.images.png import create as create_png_image
from src.BINVEC.pystim.images.png import load as load_png_image

from .base import get_path as get_base_path


_NAME = 'grey'

_WIDTH = 1920  # px
_HEIGHT = 1080  # px

_ANGULAR_RESOLUTION = 1.0 / 60.0  # °/px


def get_reference_path():

    path = os.path.join(get_base_path(), _NAME)

    return path


def get_path(image_nb):

    ref_path = get_reference_path()
    filename = 'grey_{:03d}.png'.format(image_nb)
    path = os.path.join(ref_path, filename)

    return path


def get_image_nbs(generated_only=False):

    dinfo = np.iinfo(np.uint8)
    image_nbs = np.arange(dinfo.min, dinfo.max + 1)
    if generated_only:
        image_nbs = np.array([
            image_nb
            for image_nb in image_nbs
            if os.path.isfile(get_path(image_nb))
        ])

    return image_nbs


def generate(image_nbs=None, verbose=False):

    ref_path = get_reference_path()
    if not os.path.isdir(ref_path):
        os.makedirs(ref_path)

    if image_nbs is None:
        image_nbs = get_image_nbs()
    else:
        for image_nb in image_nbs:
            assert image_nb in get_image_nbs()

    for image_nb in image_nbs:
        grey_level = image_nb
        shape = (_HEIGHT, _WIDTH)
        data = grey_level * np.ones(shape, dtype=np.uint8)
        image = create_png_image(data)
        path = get_path(image_nb)
        image.save(path)

    return


def load(image_nb):

    path = get_path(image_nb)
    image = load_png_image(path)

    return image


def load_data(image_nb):

    image = load(image_nb)
    data = image.data
    data = np.flipud(data)
    data = np.transpose(data)
    data = data.astype(np.float)

    return data


def get_horizontal_angles():

    x = np.arange(0, _WIDTH)
    x = x - np.mean(x)
    a_x = _ANGULAR_RESOLUTION * x

    return a_x


def get_vertical_angles():

    y = np.arange(0, _HEIGHT)
    y = y - np.mean(y)
    a_y = _ANGULAR_RESOLUTION * y

    return a_y
