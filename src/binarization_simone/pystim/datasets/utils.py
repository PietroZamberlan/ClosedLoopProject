from . import grey
from . import van_hateren
from .grey import generate as generate_grey
from .grey import load as load_grey
from .van_hateren import fetch as fetch_van_hateren
from .van_hateren import load_image as load_van_hateren


def get(dataset_name):

    if dataset_name == 'grey':
        dataset = grey
    elif dataset_name == 'van Hateren':
        dataset = van_hateren
    else:
        raise ValueError("unknown dataset_name value: {}".format(dataset_name))

    return dataset


def fetch(dataset_name, image_nb, **kwargs):

    if dataset_name == 'grey':
        generate_grey(image_nbs=[image_nb], **kwargs)
    elif dataset_name == 'van Hateren':
        fetch_van_hateren(image_nbs=[image_nb], **kwargs)
    else:
        raise ValueError("unknown dataset_name value: {}".format(dataset_name))

    return


def load(dataset_name, image_nb, **kwargs):

    if dataset_name == 'grey':
        image = load_grey(image_nb)
    elif dataset_name == 'van Hateren':
        image = load_van_hateren(image_nb, **kwargs)
    else:
        raise ValueError("unknown dataset_name value: {}".format(dataset_name))

    return image
