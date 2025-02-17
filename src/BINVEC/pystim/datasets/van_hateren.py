import json
import numpy as np
import os
import pandas as pd
import re
import tqdm
import urllib.request
import urllib.error
import warnings

from .base import Bunch
from .base import get_path as get_base_path
from ..images.iml import load as load_iml_image
from ..images.imc import load as load_imc_image
from ..images.png import create as create_png_image


min_image_nb = 1
max_image_nb = 4212
all_image_nbs = list(range(min_image_nb, max_image_nb + 1))
missing_image_nbs = [
    2829, 2831, 2835, 2837, 2839, 2846, 2859, 2860, 2863, 2864, 2865, 2872,
    2873, 2875, 2876, 2877, 2879, 2880, 2882, 2885, 2886, 2888, 2890, 2891,
    2892, 2893, 2922, 2924, 2925, 2927, 2936, 2938, 2939, 2941, 2963, 2964,
    2965, 2969, 2971, 2974, 2975, 2977, 2984, 3000, 3001,
]

DTYPE = np.uint16
_HEIGHT = 1024  # px
_WIDTH = 1536  # px

_ANGULAR_RESOLUTION = 1.0 / 60.0  # °/px

# TODO remove the following lines (deprecated)?
# sup_value = 6282
# sup_value = 12564  # i.e. 2*6282=12564, image 980
# sup_value = 25119  # i.e. 2*2*6282=25128, image 1910
# # uint12: 4095, uint13: 8191, uint14: 16383, uint15: 32767, uint16: 65535

# MAX_LUMINANCE = 8669.16  # cd/m²
MAX_LUMINANCE = 70302.4  # cd/m²

SATURATION_THRESHOLD_ISO200 = 6266  # inclusive for saturation
SATURATION_THRESHOLD_ISO400 = 12551  # inclusive for saturation
SATURATION_THRESHOLD_ISO800 = 25102  # inclusive for saturation

SATURATION_VALUES = {
    # iso_value: saturation_value (inclusive for saturation)
    200: 6266,  # inclusive for saturation
    400: 12551,  # inclusive for saturation
    800: 25102,  # inclusive for saturation
}

SATURATION_THRESHOLDS_DICT = {
    # (iso, aperture, shutter): saturation_value,
    (200, 4.0, 125): 6465,  # TODO correct?
    (200, 4.0, 250): 6440,  # TODO correct?
    (200, 4.0, 500): 6495,
    (200, 4.0, 1000): 6493,  # TODO double check.
    (200, 4.0, 2000): 6494,  # TODO correct?
    # (200, 5.6, 2): SATURATION_THRESHOLD_ISO200,  # missing images
    (200, 5.6, 4): 6275,
    (200, 5.6, 8): 6273,
    (200, 5.6, 15): 6273,
    (200, 5.6, 30): 6273,
    (200, 5.6, 60): 6273,
    (200, 5.6, 125): 6271,
    (200, 5.6, 250): 6271,
    (200, 5.6, 500): 6271,
    (200, 5.6, 1000): 6273,
    (200, 5.6, 2000): 6273,
    (200, 5.6, 4000): 6277,
    (200, 8.0, 60): 6275,
    (200, 8.0, 125): 6277,
    (200, 8.0, 250): 6275,
    (200, 8.0, 500): 6277,
    (200, 8.0, 1000): 6277,
    (200, 11.0, 15): 6277,
    (200, 11.0, 30): 6277,
    (200, 11.0, 60): 6277,
    (200, 11.0, 125): 6277,
    (200, 11.0, 250): SATURATION_THRESHOLD_ISO200,  # or 5466?
    (200, 16.0, 8): SATURATION_THRESHOLD_ISO200,  # or 5846?
    (200, 16.0, 15): 6275,
    (200, 16.0, 30): 6273,
    (200, 16.0, 60): 6275,
    (200, 16.0, 125): 6275,
    (200, 16.0, 250): SATURATION_THRESHOLD_ISO200,  # or 1462?
    (200, 22.0, 8): SATURATION_THRESHOLD_ISO200,  # or 5846?
    (200, 22.0, 15): 6266,
    (200, 22.0, 30): 6266,
    (200, 22.0, 60): 6275,
    (200, 22.0, 125): 6277,

    (400, 5.6, 8): 12559,
    (400, 5.6, 15): 12555,
    (400, 5.6, 30): 12551,
    (400, 5.6, 60): 12551,
    (400, 5.6, 125): 12555,
    (400, 5.6, 250): 12551,
    (400, 5.6, 500): 12555,
    (400, 16.0, 30): 12551,
    (400, 16.0, 60): 12546,
    (400, 16.0, 125): 12551,
    (400, 16.0, 250): 12555,
    (400, 16.0, 500): SATURATION_THRESHOLD_ISO400,  # or 12111?

    (800, 5.6, 4): SATURATION_THRESHOLD_ISO800,  # or 22613?
    (800, 5.6, 8): SATURATION_THRESHOLD_ISO800,  # or 24666?
    (800, 5.6, 15): 25102,
    (800, 5.6, 30): 25102,
    (800, 5.6, 60): 25102,
    (800, 5.6, 125): 25102,
    (800, 5.6, 250): 25102,
    (800, 5.6, 500): 25112,
    (800, 5.6, 1000): 25111,
    (800, 5.6, 2000): SATURATION_THRESHOLD_ISO800,  # or 23795?
    (800, 5.6, 4000): SATURATION_THRESHOLD_ISO800,  # or 18746?
}

SATURATION_FACTOR_THRESHOLD = 2e-3


def get_reference_path():

    path = os.path.join(get_base_path(), 'van_hateren')

    return path


def get_directory(format_='iml'):

    directory = format_

    return directory


def get_filename(image_nb, format_='iml'):

    filename_string = 'imk{:05d}.{}'
    filename = filename_string.format(image_nb, format_)

    return filename


def get_path(image_nb, format_='iml'):

    reference_path = get_reference_path()
    directory = get_directory(format_=format_)
    filename = get_filename(image_nb, format_=format_)
    path = os.path.join(reference_path, directory, filename)

    return path


def get_settings_url():

    url = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/camerasettings.txt"

    return url


def get_url(image_nb, format_='iml'):

    url_string = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/{f}/imk{i:05d}.{f}"
    url = url_string.format(i=image_nb, f=format_)

    return url


def fetch(image_nbs=None, format_='iml', download_if_missing=True, verbose=False):
    """Fetch the Van Hateren dataset from the Max Planck Institute.

    Arguments:
        image_nbs: iterable
            Specify the numbers of the images to be fetched.
        format_: string
            ...
        download_if_missing: boolean (optional)
            If False, raise a IOError if the data is not locally available
            instead of trying to download the data from the source site.
            The default value is True.
        verbose: boolean (optional)
            ...
            The default value is False.
    """

    reference_path = get_reference_path()
    directory = get_directory(format_)
    path = os.path.join(reference_path, directory)
    if not os.path.isdir(path):
        os.makedirs(path)

    if image_nbs is None:
        image_nbs = get_image_nbs(fetched_only=False)
    else:
        image_nbs = np.array([
            image_nb
            for image_nb in image_nbs
            if image_nb in all_image_nbs and image_nb not in missing_image_nbs
        ])

    # Fetch settings.
    path = get_settings_path()
    if not os.path.isfile(path):
        if not download_if_missing:
            raise IOError("data not found and `download_if_missing` is False.")
        else:
            url = get_settings_url()
            if verbose:
                print("Load {} to file file://{}".format(url, path))
            urllib.request.urlretrieve(url, path)

    # Fetch images.
    paths = []
    for image_nb in image_nbs:
        path = get_path(image_nb, format_=format_)
        if not os.path.isfile(path):
            if not download_if_missing:
                raise IOError("data not found and `download_if_missing` is False.")
            else:
                url = get_url(image_nb, format_=format_)
                if verbose:
                    print("Load {} to file://{}.".format(url, path))
                try:
                    urllib.request.urlretrieve(url, path)
                except urllib.error.HTTPError as error:
                    if error.code == 404:  # i.e. not found
                        # msg = "{} not found.".format(url)
                        # raise urllib.error.HTTPError(url=error.url, code=error.code, msg=msg,
                        #                              hdrs=error.hdrs, fp=error.fp) from None
                        print(image_nb)
                except urllib.error.URLError as error:
                    raise urllib.error.URLError(reason=error.reason, filename=error.filename) from None
        paths.append(path)

    return Bunch(paths=paths)


def get_image_nbs(format_='iml', fetched_only=True):

    image_nbs = all_image_nbs.copy()
    for image_nb in missing_image_nbs:
        image_nbs.remove(image_nb)

    if fetched_only:
        image_nbs = [
            image_nb
            for image_nb in image_nbs
            if os.path.isfile(get_path(image_nb, format_=format_))
        ]
    image_nbs = np.array(image_nbs)

    return image_nbs


class Metadata(dict):

    @staticmethod
    def get_filename():

        return 'camerasettings.txt'

    @classmethod
    def get_path(cls):

        ref_path = get_reference_path()
        filename = cls.get_filename()
        path = os.path.join(ref_path, filename)

        return path

    @classmethod
    def load(cls):

        path = cls.get_path()

        with open(path, mode='r') as file:
            text = file.read()

        lines = text.split('\n')
        lines = lines[3:]  # remove header lines

        settings = {}
        for line in lines:
            line = line.strip()  # remove strip spaces
            line = re.sub(' +', ' ', line)  # remove duplicated spaces
            elements = line.split(' ')
            assert len(elements) == 5, elements
            image_nb = int(elements[0])
            image_settings = {
                'iso': int(elements[1]),  # ISO setting (i.e. electronic equivalent)
                'aperture': float(elements[2]),  # aperture
                'shutter': int(elements[3]),  # reciprocal shutter time (1/s)
                'factor': float(elements[4]),  # factor for converting pixel values to luminance
                                               # (cd/m2, luminance=factor*pixel value)
            }
            settings[image_nb] = image_settings
        settings = cls(settings)

        return settings

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def image_nbs(self):

        return np.array(list(self.keys()))

    def analyze(self):

        tmp = {}
        for image_nb in self.image_nbs:
            image_settings = self[image_nb]
            image_settings = tuple([image_settings[key] for key in ['iso', 'aperture', 'shutter']])
            if image_settings not in tmp:
                tmp[image_settings] = 1
            else:
                tmp[image_settings] += 1
        records = {
            'iso': [key[0] for key in tmp.keys()],
            'aperture': [key[1] for key in tmp.keys()],
            'shutter': [key[2] for key in tmp.keys()],
            'count': [value for value in tmp.values()]
        }
        dataframe = pd.DataFrame.from_records(records)

        return dataframe

    def get_image_nbs(self, iso=None, aperture=None, shutter=None):

        image_nbs = []
        for image_nb in self.image_nbs:
            if image_nb in all_image_nbs and image_nb not in missing_image_nbs:
                image_settings = self[image_nb]
                if (iso is None or image_settings['iso'] == iso) \
                        and (aperture is None or image_settings['aperture'] == aperture) \
                        and (shutter is None or image_settings['shutter'] == shutter):
                    image_nbs.append(image_nb)

        return image_nbs

    def get_unique(self, keys=None):

        if keys is None:
            keys = ['iso', 'aperture', 'shutter']

        unique_settings = {}
        for image_nb in self.image_nbs:
            image_settings = self[image_nb]
            key = tuple([image_settings[key] for key in keys])
            if key not in unique_settings:
                value = {key: image_settings[key] for key in keys}
                unique_settings[key] = value
        unique_settings = list(unique_settings.values())

        return unique_settings


get_settings_path = Metadata.get_path


load_settings = Metadata.load


def load_image_settings(image_nb):

    assert min_image_nb <= image_nb <= max_image_nb, image_nb

    settings = load_settings()
    image_settings = settings[image_nb]

    return image_settings


def load_raw_image(image_nb, format_='iml'):

    path = get_path(image_nb, format_=format_)
    if format_ == 'iml':
        image = load_iml_image(path, DTYPE, _WIDTH, _HEIGHT)
    elif format_ == 'imc':
        image = load_imc_image(path, DTYPE, _WIDTH, _HEIGHT)
    else:
        raise ValueError("unknown format value: {}".format(format_))

    return image


def load_data(image_nb, format_='iml'):

    image = load_raw_image(image_nb, format_=format_)
    data = image.data
    data = np.flipud(data)
    data = np.transpose(data)
    data = data.astype(np.float)

    return data


def load_luminance_data(image_nb, format_='iml'):

    image_settings = load_image_settings(image_nb)
    factor = image_settings['factor']
    data = load_data(image_nb, format_=format_)
    data = factor * data

    return data


def load_image(image_nb, format_='iml', max_luminance=None):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    if max_luminance is None or max_luminance == 'dataset':
        max_luminance = MAX_LUMINANCE
    elif max_luminance == 'image':
        # max_luminance = np.max(luminance_data)
        saturation_luminance = get_saturation_luminance(image_nb)
        max_luminance = saturation_luminance
    elif isinstance(max_luminance, float):
        pass
    else:
        raise ValueError("unknown max_luminance value: {}".format(max_luminance))
    normalized_luminance_data = luminance_data / max_luminance
    normalized_luminance_data[normalized_luminance_data >= 1.0] = 1.0
    assert 0.0 <= np.min(normalized_luminance_data), np.min(luminance_data)
    assert np.max(normalized_luminance_data) <= 1.0, np.max(luminance_data)
    dtype = 'uint8'
    dinfo = np.iinfo(dtype)
    assert dinfo.min == 0, dinfo.min
    max_value = float(dinfo.max + 1)
    factor = max_value / max_luminance
    data = factor * luminance_data
    data[float(dinfo.max) <= data] = float(dinfo.max)  # correct max. values
    assert float(dinfo.min) <= np.min(data), np.min(data)
    assert np.max(data) <= float(dinfo.max), np.max(data)
    data = data.astype(dtype)
    image = create_png_image(data)

    return image


def load_normalized_image(image_nb, format_='iml', max_luminance=None):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    if max_luminance is None or max_luminance == 'dataset':
        max_luminance = MAX_LUMINANCE
    elif max_luminance == 'image':
        # max_luminance = np.max(luminance_data)
        saturation_luminance = get_saturation_luminance(image_nb)
        max_luminance = saturation_luminance
    elif isinstance(max_luminance, float):
        pass
    else:
        raise ValueError("unknown max_luminance value: {}".format(max_luminance))
    normalized_luminance_data = luminance_data / max_luminance
    normalized_luminance_data[normalized_luminance_data >= 1.0] = 1.0
    assert 0.0 <= np.min(normalized_luminance_data), np.min(luminance_data)
    assert np.max(normalized_luminance_data) <= 1.0, np.max(luminance_data)

    # Center and reduce the pixel values.
    normalized_luminance_data -= np.median(normalized_luminance_data)
    if np.median(np.abs(normalized_luminance_data)) > 0.0:
        normalized_luminance_data /= 1.4826 * np.median(np.abs(normalized_luminance_data))
    normalized_luminance_data *= 0.02  # TODO transform into parameter.
    normalized_luminance_data += 0.5  # TODO transform into parameter.
    # assert np.count_nonzero(normalized_luminance_data < 0.0) == 0
    if np.count_nonzero(normalized_luminance_data < 0.0) == 0:
        warnings.warn("negative normalized luminance values")
    normalized_luminance_data[normalized_luminance_data < 0.0] = 0.0
    # assert np.count_nonzero(normalized_luminance_data > 1.0) == 0
    if np.count_nonzero(normalized_luminance_data > 1.0) == 0:
        warnings.warn("saturated normalized luminance values")
    normalized_luminance_data[normalized_luminance_data > 1.0] = 1.0

    dtype = 'uint8'
    dinfo = np.iinfo(dtype)
    assert dinfo.min == 0, dinfo.min
    max_value = float(dinfo.max + 1)
    factor = max_value
    data = factor * normalized_luminance_data
    data[float(dinfo.max) <= data] = float(dinfo.max)  # correct max. values
    assert float(dinfo.min) <= np.min(data), np.min(data)
    assert np.max(data) <= float(dinfo.max), np.max(data)
    data = data.astype(dtype)
    image = create_png_image(data)

    return image


class ImageMetadata(dict):

    @staticmethod
    def get_filename(image_nb):

        return 'imk{:05d}.json'.format(image_nb)

    @classmethod
    def get_path(cls, image_nb, format_='iml'):

        ref_path = get_reference_path()
        directory = get_directory(format_)
        filename = cls.get_filename(image_nb)
        path = os.path.join(ref_path, directory, filename)

        return path

    @classmethod
    def load(cls, image_nb, format_='iml'):

        path = cls.get_path(image_nb, format_=format_)
        try:
            with open(path, mode='r') as file:
                kwargs = json.load(file)
        except FileNotFoundError:
            kwargs = {
                'image_nb': int(image_nb),
                'format': format_,
            }
        image_metadata = cls(**kwargs)

        return image_metadata

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def save(self):

        image_nb = self['image_nb']
        format_ = self['format']
        path = self.get_path(image_nb, format_=format_)
        with open(path, mode='w') as file:
            json.dump(self, file, indent=4)

        return

    @classmethod
    def cache(cls, key):

        def decorator(function):
            def wrapper(image_nb, format_='iml', force=False):
                image_metadata = cls.load(image_nb, format_=format_)
                if key not in image_metadata or force:
                    answer = function(image_nb, format_=format_)
                    image_metadata[key] = answer
                    image_metadata.save()
                else:
                    answer = image_metadata[key]
                return answer
            return wrapper

        return decorator


load_image_metadata = ImageMetadata.load


def repeat_for_each_image(operation):

    def repeater(image_nbs=None, format_='iml', verbose=False, force=False):

        if image_nbs is None:
            image_nbs = get_image_nbs()
        if verbose:
            image_nbs = tqdm.tqdm(image_nbs)
        answers = np.array([
            operation(image_nb, format_=format_, force=force)
            for image_nb in image_nbs
        ])

        return answers

    return repeater


@ImageMetadata.cache('min_value')
def get_min_value(image_nb, format_='iml'):

    image = load_raw_image(image_nb, format_=format_)
    min_value = int(image.min)

    return min_value


get_min_values = repeat_for_each_image(get_min_value)


@ImageMetadata.cache('max_value')
def get_max_value(image_nb, format_='iml'):

    image = load_raw_image(image_nb, format_=format_)
    max_value = int(image.max)

    return max_value


get_max_values = repeat_for_each_image(get_max_value)


@ImageMetadata.cache('min_luminance')
def get_min_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    min_luminance = np.min(luminance_data)

    return min_luminance


get_min_luminances = repeat_for_each_image(get_min_luminance)


@ImageMetadata.cache('max_luminance')
def get_max_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    max_luminance = np.max(luminance_data)

    return max_luminance


get_max_luminances = repeat_for_each_image(get_max_luminance)


@ImageMetadata.cache('mean_luminance')
def get_mean_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    mean_luminance = np.mean(luminance_data)

    return mean_luminance


get_mean_luminances = repeat_for_each_image(get_mean_luminance)


@ImageMetadata.cache('std_luminance')
def get_std_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    std_luminance = np.std(luminance_data)

    return std_luminance


get_std_luminances = repeat_for_each_image(get_std_luminance)


@ImageMetadata.cache('median_luminance')
def get_median_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    mean_luminance = np.median(luminance_data)

    return mean_luminance


get_median_luminances = repeat_for_each_image(get_median_luminance)


@ImageMetadata.cache('mad_luminance')
def get_mad_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    mad_luminance = 1.4826 * np.median(np.abs(luminance_data - np.median(luminance_data)))

    return mad_luminance


get_mad_luminances = repeat_for_each_image(get_mad_luminance)


@ImageMetadata.cache('log_mean_luminance')
def get_log_mean_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    # log_mean_luminance = np.mean(np.log(luminance_data))
    log_mean_luminance = np.mean(np.log(1.0 + luminance_data))

    return log_mean_luminance


get_log_mean_luminances = repeat_for_each_image(get_log_mean_luminance)


@ImageMetadata.cache('log_std_luminance')
def get_log_std_luminance(image_nb, format_='iml'):

    luminance_data = load_luminance_data(image_nb, format_=format_)
    log_std_luminance = np.std(np.log(1.0 + luminance_data))

    return log_std_luminance


get_log_std_luminances = repeat_for_each_image(get_log_std_luminance)


@ImageMetadata.cache('saturation_luminance')
def get_saturation_luminance(image_nb, format_='iml'):

    image_settings = load_image_settings(image_nb)
    iso_value = image_settings['iso']
    factor = image_settings['factor']
    saturation_value = SATURATION_VALUES[iso_value]
    saturation_luminance = factor * saturation_value

    return saturation_luminance


get_saturation_luminances = repeat_for_each_image(get_saturation_luminance)


# @ImageMetadata.cache('saturation_factor')
# def get_saturation_factor(image_nb, format_='iml'):
#
#     luminance_data = load_luminance_data(image_nb, format_=format_)
#     luminance_max = np.max(luminance_data)
#     nb_pixels_lum_max = np.count_nonzero(luminance_data == luminance_max)
#     nb_pixels = np.size(luminance_data)
#     if nb_pixels_lum_max == 1:
#         factor = 0.0
#     else:
#         factor = float(nb_pixels_lum_max) / float(nb_pixels)
#
#     return factor
@ImageMetadata.cache('saturation_factor')
def get_saturation_factor(image_nb, format_='iml'):

    image = load_raw_image(image_nb, format_=format_)
    values = np.ravel(image.data)
    image_settings = load_image_settings(image_nb)
    iso_value = image_settings['iso']
    saturation_value = SATURATION_VALUES[iso_value]
    nb_saturated_values = np.count_nonzero(values >= saturation_value)
    nb_values = np.size(values)
    factor = float(nb_saturated_values) / float(nb_values)

    return factor


get_saturation_factors = repeat_for_each_image(get_saturation_factor)


@ImageMetadata.cache('is_saturated')
def get_is_saturated(image_nb, format_='iml'):

    saturation_factor = get_saturation_factor(image_nb, format_=format_)
    is_saturated = saturation_factor > SATURATION_FACTOR_THRESHOLD

    return is_saturated
# @ImageMetadata.cache('is_saturated')
# def get_is_saturated(image_nb, format_='iml'):
#
#     image = load_raw_image(image_nb, format_=format_)
#     values = np.ravel(image.data)
#     unique_values = np.unique(values)
#     unique_values.sort()
#     if unique_values.size >= 2:
#         first_max_value = unique_values[-1]
#         second_max_value = unique_values[-2]
#         first_counts = np.count_nonzero(values == first_max_value)
#         second_counts = np.count_nonzero(values == second_max_value)
#         is_saturated = first_counts > second_counts
#     else:
#         is_saturated = False
#
#     return is_saturated


get_are_saturated = repeat_for_each_image(get_is_saturated)


def load_saturation_mask(image_nb, format_='iml'):

    image = load_raw_image(image_nb, format_=format_)
    image_settings = load_image_settings(image_nb)
    iso_value = image_settings['iso']
    saturation_value = SATURATION_VALUES[iso_value]
    mask = image.data >= saturation_value

    return mask


def get_horizontal_angles():
    """Get horizontal visual angles (in °)."""

    x = np.arange(0, _WIDTH)
    x = x.astype(np.float)
    x -= np.mean(x)
    a_x = _ANGULAR_RESOLUTION * x

    return a_x


def get_vertical_angles():
    """Get vertical visual angles (in °)."""

    y = np.arange(0, _HEIGHT)
    y = y.astype(np.float)
    y -= np.mean(y)
    a_y = _ANGULAR_RESOLUTION * y

    return a_y


def get_angles():

    a_x = get_horizontal_angles()
    a_y = get_vertical_angles()
    a = np.meshgrid(a_x, a_y)

    return a
