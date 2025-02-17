"""Flashed images perturbed with checkerboard"""

import array
import io
import math
# import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.interpolate
import scipy.ndimage
import tempfile
import tqdm

# from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL.Image import fromarray
from PIL.Image import open as open_image
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.csv import open_file as open_csv_file
from pystim.utils import handle_arguments_and_configurations
from pystim.utils import get_grey_frame


name = 'fipwc'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
    'reference_images': {  # a selection of promising images
        0: ('reference', 0),  # i.e. grey
        1: ('van Hateren', 5),
        2: ('van Hateren', 31),
        3: ('van Hateren', 46),
        # 4: ('van Hateren', 39),
    },
    'perturbations': {
        # 'pattern_indices': list(range(0, 2)),
        # 'pattern_indices': list(range(0, 18)),  # TODO
        'pattern_indices': list(range(0, 9)),  # TODO
        # 'amplitudes': [float(a) / float(256) for a in [10, 28]],  # TODO remove this line?
        # 'amplitudes': [float(a) / float(256) for a in [-28, +28]],
        # 'amplitudes': [float(a) / float(256) for a in [2, 4, 7, 10, 14, 18, 23, 28]],  # TODO
        'amplitudes': [float(a) / float(256) for a in [-30, -20, -15, -10, +10, +15, +20, +30]],
        # 'nb_horizontal_checks': 60,
        # 'nb_vertical_checks': 60,
        # 'nb_horizontal_checks': 57,
        # 'nb_vertical_checks': 57,
        'nb_horizontal_checks': 56,
        'nb_vertical_checks': 56,
        # 'resolution': 50.0,  # µm / pixel
        'resolution': float(15) * 3.5,  # µm / pixel
        'with_random_patterns': True,
    },
    # 'display_rate': 50.0,  # Hz  # TODO
    'display_rate': 40.0,  # Hz
    'frame': {
        'width': 864,
        'height': 864,
        'duration': 0.3,  # s
        'resolution': 3.5,  # µm / pixel  # fixed by the setup
    },
    # image_resolution = 3.3  # µm / pixel  # fixed by the monkey eye
    # 'image_resolution': 0.8,  # µm / pixel  # fixed by the salamander eye
    'image_resolution': 3.5,  # µm / pixel  # fixed by hand  # TODO correct.
    'background_luminance': 0.5,  # arb. unit
    'mean_luminance': 0.5,  # arb. unit
    # 'std_luminance': 0.06,  # arb. unit
    'std_luminance': 0.2,  # arb. unit
    'nb_repetitions': 10,
    # 'nb_repetitions': 2,
}


def load_resource(url):

    with urlopen(url) as handle:
        resource = handle.read()

    return resource


def get_palmer_resource_locator(path, index):

    assert 1 <= index <= 6

    frame_index = 1200 * (index - 1) + 600
    filename = "frame{i:04d}.png".format(i=frame_index)
    path = os.path.join(path, filename)
    url = "file://localhost" + path

    return url


def get_palmer_resource_locators(path=None, indices=None):

    if indices is None:
        indices = range(1, 7)

    urls = [
        get_palmer_resource_locator(path, index)
        for index in indices
    ]

    return urls


def convert_palmer_resource_to_image(resource):

    data = io.BytesIO(resource)
    image = open_image(data)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    image = data.reshape(image.size)

    image = image.astype(np.float)
    image = image / (2.0 ** 8)

    return image


def get_van_hateren_resource_locator(index, format_='iml', scheme='file', path=None, mirror='Lies'):

    filename = "imk{:05d}.{}".format(index, format_)

    if scheme == 'file':
        assert path is not None
        netloc = "localhost"
        path = os.path.join(path, filename)
        url = "{}://{}/{}".format(scheme, netloc, path)
    elif scheme == 'http':
        if mirror == 'Lies':
            assert 1 <= index <= 4212
            assert format_ in ['iml', 'imc']
            url_string = "http://cin-11.medizin.uni-tuebingen.de:61280/vanhateren/{f}/imk{i:05d}.{f}"
            url = url_string.format(i=index, f=format_)
        elif mirror == 'Ivanov':
            assert 1 <= index <= 99
            if format_ == 'iml':
                url = "http://pirsquared.org/research/vhatdb/imk{i:05d}.{f}".format(i=index, f=format_)
            elif format_ == 'imc':
                url = "http://pirsquared.org/research/vhatdb/{f}/imk{i:05d}.{f}".format(i=index, f=format_)
            else:
                raise ValueError("unexpected format value: {}".format(format_))
        else:
            raise ValueError("unexpected mirror value: {}".format(mirror))
    else:
        raise ValueError("unexpected scheme value: {}".format(scheme))

    return url


def get_van_hateren_resource_locators(path=None, indices=None, format_='iml', mirror='Lies'):

    assert format_ in ['iml', 'imc']

    if indices is None:
        if mirror == 'Lies':
            indices = range(0, 4213)
        elif mirror == 'Ivanov':
            indices = range(1, 100)
        else:
            raise ValueError("unexpected mirror value: {}".format(mirror))

    urls = [
        get_van_hateren_resource_locator(index, format_=format_, path=path, mirror=mirror)
        for index in indices
    ]

    return urls


def convert_van_hateren_resource_to_image(resource):

    data = array.array('H', resource)
    data.byteswap()
    data = np.array(data, dtype=np.uint16)
    image = data.reshape(1024, 1536)
    image = image.astype(np.float)
    image = image / (2.0 ** (12 + 1))

    return image


def get_checkerboard_locator(index, scheme='file', path=None):

    filename = "checkerboard{:05d}.png".format(index)

    if scheme == 'file':
        netloc = "localhost"
        path = os.path.join(path, filename)
        url = "{}://{}/{}".format(scheme, netloc, path)
    else:
        raise ValueError("unexpected scheme value: {}".format(scheme))

    return url


def get_reference_locator(index, scheme='file', path=None):

    filename = "reference_{:05d}.png".format(index)

    if scheme == 'file':
        netloc = "localhost"
        path = os.path.join(path, filename)
        url = "{}://{}/{}".format(scheme, netloc, path)
    else:
        raise ValueError("unexpected scheme value: {}".format(scheme))

    return url


def get_resource_locator(index, dataset='van Hateren', **kwargs):

    if dataset == 'van Hateren':
        url = get_van_hateren_resource_locator(index, **kwargs)
    elif dataset == 'Palmer':
        url = get_palmer_resource_locator(index, **kwargs)
    elif dataset == 'checkerboard':
        url = get_checkerboard_locator(index, **kwargs)
    elif dataset == 'reference':
        url = get_reference_locator(index, **kwargs)
    else:
        raise ValueError("unexpected dataset value: {}".format(dataset))

    return url


def is_resource(url):

    url = urlparse(url)
    if url.scheme == 'file':
        path = url.path[1:]
        ans = os.path.isfile(path)
        print("{} is resource: {}".format(path, ans))
    elif url.scheme == 'http':
        ans = True
    else:
        raise ValueError("unexpected url scheme: {}".format(url.scheme))

    return ans


def is_available_online(dataset):

    return dataset in ['van Hateren', 'Palmer']


def generate_reference_image(reference_index, dataset='reference', index=0, path=None):

    url = get_resource_locator(reference_index, dataset=dataset, scheme='file', path=path)
    url = urlparse(url)
    path = url.path[1:]
    # Generate reference image.
    if index == 0:
        height = 864  # px  # similar to the van Hateren dataset
        width = 864  # px  # similar to the van Hateren dataset
        shape = (height, width)
        dtype = np.uint8
        info = np.iinfo(dtype)
        v = (info.max - info.min + 1) // 2
        a = v * np.ones(shape, dtype=dtype)
        image = fromarray(a)
        image.save(path)
    else:
        raise NotImplementedError()

    return


def collect_reference_image(reference_index, dataset='van Hateren', index=0, config=None, **kwargs):

    local_url = get_resource_locator(reference_index, dataset='reference', scheme='file', **kwargs)
    if not is_resource(local_url):
        if is_available_online(dataset):
            # Retrieve image.
            remote_url = get_resource_locator(index, dataset=dataset, scheme='http', **kwargs)
            local_url = get_resource_locator(reference_index, dataset=dataset, scheme='file', **kwargs)
            local_url = urlparse(local_url)
            local_path = local_url.path[1:]
            urlretrieve(remote_url, local_path)
            # Process image.
            image = load_reference_image_old(reference_index, path=kwargs['path'])
            image = get_reference_frame(image, config)
            image = float_frame_to_uint8_frame(image)
            local_url = get_resource_locator(reference_index, dataset='reference', scheme='file', **kwargs)
            local_url = urlparse(local_url)
            local_path = local_url.path[1:]
            save_frame(local_path, image)
        else:
            generate_reference_image(reference_index, dataset=dataset, index=index, **kwargs)

    return


def generate_perturbation_pattern(index, nb_horizontal_checks=10, nb_vertical_checks=10, path=None):

    assert path is not None

    np.random.seed(seed=index)
    dtype = np.uint8
    info = np.iinfo(dtype)
    a = np.array([info.min, info.max], dtype=dtype)
    height = nb_vertical_checks
    width = nb_horizontal_checks
    shape = (height, width)
    pattern = np.random.choice(a=a, size=shape)
    image = fromarray(pattern)
    image.save(path)

    return


def collect_perturbation_pattern(index, nb_horizontal_checks=10, nb_vertical_checks=10, path=None):

    assert path is not None

    url = get_resource_locator(index, dataset='checkerboard', scheme='file', path=path)
    if not is_resource(url):
        url = urlparse(url)
        path = url.path[1:]
        generate_perturbation_pattern(index, nb_horizontal_checks=nb_horizontal_checks,
                                      nb_vertical_checks=nb_vertical_checks, path=path)

    return


def load_perturbation_pattern(index, path):

    url = get_resource_locator(index, dataset='checkerboard', scheme='file', path=path)
    url = urlparse(url)
    path = url.path[1:]

    image = open_image(path)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    width, height = image.size
    data = data.reshape(height, width)

    data = data.astype(np.float)
    data = data / np.iinfo(np.uint8).max

    return data


def load_reference_image_old(index, path):

    dtype = np.uint16
    height = 1024
    width = 1536

    url = get_resource_locator(index, dataset='van Hateren', scheme='file', path=path)
    url = urlparse(url)
    path = url.path[1:]

    with open(path, mode='rb') as handle:
        data_bytes = handle.read()
    data = array.array('H', data_bytes)
    data.byteswap()
    data = np.array(data, dtype=dtype)
    data = data.reshape(height, width)

    data = data.astype(np.float)
    data = data / np.iinfo(dtype).max

    return data


def load_reference_image(index, path):

    url = get_resource_locator(index, dataset='reference', scheme='file', path=path)
    url = urlparse(url)
    path = url.path[1:]

    image = open_image(path)
    data = image.getdata()
    data = np.array(data, dtype=np.uint8)
    width, height = image.size
    data = data.reshape(height, width)

    data = data.astype(np.float)
    info = np.iinfo(np.uint8)
    data = (data - float(info.min)) / float(info.max - info.min + 1)

    return data


def get_frame(image, config):

    image_height, image_width = image.shape
    image_resolution = config['image_resolution']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    frame_shape = frame_height, frame_width
    frame_resolution = config['frame']['resolution']

    background_luminance = config['background_luminance']

    if frame_resolution <= image_resolution:
        # Up-sample the image (interpolation).
        image_x = image_resolution * np.arange(0, image_height)
        image_x = image_x - np.mean(image_x)
        image_y = image_resolution * np.arange(0, image_width)
        image_y = image_y - np.mean(image_y)
        image_z = image
        spline = scipy.interpolate.RectBivariateSpline(image_x, image_y, image_z, kx=1, ky=1)
        frame_x = frame_resolution * np.arange(0, frame_height)
        frame_x = frame_x - np.mean(frame_x)
        mask_x = np.logical_and(
            np.min(image_x) - 0.5 * image_resolution <= frame_x,
            frame_x <= np.max(image_x) + 0.5 * image_resolution
        )
        frame_y = frame_resolution * np.arange(0, frame_width)
        frame_y = frame_y - np.mean(frame_y)
        mask_y = np.logical_and(
            np.min(image_y) - 0.5 * image_resolution <= frame_y,
            frame_y <= np.max(image_y) + 0.5 * image_resolution
        )
        frame_z = spline(frame_x[mask_x], frame_y[mask_y])
        frame_i_min = np.min(np.nonzero(mask_x))
        frame_i_max = np.max(np.nonzero(mask_x)) + 1
        frame_j_min = np.min(np.nonzero(mask_y))
        frame_j_max = np.max(np.nonzero(mask_y)) + 1
        frame = background_luminance * np.ones(frame_shape, dtype=np.float)
        frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z
    else:
        # Down-sample the image (decimation).
        image_frequency = 1.0 / image_resolution
        frame_frequency = 1.0 / frame_resolution
        cutoff = frame_frequency / image_frequency
        sigma = math.sqrt(2.0 * math.log(2.0)) / (2.0 * math.pi * cutoff)
        filtered_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
        # see https://en.wikipedia.org/wiki/Gaussian_filter for a justification of this formula
        image_x = image_resolution * np.arange(0, image_height)
        image_x = image_x - np.mean(image_x)
        image_y = image_resolution * np.arange(0, image_width)
        image_y = image_y - np.mean(image_y)
        image_z = filtered_image
        spline = scipy.interpolate.RectBivariateSpline(image_x, image_y, image_z, kx=1, ky=1)
        frame_x = frame_resolution * np.arange(0, frame_height)
        frame_x = frame_x - np.mean(frame_x)
        mask_x = np.logical_and(
            np.min(image_x) - 0.5 * image_resolution <= frame_x,
            frame_x <= np.max(image_x) + 0.5 * image_resolution
        )
        frame_y = frame_resolution * np.arange(0, frame_width)
        frame_y = frame_y - np.mean(frame_y)
        mask_y = np.logical_and(
            np.min(image_y) - 0.5 * image_resolution <= frame_y,
            frame_y <= np.max(image_y) + 0.5 * image_resolution
        )
        frame_z = spline(frame_x[mask_x], frame_y[mask_y])
        frame_i_min = np.min(np.nonzero(mask_x))
        frame_i_max = np.max(np.nonzero(mask_x)) + 1
        frame_j_min = np.min(np.nonzero(mask_y))
        frame_j_max = np.max(np.nonzero(mask_y)) + 1
        frame = background_luminance * np.ones(frame_shape, dtype=np.float)
        frame[frame_i_min:frame_i_max, frame_j_min:frame_j_max] = frame_z

    limits = frame_i_min, frame_i_max, frame_j_min, frame_j_max

    return frame, limits


def get_reference_frame(reference_image, config):

    frame, limits = get_frame(reference_image, config)
    i_min, i_max, j_min, j_max = limits

    mean_luminance = config['mean_luminance']
    std_luminance = config['std_luminance']

    frame_roi = frame[i_min:i_max, j_min:j_max]
    frame_roi = frame_roi - np.mean(frame_roi)
    if np.std(frame_roi) > 0.0:
        frame_roi = frame_roi / np.std(frame_roi)
        frame_roi = frame_roi * std_luminance
    frame_roi = frame_roi + mean_luminance
    frame[i_min:i_max, j_min:j_max] = frame_roi

    return frame


def get_perturbation_frame(perturbation_image, config, verbose=False):

    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    frame_shape = (frame_height, frame_width)
    frame_resolution = config['frame']['resolution']

    frame_x = frame_resolution * np.arange(0, frame_height)
    frame_x = frame_x - np.mean(frame_x)
    frame_y = frame_resolution * np.arange(0, frame_width)
    frame_y = frame_y - np.mean(frame_y)

    perturbation_resolution = config['perturbations']['resolution']

    perturbation = perturbation_image
    perturbation_height, perturbation_width = perturbation.shape
    perturbation_x = perturbation_resolution * np.arange(0, perturbation_height)
    perturbation_x = perturbation_x - np.mean(perturbation_x)
    perturbation_y = perturbation_resolution * np.arange(0, perturbation_width)
    perturbation_y = perturbation_y - np.mean(perturbation_y)
    perturbation_z = perturbation
    perturbation_x_, perturbation_y_ = np.meshgrid(perturbation_x, perturbation_y)
    perturbation_points = np.stack((perturbation_x_.flatten(), perturbation_y_.flatten()), axis=-1)
    perturbation_data = perturbation_z.flatten()
    interpolate = scipy.interpolate.NearestNDInterpolator(perturbation_points, perturbation_data)
    mask_x = np.logical_and(
        np.min(perturbation_x) - 0.5 * perturbation_resolution <= frame_x,
        frame_x <= np.max(perturbation_x) + 0.5 * perturbation_resolution
    )
    mask_y = np.logical_and(
        np.min(perturbation_y) - 0.5 * perturbation_resolution <= frame_y,
        frame_y <= np.max(perturbation_y) + 0.5 * perturbation_resolution
    )
    frame_x_, frame_y_ = np.meshgrid(frame_x[mask_x], frame_y[mask_y])
    frame_x_ = frame_x_.transpose().flatten()
    frame_y_ = frame_y_.transpose().flatten()
    frame_points_ = np.stack((frame_x_, frame_y_), axis=-1)
    frame_data_ = interpolate(frame_points_)
    frame_z_ = np.reshape(frame_data_, (frame_x[mask_x].size, frame_y[mask_y].size))
    i_min = np.min(np.nonzero(mask_x))
    i_max = np.max(np.nonzero(mask_x)) + 1
    j_min = np.min(np.nonzero(mask_y))
    j_max = np.max(np.nonzero(mask_y)) + 1
    frame = np.zeros(frame_shape, dtype=np.float)
    frame[i_min:i_max, j_min:j_max] = frame_z_

    frame_roi = frame[i_min:i_max, j_min:j_max]
    if verbose:
        print("mean (luminance): {}".format(np.mean(frame_roi)))
        print("std (luminance): {}".format(np.std(frame_roi)))
        print("min (luminance): {}".format(np.min(frame_roi)))
        print("max (luminance): {}".format(np.max(frame_roi)))
    frame_roi = -1.0 + 2.0 * frame_roi
    if verbose:
        print("mean (luminance): {}".format(np.mean(frame_roi)))
        print("std (luminance): {}".format(np.std(frame_roi)))
        print("min (luminance): {}".format(np.min(frame_roi)))
        print("max (luminance): {}".format(np.max(frame_roi)))
    frame[i_min:i_max, j_min:j_max] = frame_roi

    return frame


def float_frame_to_uint8_frame(float_frame):

    # Rescale pixel values.
    dtype = np.uint8
    dinfo = np.iinfo(dtype)
    nb_levels = (dinfo.max - dinfo.min + 1)
    float_frame = float(nb_levels) * float_frame
    # Process saturating pixels.
    float_frame[float_frame < float(dinfo.min)] = float(dinfo.min)
    float_frame[float(dinfo.max + 1) <= float_frame] = float(dinfo.max)
    # Convert pixel types.
    uint8_frame = float_frame.astype(dtype)

    return uint8_frame


def get_perturbed_frame(reference_image, perturbation_pattern, perturbation_amplitude, config):

    reference_frame = get_reference_frame(reference_image, config)
    perturbation_frame = get_perturbation_frame(perturbation_pattern, config)
    frame = reference_frame + perturbation_amplitude * perturbation_frame

    return frame


def get_combinations(reference_images_indices, perturbation_patterns_indices, perturbation_amplitudes_indices):

    index = 1  # 0 skipped because it corresponds to the frame used between stimuli
    combinations = {}

    for i in reference_images_indices:
        combinations[index] = (i, np.nan, np.nan)  # TODO check if `np.nan` are valid.
        index +=1
    for i in reference_images_indices:
        for j in perturbation_patterns_indices:
            for k in perturbation_amplitudes_indices:
                combinations[index] = (i, j, k)
                index += 1

    return combinations


def get_random_combinations(reference_images_indices, random_perturbation_patterns_indices, nb_combinations):

    index = nb_combinations  # skip deterministic combinations
    combinations = {}

    for j in random_perturbation_patterns_indices:  # pattern first
        for i in reference_images_indices:
            combinations[index] = (i, j, np.nan)
            index += 1

    return combinations


def save_frame(path, frame):

    image = fromarray(frame)
    image.save(path)

    return


def get_permutations(indices, nb_repetitions=1, seed=42):

    np.random.seed(seed)

    permutations = {
        k: np.random.permutation(indices)
        for k in range(0, nb_repetitions)
    }

    return permutations


def get_random_combination_groups(random_combination_indices, nb_repetitions=1):

    nb_combinations = len(random_combination_indices)
    nb_combinations_per_repetition = nb_combinations // nb_repetitions
    assert nb_combinations % nb_repetitions == 0, "not nb_combinations ({}) % nb_repetitions ({}) == 0".format(nb_combinations, nb_repetitions)

    groups = {
        k: random_combination_indices[(k+0)*nb_combinations_per_repetition:(k+1)* nb_combinations_per_repetition]
        for k in range(0, nb_repetitions)
    }

    return groups


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)
    print(path)

    reference_images_path = os.path.join(path, "reference_images")
    if not os.path.isdir(reference_images_path):
        os.makedirs(reference_images_path)

    perturbation_patterns_path = os.path.join(path, "perturbation_patterns")
    if not os.path.isdir(perturbation_patterns_path):
        os.makedirs(perturbation_patterns_path)

    frames_path = os.path.join(path, "frames")
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

    # Get configuration parameters.
    reference_images = config['reference_images']
    nb_horizontal_checks = config['perturbations']['nb_horizontal_checks']
    nb_vertical_checks = config['perturbations']['nb_vertical_checks']
    assert nb_horizontal_checks % 2 == 0, "number of checks should be even (horizontally): {}".format(nb_horizontal_checks)
    assert nb_vertical_checks % 2 == 0, "number of checks should be even (vertically): {}".format(nb_vertical_checks)
    with_random_patterns = config['perturbations']['with_random_patterns']
    perturbation_patterns_indices = config['perturbations']['pattern_indices']
    perturbation_amplitudes = config['perturbations']['amplitudes']
    perturbation_amplitudes_indices = [k for k, _ in enumerate(perturbation_amplitudes)]
    display_rate = config['display_rate']
    frame_width_in_px = config['frame']['width']
    frame_height_in_px = config['frame']['height']
    frame_duration = config['frame']['duration']
    nb_repetitions = config['nb_repetitions']

    # Collect reference images.
    reference_indices = [
        int(key)
        for key in reference_images.keys()
    ]
    for reference_index in reference_indices:
        dataset, index = reference_images[str(reference_index)]
        collect_reference_image(reference_index, dataset=dataset, index=index,
                                path=reference_images_path, config=config)

    # Create .csv file for reference_image.
    csv_filename = "{}_reference_images.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    columns = ['reference_image_path']
    csv_file = open_csv_file(csv_path, columns=columns)
    for index in reference_indices:
        reference_image_path = os.path.join("reference_images", "reference_{:05d}.png".format(index))
        csv_file.append(reference_image_path=reference_image_path)
    csv_file.close()

    # Prepare perturbation pattern indices.
    if with_random_patterns:
        # TODO check the following lines!
        # Compute the number of random patterns.
        nb_perturbation_patterns = len(perturbation_patterns_indices)
        nb_perturbation_amplitudes = len(perturbation_amplitudes_indices)
        nb_perturbations = nb_perturbation_patterns * nb_perturbation_amplitudes
        nb_random_patterns = nb_perturbations * nb_repetitions
        print("number of random patterns: {}".format(nb_random_patterns))  # TODO remove this line.
        # Choose the indices of the random patterns.
        random_patterns_indices = nb_perturbation_patterns + np.arange(0, nb_random_patterns)
        # Define pattern indices.
        all_patterns_indices = np.concatenate((perturbation_patterns_indices, random_patterns_indices))
    else:
        nb_random_patterns = 0
        random_patterns_indices = None
        all_patterns_indices = perturbation_patterns_indices

    print("Start collecting perturbation patterns...")

    # TODO remove the following commented lines?
    # for index in perturbation_patterns_indices:
    #     collect_perturbation_pattern(index, nb_horizontal_checks=nb_horizontal_checks,
    #                                  nb_vertical_checks=nb_vertical_checks, path=perturbation_patterns_path)
    for index in tqdm.tqdm(all_patterns_indices):
        collect_perturbation_pattern(index, nb_horizontal_checks=nb_horizontal_checks,
                                     nb_vertical_checks=nb_vertical_checks, path=perturbation_patterns_path)

    print("End collecting perturbation patterns.")

    # Create .csv file for perturbation pattern.
    csv_filename = "{}_perturbation_patterns.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    columns = ['perturbation_pattern_path']
    csv_file = open_csv_file(csv_path, columns=columns)
    # TODO remove the following commented line?
    # for index in perturbation_patterns_indices:
    for index in all_patterns_indices:
        perturbation_pattern_path = os.path.join("perturbation_patterns", "checkerboard{:05d}.png".format(index))
        csv_file.append(perturbation_pattern_path=perturbation_pattern_path)
    csv_file.close()

    # Create .csv file for perturbation amplitudes.
    csv_filename = "{}_perturbation_amplitudes.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    columns = ['perturbation_amplitude']
    csv_file = open_csv_file(csv_path, columns=columns)
    for perturbation_amplitude in perturbation_amplitudes:
        csv_file.append(perturbation_amplitude=perturbation_amplitude)
    csv_file.close()

    # Compute the number of images.
    nb_reference_images = len(reference_indices)
    nb_perturbation_patterns = len(perturbation_patterns_indices)
    nb_perturbation_amplitudes = len(perturbation_amplitudes_indices)
    nb_images = 1 + nb_reference_images * (1 + nb_perturbation_patterns * nb_perturbation_amplitudes)
    if with_random_patterns:
        nb_images = nb_images + nb_reference_images * nb_random_patterns  # TODO check this line!

    combinations = get_combinations(reference_indices, perturbation_patterns_indices,
                                    perturbation_amplitudes_indices)

    if with_random_patterns:
        nb_deterministic_combinations = len(combinations) + 1  # +1 to take inter-stimulus frame into account
        random_combinations = get_random_combinations(reference_indices, random_patterns_indices, nb_deterministic_combinations)
    else:
        random_combinations = None

    # Create .csv file.
    csv_filename = "{}_combinations.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    columns = ['reference_id', 'perturbation_pattern_id', 'perturbation_amplitude_id']
    csv_file = open_csv_file(csv_path, columns=columns, dtype='Int64')
    for combination_index in combinations:
        combination = combinations[combination_index]
        # TODO remove the following commented lines?
        # kwargs = {
        #     'reference_id': reference_indices[combination[0]],
        #     'perturbation_pattern_id': perturbation_patterns_indices[combination[1]],
        #     'perturbation_amplitude_id': perturbation_amplitudes_indices[combination[2]],
        # }
        reference_id, perturbation_pattern_id, perturbation_amplitude_id = combination
        kwargs = {
            'reference_id': reference_id,
            'perturbation_pattern_id': perturbation_pattern_id,
            'perturbation_amplitude_id': perturbation_amplitude_id,
        }
        csv_file.append(**kwargs)
    # TODO check the following lines!
    # Add random pattern (if necessary).
    if with_random_patterns:
        for combination_index in random_combinations:
            combination = random_combinations[combination_index]
            reference_id, pattern_id, amplitude_id = combination
            kwargs = {
                'reference_id': reference_id,
                'perturbation_pattern_id': pattern_id,
                'perturbation_amplitude_id': amplitude_id,
            }
            csv_file.append(**kwargs)
    csv_file.close()

    # TODO fix the permutations?

    nb_combinations = len(combinations)
    nb_frame_displays = int(display_rate * frame_duration)
    assert display_rate * frame_duration == float(nb_frame_displays)
    nb_displays = nb_frame_displays + nb_repetitions * nb_combinations * (2 * nb_frame_displays)
    if with_random_patterns:
        nb_random_combinations = len(random_combinations)
        nb_displays = nb_displays + nb_random_combinations * (2 * nb_frame_displays)

    display_time = float(nb_displays) / display_rate
    print("display time: {} s ({} min)".format(display_time, display_time / 60.0))
    # TODO improve feedback.

    combination_indices = list(combinations)
    permutations = get_permutations(combination_indices, nb_repetitions=nb_repetitions)

    if with_random_patterns:
        random_combination_indices = list(random_combinations)
        random_combination_groups = get_random_combination_groups(random_combination_indices, nb_repetitions=nb_repetitions)
    else:
        random_combination_groups = None

    print("Start creating .bin file...")

    # Create .bin file.
    bin_filename = "fipwc.bin"
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images, frame_width=frame_width_in_px, frame_height=frame_height_in_px)
    # Save grey frame.
    grey_frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=0.5)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    # # Save frame in .bin file.
    bin_file.append(grey_frame)
    # # Save frame as .png file.
    grey_frame_filename = "grey.png"
    grey_frame_path = os.path.join(frames_path, grey_frame_filename)
    save_frame(grey_frame_path, grey_frame)
    # Save reference frames.
    for reference_index in reference_indices:
        # Get reference frame.
        reference_image = load_reference_image(reference_index, reference_images_path)
        reference_frame = float_frame_to_uint8_frame(reference_image)
        # Save frame in .bin file.
        bin_file.append(reference_frame)
        # Save frame as .png file.
        reference_frame_filename = "reference_{:05d}.png".format(reference_index)
        reference_frame_path = os.path.join(frames_path, reference_frame_filename)
        save_frame(reference_frame_path, reference_frame)
    # Save perturbed frames.
    for reference_index in tqdm.tqdm(reference_indices):
        reference_image = load_reference_image(reference_index, reference_images_path)
        for perturbation_pattern_index in perturbation_patterns_indices:
            perturbation_pattern = load_perturbation_pattern(perturbation_pattern_index, perturbation_patterns_path)
            for perturbation_amplitude_index in perturbation_amplitudes_indices:
                # Get perturbed frame.
                perturbation_amplitude = perturbation_amplitudes[perturbation_amplitude_index]
                perturbed_frame = get_perturbed_frame(reference_image, perturbation_pattern, perturbation_amplitude,
                                                      config)
                perturbed_frame = float_frame_to_uint8_frame(perturbed_frame)
                # Save frame in .bin file.
                bin_file.append(perturbed_frame)
                # Save frame as .png file.
                perturbed_frame_filename = "perturbed_r{:05d}_p{:05d}_a{:05d}.png".format(reference_index,
                                                                                          perturbation_pattern_index,
                                                                                          perturbation_amplitude_index)
                perturbed_frame_path = os.path.join(frames_path, perturbed_frame_filename)
                save_frame(perturbed_frame_path, perturbed_frame)
    # TODO check the following lines!
    # Save randomly perturbed frames (if necessary).
    if with_random_patterns:
        for reference_index in tqdm.tqdm(reference_indices):
            reference_image = load_reference_image(reference_index, reference_images_path)
            for perturbation_pattern_index in random_patterns_indices:
                pattern = load_perturbation_pattern(perturbation_pattern_index, perturbation_patterns_path)
                # Get perturbed frame.
                amplitude = float(15) / float(256)  # TODO change this value?
                frame = get_perturbed_frame(reference_image, pattern, amplitude, config)
                frame = float_frame_to_uint8_frame(frame)
                # Save frame in .bin file.
                bin_file.append(frame)
                # Save frame as .png file (if necessary).
                if perturbation_pattern_index < 100:
                    perturbed_frame_filename = "perturbed_r{:05d}_p{:05d}.png".format(reference_index,
                                                                                      perturbation_pattern_index)
                    perturbed_frame_path = os.path.join(frames_path, perturbed_frame_filename)
                    save_frame(perturbed_frame_path, frame)

    bin_file.close()

    print("End creating .bin file.")

    print("Start creating .vec and .csv files...")

    # Create .vec and .csv files.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    csv_filename = "{}.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    csv_file = open_csv_file(csv_path, columns=['k_min', 'k_max', 'combination_id'])
    # Append adaptation.
    grey_frame_id = 0
    for _ in range(0, nb_frame_displays):
        vec_file.append(grey_frame_id)
    # For each repetition...
    for repetition_index in range(0, nb_repetitions):
        # Add frozen patterns.
        combination_indices = permutations[repetition_index]
        for combination_index in combination_indices:
            combination_frame_id = combination_index
            k_min = vec_file.get_display_index() + 1
            # Append trial.
            for _ in range(0, nb_frame_displays):
                vec_file.append(combination_frame_id)
            k_max = vec_file.get_display_index()
            csv_file.append(k_min=k_min, k_max=k_max, combination_id=combination_index)
            # Append intertrial.
            for _ in range(0, nb_frame_displays):
                vec_file.append(grey_frame_id)
        # TODO add a random pattern.
        # Add random patterns (if necessary).
        if with_random_patterns:
            random_combination_indices = random_combination_groups[repetition_index]
            for combination_index in random_combination_indices:
                combination_frame_id = combination_index
                k_min = vec_file.get_display_index() + 1
                # Append trial.
                for _ in range(0, nb_frame_displays):
                    vec_file.append(combination_frame_id)
                k_max = vec_file.get_display_index()
                csv_file.append(k_min=k_min, k_max=k_max, combination_id=combination_index)
                # Append intertrial.
                for _ in range(0, nb_frame_displays):
                    vec_file.append(grey_frame_id)

    csv_file.close()
    vec_file.close()

    print("End creating .vec and .csv files.")

    return
