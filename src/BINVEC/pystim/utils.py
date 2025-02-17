import argparse
import importlib
import json
import numpy as np
import os
import shutil


environment_variable_name = 'PYSTIMPATH'


def list_stimuli():

    stimuli = [
        'test',
        'square',
        'euler',
        'dg',
        'fi',
        'fi_comp',
        'fi_merge',
        'fipwc',
        'fipwfc',
        'fipwrc',
    ]
    # TODO list the stimuli.

    return stimuli


def get_default_configuration_path():

    path = os.path.join("~", ".config", "pystim")
    path = os.path.expanduser(path)

    path = os.getenv(environment_variable_name, path)

    return path


def configure(args):

    _ = args
    path = get_default_configuration_path()
    message = "Default configuration path: {}".format(path)
    print(message)

    return


def initialize(args):

    _ = args

    # Create the default configuration path (if necessary).
    path = get_default_configuration_path()
    if not os.path.isdir(path):
        os.makedirs(path)

    # List the stimuli.
    stimuli = list_stimuli()
    # For each stimulus...
    for stimulus in stimuli:
        # Get the corresponding configuration.
        module_name = 'pystim.{}'.format(stimulus)
        module = importlib.import_module(module_name)
        configuration = module.default_configuration
        # Store the configuration in a file.
        output_filename = "{}.json".format(stimulus)
        output_path = os.path.join(path, output_filename)
        with open(output_path, mode='w') as output_file:
            json.dump(configuration, output_file, indent=4)

    return


def reinitialize(args):

    path = get_default_configuration_path()
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    initialize(args)

    return


def load_configuration(path):

    try:
        with open(path, mode='r') as file:
            configuration = json.load(file)
    except FileNotFoundError:
        configuration = {}

    return configuration


def load_global_configuration(name):

    path = get_default_configuration_path()
    filename = "{}.json".format(name)
    path = os.path.join(path, filename)
    configuration = load_configuration(path)

    return configuration


def load_local_configuration(name):

    path = os.getcwd()
    filename = "{}.json".format(name)
    path = os.path.join(path, filename)
    configuration = load_configuration(path)

    return configuration


def handle_arguments_and_configurations(name, args):

    global_configuration = load_global_configuration(name)
    local_configuration = load_local_configuration(name)
    if isinstance(args, argparse.Namespace):
        arguments = vars(args)
    elif isinstance(args, dict):
        arguments = args
    else:
        raise TypeError("unsupported type for `args`: {}".format(type(args)))

    assert 'func' not in global_configuration
    assert 'func' not in local_configuration

    configuration = {}
    configuration.update(global_configuration)
    configuration.update(local_configuration)
    configuration.update(arguments)  # TODO handle nested fields.

    return configuration


def linspace(pixel_size, width=None, height=None):

    dmd_width_in_px = 1920  # px
    dmd_height_in_px = 1080  # px
    dmd_width_in_um = dmd_width_in_px * pixel_size
    dmd_height_in_um = dmd_height_in_px * pixel_size

    x = np.linspace(0.0, dmd_width_in_um, num=dmd_width_in_px, endpoint=False) + 0.5 * pixel_size - 0.5 * dmd_width_in_um
    y = np.linspace(0.0, dmd_height_in_um, num=dmd_height_in_px, endpoint=False) + 0.5 * pixel_size - 0.5 * dmd_height_in_um

    frame_width = width if width is not None else dmd_width_in_um
    frame_height = height if height is not None else dmd_height_in_um
    frame_horizontal_offset = 0.0  # µm
    frame_vertical_offset = 0.0  # µm

    x_min = frame_horizontal_offset - frame_width / 2.0
    x_max = frame_horizontal_offset + frame_width / 2.0
    y_min = frame_vertical_offset - frame_height / 2.0
    y_max = frame_vertical_offset + frame_height / 2.0

    xm = np.logical_and(x_min <= x, x <= x_max)
    ym = np.logical_and(y_min <= y, y <= y_max)
    x = x[xm]
    y = y[ym]

    return x, y


def shape(pixel_size, width=None, height=None):

    x, y = linspace(pixel_size, width=width, height=height)

    return y.size, x.size


def meshgrid(pixel_size, width=None, height=None):

    x, y = linspace(pixel_size, width=width, height=height)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    return xv, yv


def get_grey_frame(width, height, luminance=0.5):

    shape = (height, width)
    dtype = np.float
    frame = luminance * np.ones(shape, dtype=dtype)

    return frame


# TODO remove the following funciton (deprecated)?
# def float_frame_to_uint8_frame(float_frame):
#
#     dtype = np.uint8
#     dinfo = np.iinfo(dtype)
#     float_frame = float_frame * dinfo.max
#     float_frame[float_frame < dinfo.min] = dinfo.min
#     float_frame[dinfo.max + 1 <= float_frame] = dinfo.max
#     uint8_frame = float_frame.astype(dtype)
#
#     return uint8_frame


def float_frame_to_uint8_frame(float_frame):

    dtype = np.uint8
    min_value = 0.0
    max_value = 254.0

    dinfo = np.iinfo(dtype)
    assert dinfo.min <= int(min_value)
    assert int(max_value) <= dinfo.max

    # Rescale pixel values.
    float_frame = max_value * float_frame
    # Process saturating pixels.
    float_frame[float_frame < min_value] = min_value
    float_frame[max_value < float_frame] = max_value
    # Convert pixel types.
    uint8_frame = float_frame.astype(dtype)

    return uint8_frame


def compute_horizontal_angles(width=600, angular_resolution=1.0):

    x = np.arange(0, width)
    x = x.astype(np.float)
    x -= np.mean(x)
    a_x = angular_resolution * x

    return a_x


def compute_vertical_angles(height=600, angular_resolution=1.0):

    y = np.arange(0, height)
    y = y.astype(np.float)
    y -= np.mean(y)
    a_y = angular_resolution * y

    return a_y
