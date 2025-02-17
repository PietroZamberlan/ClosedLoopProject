import numpy as np
import os
import tempfile

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.utils import handle_arguments_and_configurations
from pystim.utils import linspace, shape
from pystim.utils import get_grey_frame
from pystim.utils import float_frame_to_uint8_frame

name = 'square'

default_configuration = {
    'frame': {
        'width': 3024.00,  # µm
        'height': 3024.00,  # µm
        'rate': 50.0,  # Hz
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    # 'background_luminance': 0.5,
    'background_luminance': 0.0,
    'size': float(16) * 30.0,  # µm
    # 'duration': 2.0,  # s
    'duration': 20.0,  # s
    'nb_repetitions': 60,
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name)
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    frame_width_in_um = config['frame']['width']
    frame_height_in_um = config['frame']['height']
    frame_rate = config['frame']['rate']
    frame_resolution = config['frame']['resolution']
    background_luminance = config['background_luminance']
    size = config['size']
    duration = config['duration']
    nb_repetitions = config['nb_repetitions']
    path = config['path']

    pixel_size = frame_resolution * 1e+6  # µm

    # Check duration.
    assert (duration * frame_rate).is_integer()
    assert int(duration * frame_rate) % 2 == 0

    # Create output directory (if necessary).
    if not os.path.isdir(path):
        os.makedirs(path)
    print(path)

    # Collect frame parameters.
    frame_height_in_px, frame_width_in_px = shape(pixel_size, width=frame_width_in_um, height=frame_height_in_um)
    x, y = linspace(pixel_size, width=frame_width_in_um, height=frame_height_in_um)
    xm = np.logical_and(- size / 2.0 <= x, x <= + size / 2.0)
    ym = np.logical_and(- size / 2.0 <= y, y <= + size / 2.0)
    i = np.nonzero(xm)[0]
    j = np.nonzero(ym)[0]
    i_min, i_max = i[0], i[-1]
    j_min, j_max = j[0], j[-1]
    # Create white frame.
    white_frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=background_luminance)
    white_frame[j_min:j_max+1, i_min:i_max+1] = 1.0
    white_frame = float_frame_to_uint8_frame(white_frame)
    # Create black frame.
    black_frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=background_luminance)
    black_frame[j_min:j_max+1, i_min:i_max+1] = 0.0
    black_frame = float_frame_to_uint8_frame(black_frame)

    nb_images = 2

    # Create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(
        bin_path, nb_images, frame_width=frame_width_in_px, frame_height=frame_height_in_px, mode='w'
    )
    bin_file.append(black_frame)
    bin_file.append(white_frame)
    bin_file.close()

    nb_displays_per_repetition = int(duration * frame_rate)
    nb_displays = nb_displays_per_repetition * nb_repetitions

    # Create .vec file.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays)
    for _ in range(0, nb_repetitions):
        for _ in range(0, nb_displays_per_repetition // 2):
            frame_index = 1  # i.e.white
            vec_file.append(frame_index)
        for _ in range(0, nb_displays_per_repetition // 2):
            frame_index = 0  # i.e. black
            vec_file.append(frame_index)
    vec_file.close()

    return
