import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.csv import open_file as open_csv_file
from pystim.utils import handle_arguments_and_configurations
from pystim.utils import shape
from pystim.utils import get_grey_frame
from pystim.utils import float_frame_to_uint8_frame


name = 'euler'

default_configuration = {
    'frame': {
        # 'width': 3024.0,  # µm
        'width': 2688.0,  # µm
        # 'height': 3024.0,  # µm
        'height': 2688.0,  # µm
        'rate': 50.0,  # Hz
    },
    'pattern': {
        'step': {
            'd_ante': 2.0,  # s
            'd': 3.0,  # s
            'd_post': 3.0,  # s
        },
        'frequency_chirp': {
            'd_ante': 2.0,  # s
            'd': 8.0,  # s
            'd_post': 1.0,  # s
            'nb_periods': 32,
        },
        'amplitude_chirp': {
            'd_ante': 1.0,  # s
            'd': 8.0,  # s
            'd_post': 2.0,  # s
            'nb_periods': 16,
        },
        'contrast': 1.0,  # arb. unit
    },
    'initial_adaptation_duration': 5.0,  # s
    'intertrial_duration': 2.0,  # s
    'nb_repetitions': 20,
    'path': os.path.join(tempfile.gettempdir(), "pystim", name)
}


def get_pattern(config, frame_rate):

    step_config = config['step']
    step_d_ante = step_config['d_ante']
    step_d = step_config['d']
    step_d_post = step_config['d_post']
    assert (step_d_ante * frame_rate).is_integer()
    assert (step_d * frame_rate).is_integer()
    assert (step_d_post * frame_rate).is_integer()
    step_duration = step_d_ante + step_d + step_d_post

    frequency_chirp_config = config['frequency_chirp']
    frequency_chirp_d_ante = frequency_chirp_config['d_ante']
    frequency_chirp_d = frequency_chirp_config['d']
    frequency_chirp_d_post = frequency_chirp_config['d_post']
    assert (frequency_chirp_d_ante * frame_rate).is_integer()
    assert (frequency_chirp_d * frame_rate).is_integer()
    assert (frequency_chirp_d_post * frame_rate).is_integer()
    frequency_chirp_duration = frequency_chirp_d_ante + frequency_chirp_d + frequency_chirp_d_post
    frequency_chirp_nb_periods = frequency_chirp_config['nb_periods']

    amplitude_chirp_config = config['amplitude_chirp']
    amplitude_chirp_d_ante = amplitude_chirp_config['d_ante']
    amplitude_chirp_d = amplitude_chirp_config['d']
    amplitude_chirp_d_post = amplitude_chirp_config['d_post']
    assert (amplitude_chirp_d_ante * frame_rate).is_integer()
    assert (amplitude_chirp_d * frame_rate).is_integer()
    assert (amplitude_chirp_d_post * frame_rate).is_integer()
    amplitude_chirp_duration = amplitude_chirp_d_ante + amplitude_chirp_d + amplitude_chirp_d_post
    amplitude_chirp_nb_periods = amplitude_chirp_config['nb_periods']

    duration = step_duration + frequency_chirp_duration + amplitude_chirp_duration

    contrast = config['contrast']

    nb_displays = int(np.round(duration * frame_rate))
    pattern = 0.5 * np.ones(nb_displays, dtype=np.float)

    # Create step.
    i_0 = int(np.round(0.0 * frame_rate))
    i_ante = i_0 + int(np.round(step_d_ante * frame_rate))
    i = i_ante + int(np.round(step_d * frame_rate))
    i_post = i + int(np.round(step_d_post * frame_rate))
    pattern[i_0:i_ante] = 0.5 - contrast / 2.0
    pattern[i_ante:i] = 0.5 + contrast / 2.0
    pattern[i:i_post] = 0.5 - contrast / 2.0

    # Create frequency chirp.
    i_0 = int(np.round(step_duration * frame_rate))
    i_ante = i_0 + int(np.round(frequency_chirp_d_ante * frame_rate))
    i = i_ante + int(np.round(frequency_chirp_d * frame_rate))
    i_post = i + int(np.round(frequency_chirp_d_post * frame_rate))
    t = np.linspace(0.0, frequency_chirp_d, num=i-i_ante, endpoint=False)
    amplitude = 1.0
    omega = 2.0 * np.pi * (np.linspace(0.0, float(frequency_chirp_nb_periods), num=i-i_ante, endpoint=False) / frequency_chirp_d)
    frequency_chirp = 0.5 + 0.5 * amplitude * np.sin(omega * t)
    pattern[i_0:i_ante] = 0.5
    pattern[i_ante:i] = frequency_chirp
    pattern[i:i_post] = 0.5

    # Create amplitude chirp.
    i_0 = int(np.round((step_duration + frequency_chirp_duration) * frame_rate))
    i_ante = i_0 + int(np.round(amplitude_chirp_d_ante * frame_rate))
    i = i_ante + int(np.round(amplitude_chirp_d * frame_rate))
    i_post = i + int(np.round(amplitude_chirp_d_post * frame_rate))
    t = np.linspace(0.0, amplitude_chirp_d, num=i-i_ante, endpoint=False)
    amplitude = np.linspace(0.0, 1.0, num=i-i_ante, endpoint=False)
    omega = 2.0 * np.pi * (float(amplitude_chirp_nb_periods) / amplitude_chirp_d)
    amplitude_chirp = 0.5 + 0.5 * amplitude * np.sin(omega * t)
    pattern[i_0:i_ante] = 0.5
    pattern[i_ante:i] = amplitude_chirp
    pattern[i:i_post] = 0.5

    return pattern


def digitize(pattern):

    dtype = np.uint8
    nb_bins = np.iinfo(dtype).max - np.iinfo(dtype).min + 1
    bins = np.linspace(0.0, 1.0, num=nb_bins, endpoint=False)
    bins = bins[1:]  # remove first element

    indices = np.digitize(pattern, bins)

    return indices


def plot_pattern(pattern, frame_rate):

    x = np.arange(0, len(pattern), dtype=np.float) / frame_rate
    y = pattern

    fig, ax = plt.subplots()
    ax.step(x, y, label='post')
    ax.grid()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("luminance (arb. unit)")
    ax.set_title("Pattern profile")
    fig.tight_layout()

    return fig, ax


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    pixel_size = 3.5  # µm
    # dmd_width = 1920  # px
    # dmd_height = 1080  # px

    frame_width_in_um = config['frame']['width']
    frame_height_in_um = config['frame']['height']
    frame_rate = config['frame']['rate']
    nb_repetitions = config['nb_repetitions']

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)
    print(path)

    dtype = np.uint8
    nb_grey_levels = np.iinfo(dtype).max - np.iinfo(dtype).min + 1
    nb_images = nb_grey_levels

    frame_height_in_px, frame_width_in_px = shape(pixel_size, width=frame_width_in_um, height=frame_height_in_um)

    # Create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images, frame_width=frame_width_in_px, frame_height=frame_height_in_px, mode='w')
    for k in range(0, nb_grey_levels):
        grey_level = float(k) / nb_grey_levels
        frame = get_grey_frame(frame_width_in_px, frame_height_in_px, luminance=grey_level)
        frame = float_frame_to_uint8_frame(frame)
        bin_file.append(frame)
        bin_file.flush()
    bin_file.close()

    pattern_config = config['pattern']
    pattern = get_pattern(pattern_config, frame_rate)
    intertrial_duration = config['intertrial_duration']
    initial_adaptation_duration = config['initial_adaptation_duration']
    assert (intertrial_duration * frame_rate).is_integer()

    # Create .csv file for pattern profile.
    csv_filename = "{}_luminance_profile.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    columns = ['luminance']
    csv_file = open_csv_file(csv_path, columns=columns)
    luminances = digitize(pattern)
    for luminance in luminances:
        csv_file.append(luminance=luminance)
    csv_file.close()

    # Plot pattern profile.
    plot_filename = "{}.pdf".format(name)
    plot_path = os.path.join(path, plot_filename)
    fig, ax = plot_pattern(pattern, frame_rate)
    fig.savefig(plot_path)
    plt.close(fig)

    # nb_displays_per_trial = int(np.round(trial_duration * frame_rate))
    nb_displays_per_trial = pattern.size
    nb_displays_per_intertrial = int(np.round(intertrial_duration * frame_rate))
    nb_displays_in_initial_adaptation = int(np.round(initial_adaptation_duration * frame_rate))

    nb_trials = nb_repetitions
    nb_intertrials = nb_trials
    nb_displays = \
        nb_displays_in_initial_adaptation \
        + nb_trials * nb_displays_per_trial \
        + nb_intertrials * nb_displays_per_intertrial

    frame_indices = digitize(pattern)

    # Create .vec and .csv files.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    csv_filename = "{}.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    csv_file = open_csv_file(csv_path, columns=['k_min', 'k_max', 'combination_id'])
    # Append initial adaptation.
    for _ in range(0, nb_displays_in_initial_adaptation):
        frame_index = 0
        vec_file.append(frame_index)
    # For each repetition...
    for _ in range(0, nb_repetitions):
        k_min = vec_file.get_display_index() + 1
        # Append trial.
        for k in range(0, nb_displays_per_trial):
            frame_id = frame_indices[k]
            vec_file.append(frame_id)
        k_max = vec_file.get_display_index()
        csv_file.append(k_min=k_min, k_max=k_max, combination_id=0)
        # Append intertrial.
        for _ in range(0, nb_displays_per_intertrial):
            frame_id = 0
            vec_file.append(frame_id)
    csv_file.close()
    vec_file.close()

    # Print the total duration.
    duration = float(nb_displays) / frame_rate
    print("duration: {} s".format(duration))
    print("duration: {} min".format(duration / 60.0))

    return
