import itertools
import numpy as np
import os
import tempfile

from PIL.Image import fromarray

from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.csv import open_file as open_csv_file
from pystim.utils import handle_arguments_and_configurations
from pystim.utils import shape, meshgrid


name = 'dg'

default_configuration = {
    'frame': {
        'rate': 50.0,  # Hz
        'width': 3024.0,  # µm
        'height': 3024.0,  # µm
        # 'horizontal_offset': 0.0,  # µm
        # 'vertical_offset': 0.0,  # µm
    },
    'spatial_frequencies': [600.0],  # µm
    'speeds': [450.0],  # µm / s
    'contrasts': [1.0],
    'directions': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # rad
    'trial_duration': 5.0,  # s
    'intertrial_duration': 1.67,  # s
    'nb_repetitions': 20,  # TODO change to 5 (following Tim)?
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
}


def get_combinations(spatial_frequencies, speeds, contrasts, directions):

    sf = np.sort(spatial_frequencies)
    s = np.sort(speeds)
    c = np.sort(contrasts)
    d = np.sort(directions)

    sf_indices = np.arange(0, len(sf))
    s_indices = np.arange(0, len(s))
    c_indices = np.arange(0, len(c))
    d_indices = np.arange(0, len(d))

    combinations = {
        'condition': {
            0: {
                'name': 'spatial_frequency',
                'values': sf,
            },
            1: {
                'name': 'speed',
                'values': s,
            },
            2: {
                'name': 'contrast',
                'values': c,
            },
            3: {
                'name': 'direction',
                'values': d,
            },
        },
        'combination': {
            k: combination
            for k, combination in enumerate(itertools.product(sf_indices, s_indices, c_indices, d_indices))
        }
    }

    return combinations


def get_grey_frame(width, height, luminance=0.5):

    shape = (height, width)
    dtype = np.float
    frame = luminance * np.ones(shape, dtype=dtype)

    return frame


def float_frame_to_uint8_frame(float_frame):

    dtype = np.uint8
    dinfo = np.iinfo(dtype)
    float_frame = float_frame * dinfo.max
    float_frame[float_frame < dinfo.min] = dinfo.min
    float_frame[dinfo.max + 1 <= float_frame] = dinfo.max
    uint8_frame = float_frame.astype(dtype)

    return uint8_frame


def save_frame(path, frame):

    image = fromarray(frame)
    image.save(path)

    return


def get_frame(frame_id, pixel_size, direction=0.0, spatial_frequency=600.0, contrast=1.0, speed=450.0, width=None, height=None, rate=60.0):

    xv, yv = meshgrid(pixel_size, width=width, height=height)
    pv = xv + yv * 1j

    angle = direction * np.pi
    t = float(frame_id) / rate
    d = speed * t
    u = np.exp(1j * angle)  # TODO handle special cases (e.g. horizontal, vertical).
    dv = pv * u
    theta = (dv.real - d) / spatial_frequency
    frame = 0.5 + (contrast / 2.0) * np.ones_like(theta)
    mask = (theta % 1.0) >= 0.5
    frame[mask] = 0.5 - (contrast / 2.0)

    return frame


def get_permutations(indices, nb_repetitions=1, seed=42):

    np.random.seed(seed)

    permutations = {
        k: np.random.permutation(indices)
        for k in range(0, nb_repetitions)
    }

    return permutations


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    # Experimental rig parameters.
    pixel_size = 3.5  # µm

    # Display parameters.
    frame_rate = config['frame']['rate']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']

    # Stimulus parameters.
    spatial_frequencies = config['spatial_frequencies']
    speeds = config['speeds']
    contrasts = config['contrasts']
    directions = config['directions']
    trial_duration = config['trial_duration']
    intertrial_duration = config['intertrial_duration']
    nb_repetitions = config['nb_repetitions']

    path = config['path']
    if not os.path.isdir(path):
        os.makedirs(path)
    frames_path = os.path.join(path, "frames")
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)

    for condition_name in ['spatial_frequencies', 'speeds', 'contrasts', 'directions']:
        csv_filename = "{}_{}.csv".format(name, condition_name)
        csv_path = os.path.join(path, csv_filename)
        csv_file = open_csv_file(csv_path, columns=[condition_name])
        for condition_value in config[condition_name]:
            csv_file.append(**{condition_name: condition_value})
        csv_file.close()

    frame_height_in_px, frame_width_in_px = shape(pixel_size, width=frame_width, height=frame_height)
    print(frame_height_in_px)
    print(frame_width_in_px)

    # Get combinations.
    combinations = get_combinations(spatial_frequencies, speeds, contrasts, directions)
    nb_combinations = len(combinations['combination'])

    # Create .csv file for combinations.
    csv_filename = "{}_combinations.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    columns = [
        "{}_id".format(condition['name'])
        for condition in combinations['condition'].values()
    ]
    csv_file = open_csv_file(csv_path, columns=columns)
    for combination_index in combinations['combination']:
        kwargs = {
            '{}_id'.format(combinations['condition'][condition_id]['name']): combinations['combination'][combination_index][condition_id]
            for condition_id in combinations['condition']
        }
        csv_file.append(**kwargs)
    csv_file.close()

    nb_trials = nb_combinations * nb_repetitions
    stimulus_duration = nb_trials * trial_duration + (nb_trials - 1) * intertrial_duration
    print("stimulus durations: {} s ({} min)".format(stimulus_duration, stimulus_duration / 60.0))
    # TODO improve feedback.

    nb_images_per_trial = int(trial_duration * frame_rate)
    nb_images = 1 + nb_images_per_trial * nb_combinations
    print("nb_images: {}".format(nb_images))

    # Get permutations.
    combination_indices = list(combinations['combination'].keys())
    permutations = get_permutations(combination_indices, nb_repetitions=nb_repetitions)

    # Create .bin file.
    bin_filename = "{}.bin".format(name)
    bin_path = os.path.join(path, bin_filename)
    bin_file = open_bin_file(bin_path, nb_images, frame_width=frame_width_in_px, frame_height=frame_height_in_px)
    # Get grey frame.
    grey_frame = get_grey_frame(frame_width_in_px, frame_height_in_px)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    print(grey_frame.shape)
    # Save frame in .bin file.
    bin_file.append(grey_frame)
    bin_file.flush()
    # Save frame as .png file.
    grey_frame_filename = "grey.png"
    grey_frame_path = os.path.join(frames_path, grey_frame_filename)
    save_frame(grey_frame_path, grey_frame)
    for combination_index in combinations['combination']:
        combination = combinations['combination'][combination_index]
        sf_index = combination[0]
        sf = combinations['condition'][0]['values'][sf_index]
        s_index = combination[1]
        s = combinations['condition'][1]['values'][s_index]
        c_index = combination[2]
        c = combinations['condition'][2]['values'][c_index]
        d_index = combination[3]
        d = combinations['condition'][3]['values'][d_index]
        for frame_id in range(0, nb_images_per_trial):
            frame = get_frame(frame_id, pixel_size, spatial_frequency=sf, speed=s, contrast=c, direction=d, width=frame_width, height=frame_height, rate=frame_rate)
            frame = float_frame_to_uint8_frame(frame)
            # Save frame in .bin file.
            print(frame.shape)
            bin_file.append(frame)
            bin_file.flush()
            # Save frame as .png file.
            frame_number = 1 + combination_index * nb_images_per_trial + frame_id
            frame_filename = "frame_{:05d}.png".format(frame_number)
            frame_path = os.path.join(frames_path, frame_filename)
            save_frame(frame_path, frame)
    bin_file.close()

    nb_displays_per_trial = int(np.round(trial_duration * frame_rate))
    nb_displays_per_intertrial = int(np.round(intertrial_duration * frame_rate))

    nb_trials = 1 + nb_combinations * nb_repetitions
    nb_intertrials = 1 + nb_combinations * nb_repetitions  # i.e. one after each trial

    nb_displays = nb_displays_per_trial * nb_trials + nb_displays_per_intertrial * nb_intertrials

    # Create .vec and .csv file.
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(path, vec_filename)
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    csv_filename = "{}.csv".format(name)
    csv_path = os.path.join(path, csv_filename)
    csv_file = open_csv_file(csv_path, columns=['k_min', 'k_max', 'combination_id', 'repetition_id'])
    # Append initial trial.
    frame_id = 0  # grey
    for _ in range(0, nb_displays_per_trial):
        vec_file.append(frame_id)
    # Append intertrial.
    frame_id = 0
    for _ in range(0, nb_displays_per_intertrial):
        vec_file.append(frame_id)
    # Append repetitions.
    for repetition_index in range(0, nb_repetitions):
        combination_indices = permutations[repetition_index]
        # Append combination.
        for combination_index in combination_indices:
            # Append trial.
            k_min = vec_file.get_display_index() + 1
            for l in range(0, nb_images_per_trial):
                frame_id = 1 + combination_index * nb_images_per_trial + l
                vec_file.append(frame_id)
            k_max = vec_file.get_display_index()
            csv_file.append(k_min=k_min, k_max=k_max, combination_id=combination_index, repetition_id=repetition_index)
            # Append intertrial.
            frame_id = 0
            for _ in range(0, nb_displays_per_intertrial):
                vec_file.append(frame_id)
    csv_file.close()
    vec_file.close()

    return
