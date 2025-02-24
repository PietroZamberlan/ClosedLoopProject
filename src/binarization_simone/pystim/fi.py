import math
import numpy as np
import os
import scipy as sp
import scipy.interpolate
import tempfile
import tqdm
import warnings

import pystim.datasets.van_hateren as vh

from pystim.images.png import create as create_png_image
from pystim.images.png import load as load_png_image
from pystim.io.bin import open_file as open_bin_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.csv import open_file as open_csv_file
from pystim.utils import compute_horizontal_angles
from pystim.utils import compute_vertical_angles
from pystim.utils import float_frame_to_uint8_frame
from pystim.utils import get_grey_frame
from pystim.utils import handle_arguments_and_configurations


name = 'fi'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), "pystim", name),
    'vh_image_nbs': None,
    # 'vh_image_nbs': list(range(1, 10)),
    'eye_diameter': 1.2e-2,  # m
    # 'eye_diameter': 1.2e-2,  # m  # human
    # 'eye_diameter': 2.7e-3,  # m  # axolotl
    'mean_luminance': 0.25,
    'std_luminance': 0.05,
    'normalized_value_median': 0.5,  # TODO remove (deprecated)?
    'normalized_value_mad': 0.01,  # TODO remove (deprecated)?
    'display_rate': 40.0,  # Hz
    # 'adaptation_duration': 0.1,  # s
    'adaptation_duration': 60.0,  # s
    # 'flash_duration': 0.1,  # s
    'flash_duration': 0.3,  # s
    # 'inter_flash_duration': 0.1,  # s
    'inter_flash_duration': 0.3,  # s
    'frame': {
        'width': 864,  # px
        'height': 864,  # px
        # 'duration': 0.3,  # s  # TODO remove (deprecated)?
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    # 'stuttering_vh_image_nbs': [1, 3],
    'stuttering_vh_image_nbs': [5, 31, 2219],  # 46 is saturated
    # 'nb_stuttering_vh_images': 2,
    'nb_stuttering_vh_images': 30,
    # 'nb_stutters': 2,
    'nb_stutters': 20,
    # 'nb_repetitions': 2,
    'nb_repetitions': 1,
    'seed': 42,
    'verbose': True,
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    base_path = config['path']
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    print("Generation in: {}".format(base_path))

    images_path = os.path.join(base_path, "images")
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    # Get configuration parameters.
    vh_image_nbs = config['vh_image_nbs']
    eye_diameter = config['eye_diameter']
    mean_luminance = config['mean_luminance']
    std_luminance = config['std_luminance']
    normalized_value_median = config['normalized_value_median']
    normalized_value_mad = config['normalized_value_mad']
    display_rate = config['display_rate']
    adaptation_duration = config['adaptation_duration']
    flash_duration = config['flash_duration']
    inter_flash_duration = config['inter_flash_duration']
    frame_resolution = config['frame']['resolution']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    nb_repetitions = config['nb_repetitions']
    stuttering_vh_image_nbs = config['stuttering_vh_image_nbs']
    nb_stuttering_vh_images = config['nb_stuttering_vh_images']
    nb_stutters = config['nb_stutters']
    seed = config['seed']
    verbose = config['verbose']

    # Fetch van Hateren images.
    vh.fetch(image_nbs=vh_image_nbs, download_if_missing=False, verbose=verbose)

    # Select unsaturated van Hateren image numbers.
    if vh_image_nbs is None:
        vh_image_nbs = vh.get_image_nbs()
    else:
        vh_image_nbs = np.array(vh_image_nbs)
    are_saturated = vh.get_are_saturated(image_nbs=vh_image_nbs, verbose=verbose)
    are_unsaturated = np.logical_not(are_saturated)
    assert vh_image_nbs.size == are_unsaturated.size, "{} != {}".format(vh_image_nbs.size, are_unsaturated.size)
    unsaturated_vh_image_nbs = vh_image_nbs[are_unsaturated]

    # Select ...
    # mean_luminances = vh.get_mean_luminances(image_nbs=unsaturated_vh_image_nbs, verbose=verbose)
    # std_luminances = vh.get_std_luminances(image_nbs=unsaturated_vh_image_nbs, verbose=verbose)
    max_luminances = vh.get_max_luminances(image_nbs=unsaturated_vh_image_nbs, verbose=verbose)
    # TODO remove the following lines?
    # max_centered_luminances = max_luminances / mean_luminances
    # # print(np.min(max_centered_luminances))
    # # print(np.median(max_centered_luminances))
    # # print(np.max(max_centered_luminances))
    # are_good = max_centered_luminances <= 8.3581  # TODO correct?
    # TODO remove the following lines?
    # max_normalized_luminances = (max_luminances / mean_luminances - 1.0) / std_luminances + 1.0
    # are_good = max_normalized_luminances <= 1.02
    # TODO remove the following lines?
    log_mean_luminances = vh.get_log_mean_luminances(image_nbs=unsaturated_vh_image_nbs, verbose=verbose)
    log_std_luminances = vh.get_log_mean_luminances(image_nbs=unsaturated_vh_image_nbs, verbose=verbose)
    log_max_luminances = np.log(1.0 + max_luminances)
    log_max_normalized_luminances = (log_max_luminances - log_mean_luminances) / log_std_luminances
    are_good = log_max_normalized_luminances <= 5.0
    # ...
    good_vh_image_nbs = unsaturated_vh_image_nbs[are_good]
    # ...
    # selected_vh_image_nbs = unsaturated_vh_image_nbs
    selected_vh_image_nbs = good_vh_image_nbs

    # Check stuttering van Hateren image numbers.
    np.random.seed(seed)
    if stuttering_vh_image_nbs is None:
        stuttering_vh_image_nbs = np.array([])
    else:
        assert len(stuttering_vh_image_nbs) <= nb_stuttering_vh_images
        for stuttering_vh_image_nb in stuttering_vh_image_nbs:
            assert stuttering_vh_image_nb in selected_vh_image_nbs, stuttering_vh_image_nb
    potential_stuttering_vh_image_nbs = np.setdiff1d(selected_vh_image_nbs, stuttering_vh_image_nbs, assume_unique=True)
    nb_missing_stuttering_vh_image_nbs = nb_stuttering_vh_images - len(stuttering_vh_image_nbs)
    stuttering_vh_image_nbs = np.concatenate((
        stuttering_vh_image_nbs,
        np.random.choice(potential_stuttering_vh_image_nbs, nb_missing_stuttering_vh_image_nbs, replace=False)
    ))
    stuttering_vh_image_nbs.sort()

    # Generate grey image.
    image_filename = "image_{i:04d}.png".format(i=0)
    image_path = os.path.join(images_path, image_filename)
    # if not os.path.isfile(image_path):  # TODO uncomment this line.
    frame = get_grey_frame(frame_width, frame_height, luminance=mean_luminance)
    frame = float_frame_to_uint8_frame(frame)
    image = create_png_image(frame)
    image.save(image_path)

    # Extract images from the van Hateren images.
    for vh_image_nb in tqdm.tqdm(selected_vh_image_nbs):
        # Check if image already exists.
        image_filename = "image_{i:04d}.png".format(i=vh_image_nb)
        image_path = os.path.join(images_path, image_filename)
        if os.path.isfile(image_path):
            continue
        # Cut out central sub-region.
        a_x = vh.get_horizontal_angles()
        a_y = vh.get_vertical_angles()
        luminance_data = vh.load_luminance_data(vh_image_nb)
        rbs = sp.interpolate.RectBivariateSpline(a_x, a_y, luminance_data, kx=1, ky=1)
        angular_resolution = math.atan(frame_resolution / eye_diameter) * (180.0 / math.pi)
        a_x = compute_horizontal_angles(width=frame_width, angular_resolution=angular_resolution)
        a_y = compute_vertical_angles(height=frame_height, angular_resolution=angular_resolution)
        a_x, a_y = np.meshgrid(a_x, a_y)
        luminance_data = rbs.ev(a_x, a_y)
        luminance_data = luminance_data.transpose()
        # TODO uncomment the following lines?
        # # Prepare data.
        # luminance_median = np.median(luminance_data)
        # luminance_mad = np.median(np.abs(luminance_data - luminance_median))
        # centered_luminance_data = luminance_data - luminance_median
        # if luminance_mad > 0.0:
        #     centered_n_reduced_luminance_data = centered_luminance_data / luminance_mad
        # else:
        #     centered_n_reduced_luminance_data = centered_luminance_data
        # normalized_luminance_data = centered_n_reduced_luminance_data
        # scaled_data = normalized_value_mad * normalized_luminance_data
        # shifted_n_scaled_data = scaled_data + normalized_value_median
        # data = shifted_n_scaled_data
        # TODO remove the 2 following lines?
        # scaled_luminance_data = luminance_data / np.mean(luminance_data)
        # data = scaled_luminance_data / 8.3581
        # TODO remove the 2 following lines?
        # normalized_luminance_data = (luminance_data / np.mean(luminance_data) - 1.0) / np.std(luminance_data) + 1.0
        # data = (normalized_luminance_data - 1.0) / 0.02 * 0.8 + 0.2
        # TODO remove the following lines?
        log_luminance_data = np.log(1.0 + luminance_data)
        log_mean_luminance = np.mean(log_luminance_data)
        log_std_luminance = np.std(log_luminance_data)
        normalized_log_luminance_data = log_luminance_data - log_mean_luminance
        if log_std_luminance > 1e-13:
            normalized_log_luminance_data = normalized_log_luminance_data / log_std_luminance
        normalized_log_luminance_data = 0.2 * normalized_log_luminance_data
        normalized_luminance_data = np.exp(normalized_log_luminance_data) - 1.0
        normalized_luminance_data = normalized_luminance_data - np.mean(normalized_luminance_data)
        if np.std(normalized_luminance_data) > 1e-13:
            normalized_luminance_data = normalized_luminance_data / np.std(normalized_luminance_data)
        data = std_luminance * normalized_luminance_data + mean_luminance
        # Prepare image data
        if np.count_nonzero(data < 0.0) > 0:
            s = "some pixels are negative in image {} (consider changing the configuration, 'normalized_value_mad': {})"
            message = s.format(vh_image_nb, normalized_value_mad /
                               ((normalized_value_median - np.min(data)) / (normalized_value_median - 0.0)))
            warnings.warn(message)
        data[data < 0.0] = 0.0
        if np.count_nonzero(data > 1.0) > 0:
            s = "some pixels saturate in image {} (consider changing the configuration, 'normalized_value_mad': {})"
            message = s.format(vh_image_nb, normalized_value_mad /
                               ((np.max(data) - normalized_value_median) / (1.0 - normalized_value_median)))
            warnings.warn(message)
        data[data > 1.0] = 1.0
        data = np.array(254.0 * data, dtype=np.uint8)  # 0.0 -> 0 and 1.0 -> 254 such that 0.5 -> 127
        data = np.transpose(data)
        data = np.flipud(data)
        image = create_png_image(data)
        # Save image.
        image.save(image_path)

    # Set condition numbers and image paths.
    condition_nbs = []
    stuttering_condition_nbs = []
    image_paths = {}
    for k, vh_image_nb in enumerate(selected_vh_image_nbs):
        # condition_nb = k + 1
        condition_nb = k
        image_filename = 'image_{i:04d}.png'.format(i=vh_image_nb)
        image_path = os.path.join(images_path, image_filename)
        assert condition_nb not in condition_nbs
        assert os.path.isfile(image_path)
        condition_nbs.append(condition_nb)
        if vh_image_nb in stuttering_vh_image_nbs:
            stuttering_condition_nbs.append(condition_nb)
        image_paths[condition_nb] = image_path
    condition_nbs = np.array(condition_nbs)
    stuttering_condition_nbs = np.array(stuttering_condition_nbs)
    nb_conditions = len(condition_nbs)
    nb_stuttering_conditions = len(stuttering_condition_nbs)

    # Create conditions .csv file.
    conditions_csv_filename = '{}_conditions.csv'.format(name)
    conditions_csv_path = os.path.join(base_path, conditions_csv_filename)
    print("Start creating conditions .csv file...")
    # Open conditions .csv file.
    conditions_csv_file = open_csv_file(conditions_csv_path, columns=['path'])
    # Add conditions for van Hateren images.
    for condition_nb in condition_nbs:
        image_path = image_paths[condition_nb]
        path = image_path.replace(base_path, '')
        path = path[1:]  # remove separator
        conditions_csv_file.append(path=path)
    # Close conditions .csv file.
    conditions_csv_file.close()
    # ...
    print("End of conditions .csv file creation.")

    # Set sequence of conditions for each repetition.
    repetition_sequences = {}
    np.random.seed(seed)
    normal_condition_nbs = np.setdiff1d(condition_nbs, stuttering_condition_nbs, assume_unique=True)
    nb_normal_indices = len(normal_condition_nbs)
    nb_stuttering_indices = nb_stuttering_conditions * nb_stutters
    nb_indices = nb_normal_indices + nb_stuttering_indices
    sequence = np.empty(nb_indices, dtype=np.int)
    stuttering_indices = np.linspace(0, nb_indices, num=nb_stuttering_indices, endpoint=False)
    stuttering_indices = stuttering_indices.astype(np.int)
    normal_indices = np.setdiff1d(np.arange(0, nb_indices), stuttering_indices)
    sequence[stuttering_indices] = np.concatenate(tuple([
        stuttering_condition_nbs
        for _ in range(0, nb_stutters)
    ]))
    for repetition_nb in range(0, nb_repetitions):
        repetition_sequence = np.copy(sequence)
        # Normal.
        repetition_normal_sequence = np.copy(normal_condition_nbs)
        np.random.shuffle(repetition_normal_sequence)
        repetition_sequence[normal_indices] = repetition_normal_sequence
        # Stuttering.
        repetition_stuttering_sequence = []
        for _ in range(0, nb_stutters):
            repetition_stuttering_condition_nbs = np.copy(stuttering_condition_nbs)
            np.random.shuffle(repetition_stuttering_condition_nbs)
            repetition_stuttering_sequence.append(repetition_stuttering_condition_nbs)
        repetition_stuttering_sequence = np.concatenate(tuple(repetition_stuttering_sequence))
        repetition_sequence[stuttering_indices] = repetition_stuttering_sequence
        # ...
        repetition_sequences[repetition_nb] = repetition_sequence

    # Create .bin file.
    print("Start creating .bin file...")
    bin_filename = '{}.bin'.format(name)
    bin_path = os.path.join(base_path, bin_filename)
    nb_bin_images = 1 + nb_conditions  # i.e. grey image and other conditions
    bin_frame_nbs = {}
    # Open .bin file.
    bin_file = open_bin_file(bin_path, nb_bin_images, frame_width=frame_width,
                             frame_height=frame_height, reverse=False, mode='w')
    # Add grey frame.
    grey_frame = get_grey_frame(frame_width, frame_height, luminance=mean_luminance)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    bin_file.append(grey_frame)
    bin_frame_nbs[None] = bin_file.get_frame_nb()
    # Add van Hateren frames.
    for condition_nb in tqdm.tqdm(condition_nbs):
        image_path = image_paths[condition_nb]
        image = load_png_image(image_path)
        frame = image.data
        bin_file.append(frame)
        bin_frame_nbs[condition_nb] = bin_file.get_frame_nb()
    # Close .bin file.
    bin_file.close()
    # ...
    print("End of .bin file creation.")

    # Create .vec file and stimulation .csv file.
    print("Start creating .vec file...")
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(base_path, vec_filename)
    csv_filename = "{}_trials.csv".format(name)
    csv_path = os.path.join(base_path, csv_filename)
    # ...
    nb_displays_during_adaptation = int(np.ceil(adaptation_duration * display_rate))
    nb_displays_per_flash = int(np.ceil(flash_duration * display_rate))
    nb_displays_per_inter_flash = int(np.ceil(inter_flash_duration * display_rate))
    nb_flashes_per_repetition = nb_conditions + nb_stuttering_vh_images * (nb_stutters - 1)
    nb_displays_per_repetition = nb_flashes_per_repetition * (nb_displays_per_flash + nb_displays_per_inter_flash)
    nb_displays = nb_displays_during_adaptation + nb_repetitions * nb_displays_per_repetition
    # Open .vec file.
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    # Open .csv file.
    csv_file = open_csv_file(csv_path, columns=['condition_nb', 'start_display_nb', 'end_display_nb'])
    # Add adaptation.
    bin_frame_nb = bin_frame_nbs[None]  # i.e. default frame (grey)
    for _ in range(0, nb_displays_during_adaptation):
        vec_file.append(bin_frame_nb)
    # Add repetitions.
    for repetition_nb in tqdm.tqdm(range(0, nb_repetitions)):
        repetition_sequence = repetition_sequences[repetition_nb]
        for condition_nb in repetition_sequence:
            # Add flash.
            start_display_nb = vec_file.get_display_nb() + 1
            bin_frame_nb = bin_frame_nbs[condition_nb]
            for _ in range(0, nb_displays_per_flash):
                vec_file.append(bin_frame_nb)
            end_display_nb = vec_file.get_display_nb()
            csv_file.append(condition_nb=condition_nb, start_display_nb=start_display_nb, end_display_nb=end_display_nb)
            # Add inter flash.
            bin_frame_nb = bin_frame_nbs[None]  # i.e. default frame (grey)
            for _ in range(0, nb_displays_per_inter_flash):
                vec_file.append(bin_frame_nb)
    # Close .csv file.
    csv_file.close()
    # Close .vec file.
    vec_file.close()
    # ...
    print("End of .vec file creation.")

    return
