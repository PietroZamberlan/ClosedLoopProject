"""Flashed images composition"""

import collections
import numpy as np
import os
import tempfile
import tqdm

# from pystim.images.png import load as load_png_image
from pystim.io.bin import open_file as open_bin_file
from pystim.io.csv import load_file as load_csv_file
from pystim.io.csv import open_file as open_csv_file
from pystim.io.vec import load_file as load_vec_file
from pystim.io.vec import open_file as open_vec_file
from pystim.utils import float_frame_to_uint8_frame
from pystim.utils import get_grey_frame
from pystim.utils import handle_arguments_and_configurations


name = 'fi_comp'

default_configuration = {
    'path': os.path.join(tempfile.gettempdir(), 'pystim', name),
    'stimuli': collections.OrderedDict([
        (0, os.path.join(tempfile.gettempdir(), 'pystim', 'fipwfc')),
        (1, os.path.join(tempfile.gettempdir(), 'pystim', 'fipwrc')),
        (2, os.path.join(tempfile.gettempdir(), 'pystim', 'fi')),
    ]),
    'mean_luminance': 0.25,
    'display_rate': 40.0,  # Hz
    'adaptation_duration': 5.0,  # s
    # 'adaptation_duration': 60.0,  # s
    # 'flash_duration': 10.0,  # s
    'flash_duration': 0.3,  # s
    # 'inter_flash_duration': 1.0,  # s
    'inter_flash_duration': 0.3,  # s
    'frame': {
        'width': 864,  # px
        'height': 864,  # px
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    'nb_repetitions': 1,
    'seed': 42,
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    base_path = config['path']
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    print("Generation in {}.".format(base_path))

    # Get configuration parameters.
    stimuli_dirnames = config['stimuli']
    mean_luminance = config['mean_luminance']
    display_rate = config['display_rate']
    adaptation_duration = config['adaptation_duration']
    flash_duration = config['flash_duration']
    inter_flash_duration = config['inter_flash_duration']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    # nb_repetitions = config['nb_repetitions']  # TODO remove?
    seed = config['seed']

    # ...
    stimulus_nbs = []
    stimuli_params = {}
    for stimulus_nb, stimulus_dirname in stimuli_dirnames.items():
        stimulus_nb = int(stimulus_nb)  # TODO improve?
        assert os.path.isdir(stimulus_dirname), stimulus_dirname
        stimulus_nbs.append(stimulus_nb)
        stimuli_params[stimulus_nb] = {
            'dirname': stimulus_dirname,
            'name': os.path.split(stimulus_dirname)[-1],
        }
    stimulus_nbs = np.array(stimulus_nbs)

    # Set conditions parameters.
    condition_nb = 0
    condition_params = collections.OrderedDict()
    for stimulus_nb in stimulus_nbs:
        assert condition_nb not in condition_params
        condition_params[condition_nb] = collections.OrderedDict([
            ('stimulus_nb', stimulus_nb)
        ])
        condition_nb += 1
    condition_nbs = np.array(list(condition_params))
    nb_conditions = len(condition_nbs)
    _ = nb_conditions

    # ...
    stimuli_nb_trials = {}
    stimuli_condition_nbs = {}  # stimulus_nb, trial_nb -> condition_nb
    stimuli_bin_frame_nbs = {  # stimulus_nb, condition_nb -> bin_frame_nb
        None: 0,  # i.e. inter-flash frame (grey frame)
    }
    for stimulus_nb in stimulus_nbs:
        stimulus_params = stimuli_params[stimulus_nb]
        # ...
        stimulus_trial_csv_dirname = stimulus_params['dirname']
        stimulus_trial_csv_filename = '{}_trials.csv'.format(stimulus_params['name'])
        stimulus_trial_csv_path = os.path.join(stimulus_trial_csv_dirname, stimulus_trial_csv_filename)
        stimulus_trials = load_csv_file(stimulus_trial_csv_path)
        # ...
        stimulus_condition_nbs = {}  # trial_nb -> condition_nb
        stimulus_start_frame_nbs = {}  # condition_nb -> start_frame_nb
        for trial_nb, stimulus_trial in stimulus_trials.iterrows():
            stimulus_condition_nbs[trial_nb] = stimulus_trial['condition_nb']
            stimulus_start_frame_nbs[trial_nb] = stimulus_trial['start_frame_nb']
        # ...
        stimulus_vec_dirname = stimulus_params['dirname']
        stimulus_vec_filename = '{}.vec'.format(stimulus_params['name'])
        stimulus_vec_path = os.path.join(stimulus_vec_dirname, stimulus_vec_filename)
        stimulus_vec = load_vec_file(stimulus_vec_path)
        # ...
        stimulus_bin_frame_nbs = {}
        nb_trials = len(stimulus_trials)
        for trial_nb in range(0, nb_trials):
            condition_nb = stimulus_condition_nbs[trial_nb]
            start_frame_nb = stimulus_start_frame_nbs[condition_nb]
            if condition_nb not in stimulus_bin_frame_nbs:
                stimulus_bin_frame_nbs[condition_nb] = stimulus_vec[start_frame_nb]
            else:
                assert stimulus_bin_frame_nbs[condition_nb] == stimulus_vec[start_frame_nb]
        # ...
        stimuli_nb_trials[stimulus_nb] = len(stimulus_trials)
        stimuli_condition_nbs[stimulus_nb] = stimulus_condition_nbs
        stimuli_bin_frame_nbs[stimulus_nb] = stimulus_bin_frame_nbs

    stimulus_sequence = np.concatenate(tuple([
        np.repeat(stimulus_nb, len(stimuli_condition_nbs[stimulus_nb]))
        for stimulus_nb in stimulus_nbs
    ]))
    np.random.seed(seed)
    np.random.shuffle(stimulus_sequence)
    stimuli_indices = {
        stimulus_nb: np.where(stimulus_sequence == stimulus_nb)[0]
        for stimulus_nb in stimulus_nbs
    }

    trials = {}  # trial_nb -> stimulus_nb, condition_nb
    trial_nb = 0
    ordering = np.empty_like(stimulus_sequence, dtype=np.int)
    for stimulus_nb in stimulus_nbs:
        stimulus_condition_nbs = stimuli_condition_nbs[stimulus_nb]
        stimulus_indices = stimuli_indices[stimulus_nb]
        for condition_nb, stimulus_index in zip(stimulus_condition_nbs.values(), stimulus_indices):
            trials[trial_nb] = (stimulus_nb, condition_nb)
            ordering[stimulus_index] = trial_nb
            trial_nb += 1
    nb_trials = len(trials)

    # # Set ordering.
    # np.random.seed(seed)
    # trial_nbs = np.arange(0, nb_trials)
    # ordering = np.copy(trial_nbs)
    # np.random.shuffle(ordering)

    # Create conditions .csv file.
    # TODO complete.

    # Get number of images in .bin files.
    stimuli_nb_bin_images = {}
    for stimulus_nb in stimulus_nbs:
        stimulus_params = stimuli_params[stimulus_nb]
        stimulus_bin_dirname = stimulus_params['dirname']
        stimulus_bin_filename = '{}.bin'.format(stimulus_params['name'])
        stimulus_bin_path = os.path.join(stimulus_bin_dirname, stimulus_bin_filename)
        stimulus_bin_file = open_bin_file(stimulus_bin_path, mode='r')
        stimuli_nb_bin_images[stimulus_nb] = stimulus_bin_file.nb_frames

    # TODO Map stimulus bin frame numbers to bin frame numbers.
    bin_frame_nbs = {
        None: 0,
    }
    bin_frame_nb_offset = 1
    for stimulus_nb in stimulus_nbs:
        bin_frame_nbs[stimulus_nb] = {}
        stimulus_bin_frame_nbs = stimuli_bin_frame_nbs[stimulus_nb]
        for condition_nb in stimulus_bin_frame_nbs.keys():
            stimulus_bin_frame_nb = stimulus_bin_frame_nbs[condition_nb]
            bin_frame_nbs[stimulus_nb][condition_nb] = stimulus_bin_frame_nb + bin_frame_nb_offset
        bin_frame_nb_offset += stimuli_nb_bin_images[stimulus_nb]

    # Create .bin file.
    bin_filename = '{}.bin'.format(name)
    bin_path = os.path.join(base_path, bin_filename)
    nb_bin_images = 1 + int(np.sum([n for n in stimuli_nb_bin_images.values()]))
    # Open .bin file.
    bin_file = open_bin_file(bin_path, nb_bin_images, frame_width=frame_width, frame_height=frame_height, reverse=False, mode='w')
    # Add grey frame.
    grey_frame = get_grey_frame(frame_width, frame_height, luminance=mean_luminance)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    bin_file.append(grey_frame)
    # Add frames.
    for stimulus_nb in stimulus_nbs:
        stimulus_params = stimuli_params[stimulus_nb]
        stimulus_bin_dirname = stimulus_params['dirname']
        stimulus_bin_filename = '{}.bin'.format(stimulus_params['name'])
        stimulus_bin_path = os.path.join(stimulus_bin_dirname, stimulus_bin_filename)
        stimulus_bin_file = open_bin_file(stimulus_bin_path, mode='r')
        assert stimulus_bin_file.width == frame_width
        assert stimulus_bin_file.height == frame_height
        frame_nbs = stimulus_bin_file.get_frame_nbs()
        for frame_nb in frame_nbs:
            frame_bytes = stimulus_bin_file.read_frame_as_bytes(frame_nb)
            bin_file.append(frame_bytes)
    # Close .bin file.
    bin_file.close()
    # ...
    print("End of .bin file creation.")

    # Create .vec file.
    print("Start creating .vec file...")
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(base_path, vec_filename)
    csv_filename = "{}_trials.csv".format(name)
    csv_path = os.path.join(base_path, csv_filename)
    # ...
    nb_displays_during_adaptation = int(np.ceil(adaptation_duration * display_rate))
    nb_displays_per_flash = int(np.ceil(flash_duration * display_rate))
    nb_displays_per_inter_flash = int(np.ceil(inter_flash_duration * display_rate))
    nb_displays_per_trial = nb_displays_per_flash + nb_displays_per_inter_flash
    nb_displays = nb_displays_during_adaptation + nb_trials * nb_displays_per_trial
    # Open .vec file.
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    # Open .csv file.
    csv_file = open_csv_file(csv_path, columns=['stimulus_nb', 'start_frame_nb', 'end_frame_nb'])
    # Add adaptation.
    # TODO swap and clean the 2 following lines.
    # bin_frame_nb = stimuli_bin_frame_nbs[None]  # i.e. default frame (grey)
    bin_frame_nb = bin_frame_nbs[None]  # i.e. default frame (grey)
    for _ in range(0, nb_displays_during_adaptation):
        vec_file.append(bin_frame_nb)
    # TODO remove the following commented lines.
    # # Add repetitions.
    # for repetition_nb in tqdm.tqdm(range(0, nb_repetitions)):
    #     condition_nbs = repetition_orderings[repetition_nb]
    #     for condition_nb in condition_nbs:
    #         # Add flash.
    #         start_frame_nb = vec_file.get_display_nb() + 1
    #         bin_frame_nb = bin_frame_nbs[condition_nb]
    #         for _ in range(0, nb_displays_per_flash):
    #             vec_file.append(bin_frame_nb)
    #         end_frame_nb = vec_file.get_display_nb()
    #         csv_file.append(condition_nb=condition_nb, start_frame_nb=start_frame_nb, end_frame_nb=end_frame_nb)
    #         # Add inter flash.
    #         bin_frame_nb = bin_frame_nbs[None]  # i.e. default frame (grey)
    #         for _ in range(0, nb_displays_per_inter_flash):
    #             vec_file.append(bin_frame_nb)
    for trial_nb in tqdm.tqdm(ordering):
        stimulus_nb, condition_nb = trials[trial_nb]
        # Add flash.
        start_frame_nb = vec_file.get_display_nb() + 1
        # TODO swap and clean the 2 following lines.
        # bin_frame_nb = stimuli_bin_frame_nbs[stimulus_nb][condition_nb]
        bin_frame_nb = bin_frame_nbs[stimulus_nb][condition_nb]
        for _ in range(0, nb_displays_per_flash):
            vec_file.append(bin_frame_nb)
        end_frame_nb = vec_file.get_display_nb()
        csv_file.append(stimulus_nb=stimulus_nb, start_frame_nb=start_frame_nb, end_frame_nb=end_frame_nb)
        # Add inter flash.
        # TODO swap and clean the 2 following lines.
        # bin_frame_nb = stimuli_bin_frame_nbs[None]
        bin_frame_nb = bin_frame_nbs[None]
        for _ in range(0, nb_displays_per_inter_flash):
            vec_file.append(bin_frame_nb)
    # Close .csv file.
    csv_file.close()
    # Close .vec file.
    vec_file.close()
    # ...
    print("End of .vec file creation.")

    # TODO create conditions .csv file for each stimulus.

    return
