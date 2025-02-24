import collections
import numpy as np
import os
import pandas as pd
import tempfile
import tqdm

from pystim.io.bin import load_nb_frames as load_nb_bin_frames
from pystim.io.bin import open_file as open_bin_file
from pystim.io.csv import load_file as load_csv_file
from pystim.io.csv import open_file as open_csv_file
from pystim.io.vec import open_file as open_vec_file
from pystim.io.vec import load_file as load_vec_file
from pystim.utils import float_frame_to_uint8_frame
from pystim.utils import get_grey_frame
from pystim.utils import handle_arguments_and_configurations


pystim_path = os.path.join(tempfile.gettempdir(), 'pystim')

name = 'fi_merge'

default_configuration = {
    'path': os.path.join(pystim_path, name),
    'stimuli': [
        'fipwfc',
        'fipwrc',
        'fi',
    ],
    'mean_luminance': 0.25,
    'display_rate': 40.0,  # Hz
    # 'adaptation_duration': 5.0,  # s
    'adaptation_duration': 60.0,  # s
    # 'flash_duration': 10.0,  # s
    'flash_duration': 0.3,  # s
    # 'inter_flash_duration': 1.0,  # s
    'inter_flash_duration': 0.3,  # s
    'frame': {
        'width': 864,  # px
        'height': 864,  # px
        'resolution': 3.5e-6,  # m / pixel  # fixed by the setup
    },
    'to_interleave': True,
    'seed': 42,
}


def generate(args):

    config = handle_arguments_and_configurations(name, args)

    base_path = config['path']
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    print("Generation in {}.".format(base_path))

    # Get configuration parameters.
    stimulus_names = config['stimuli']
    mean_luminance = config['mean_luminance']
    display_rate = config['display_rate']
    adaptation_duration = config['adaptation_duration']
    flash_duration = config['flash_duration']
    inter_flash_duration = config['inter_flash_duration']
    frame_width = config['frame']['width']
    frame_height = config['frame']['height']
    seed = config['seed']
    to_interleave = config['to_interleave']

    # TODO pour chaque stimulus, récupérer les condition_nbs.
    # TODO pour chaque stimulus, récupérer le mapping condition_nb -> bin_frame_nb.
    # TODO pour chaque stimulus, récupérer la séquence de condition_nb.
    # TODO créer une séquence 'interleaved' de (stimulus_nb, condition_nb).
    # TODO convertir en séquence (stimulus_nb, bin_frame_nb).
    # TODO convertir en séquence bin_frame_nb.

    nb_stimuli = len(stimulus_names)
    stimulus_nbs = np.arange(0, nb_stimuli)
    stimuli_params_dict = {}  # stimulus_nb -> stimulus_params
    for stimulus_nb in stimulus_nbs:
        stimulus_name = stimulus_names[stimulus_nb]
        stimulus_params = {
            'name': stimulus_name,
            'bin_path': os.path.join(pystim_path, stimulus_name, '{}.bin'.format(stimulus_name)),
            'vec_path': os.path.join(pystim_path, stimulus_name, '{}.vec'.format(stimulus_name)),
            'trials_path': os.path.join(pystim_path, stimulus_name, '{}_trials.csv'.format(stimulus_name)),
        }
        stimuli_params_dict[stimulus_nb] = stimulus_params
    # Get number of bin frames for each stimulus.
    for stimulus_nb in stimulus_nbs:
        stimulus_bin_path = stimuli_params_dict[stimulus_nb]['bin_path']
        stimulus_nb_bin_frames = load_nb_bin_frames(stimulus_bin_path)
        stimuli_params_dict[stimulus_nb]['nb_bin_frames'] = stimulus_nb_bin_frames

    # Create .bin file.
    bin_filename = '{}.bin'.format(name)
    bin_path = os.path.join(base_path, bin_filename)
    nb_bin_frames = 1 + int(np.sum([
        stimuli_params_dict[stimulus_nb]['nb_bin_frames']
        for stimulus_nb in stimulus_nbs
    ]))
    # Open .bin file.
    bin_file = open_bin_file(bin_path, nb_bin_frames, frame_width=frame_width,
                             frame_height=frame_height, reverse=False, mode='w')
    # ...
    bin_frame_nbs_dict = {}  # stimulus_nb -> stimulus_bin_frame_nb -> bin_frame_nb
    # Add grey frame.
    grey_frame = get_grey_frame(frame_width, frame_height, luminance=mean_luminance)
    grey_frame = float_frame_to_uint8_frame(grey_frame)
    bin_file.append(grey_frame)
    bin_frame_nbs_dict[None] = bin_file.get_frame_nb()
    # Add frames.
    for stimulus_nb in stimulus_nbs:
        stimulus_params = stimuli_params_dict[stimulus_nb]
        # Open stimulus .bin file.
        stimulus_bin_path = stimulus_params['bin_path']
        stimulus_bin_file = open_bin_file(stimulus_bin_path, mode='r')
        # Copy frames from stimulus .bin file to .bin file.
        stimulus_bin_frame_nbs = stimulus_bin_file.get_frame_nbs()
        bin_frame_nbs_dict[stimulus_nb] = {}
        for stimulus_bin_frame_nb in stimulus_bin_frame_nbs:
            frame_bytes = stimulus_bin_file.read_frame_as_bytes(stimulus_bin_frame_nb)
            bin_file.append(frame_bytes)
            bin_frame_nbs_dict[stimulus_nb][stimulus_bin_frame_nb] = bin_file.get_frame_nb()
        # Close stimulus .bin file.
        stimulus_bin_file.close()
    # ...
    assert bin_file.get_frame_nb() == nb_bin_frames - 1, "{} != {} - 1".format(bin_file.get_frame_nb(), nb_bin_frames)
    # Close .bin file.
    bin_file.close()
    # ...
    print("End of .bin file creation.")

    # Get trials for each stimulus.
    for stimulus_nb in stimulus_nbs:
        stimulus_trials_path = stimuli_params_dict[stimulus_nb]['trials_path']
        stimulus_trials = load_csv_file(stimulus_trials_path, expected_columns=['condition_nb', 'start_display_nb', 'end_display_nb'])
        stimuli_params_dict[stimulus_nb]['trials'] = stimulus_trials
    # Compute the number of trials for each stimulus.
    for stimulus_nb in stimulus_nbs:
        stimulus_trials = stimuli_params_dict[stimulus_nb]['trials']
        stimulus_nb_trials = len(stimulus_trials)
        stimuli_params_dict[stimulus_nb]['nb_trials'] = stimulus_nb_trials
    # Compute the number of trials after the merger.
    nb_trials = int(np.sum([
        stimuli_params_dict[stimulus_nb]['nb_trials']
        for stimulus_nb in stimulus_nbs
    ]))

    # Generate the interleaved sequence of stimulus numbers.
    merged_stimulus_nbs_sequence = []
    for stimulus_nb in stimulus_nbs:
        stimulus_nb_trials = stimuli_params_dict[stimulus_nb]['nb_trials']
        merged_stimulus_nbs_sequence.extend(stimulus_nb_trials * [stimulus_nb])
    merged_stimulus_nbs_sequence = np.array(merged_stimulus_nbs_sequence)
    assert merged_stimulus_nbs_sequence.size == nb_trials
    if to_interleave:
        np.random.seed(seed)
        np.random.shuffle(merged_stimulus_nbs_sequence)
    # Compute the corresponding interleaved sequence of stimulus condition numbers.
    merged_stimulus_condition_nbs_sequence = np.empty(nb_trials, dtype=np.int)
    for stimulus_nb in stimulus_nbs:
        indices = np.where(merged_stimulus_nbs_sequence == stimulus_nb)[0]
        merged_stimulus_condition_nbs_sequence[indices] = stimuli_params_dict[stimulus_nb]['trials']['condition_nb']
    # Compute the corresponding interleaved sequence of stimulus bin frame numbers.
    merged_stimulus_bin_frame_nbs_sequence = np.empty(nb_trials, dtype=np.int)
    for stimulus_nb in stimulus_nbs:
        stimulus_trials = stimuli_params_dict[stimulus_nb]['trials']
        stimulus_vec_path = stimuli_params_dict[stimulus_nb]['vec_path']
        stimulus_bin_frame_nbs_sequence = load_vec_file(stimulus_vec_path)
        tmp = np.empty(len(stimulus_trials), dtype=np.int)
        for index, stimulus_trial in stimulus_trials.iterrows():
            start_display_nb = stimulus_trials['start_display_nb'][index]
            end_display_nb = stimulus_trials['end_display_nb'][index]
            try:
                bin_frame_nbs_sequence = stimulus_bin_frame_nbs_sequence[start_display_nb:end_display_nb+1]
            except TypeError as e:
                print(start_display_nb)
                print(end_display_nb)
                raise e
            assert np.all(bin_frame_nbs_sequence == bin_frame_nbs_sequence[0])
            bin_frame_nb = bin_frame_nbs_sequence[0]
            tmp[index] = bin_frame_nb
        indices = np.where(merged_stimulus_nbs_sequence == stimulus_nb)[0]
        merged_stimulus_bin_frame_nbs_sequence[indices] = tmp
    # Summarize everything in a data frame.
    merged_trials_dataframe = pd.DataFrame(data={
        'stimulus_nb': merged_stimulus_nbs_sequence,
        'condition_nb': merged_stimulus_condition_nbs_sequence,
        'bin_frame_nb': merged_stimulus_bin_frame_nbs_sequence,
    })

    # Create .vec file.
    print("Start creating .vec file...")
    vec_filename = "{}.vec".format(name)
    vec_path = os.path.join(base_path, vec_filename)
    csv_filename = "{}_trials.csv".format(name)
    csv_path = os.path.join(base_path, csv_filename)
    csv_filenames = {
        stimulus_nb: os.path.join(base_path, '{}_{}_trials.csv'.format(name, stimulus_names[stimulus_nb]))
        for stimulus_nb in stimulus_nbs
    }
    csv_paths = {
        stimulus_nb: os.path.join(base_path, csv_filenames[stimulus_nb])
        for stimulus_nb in stimulus_nbs
    }
    # ...
    nb_displays_during_adaptation = int(np.ceil(adaptation_duration * display_rate))
    nb_displays_per_flash = int(np.ceil(flash_duration * display_rate))
    nb_displays_per_inter_flash = int(np.ceil(inter_flash_duration * display_rate))
    nb_displays_per_trial = nb_displays_per_flash + nb_displays_per_inter_flash
    nb_displays = nb_displays_during_adaptation + nb_trials * nb_displays_per_trial
    # # Open .vec file.
    vec_file = open_vec_file(vec_path, nb_displays=nb_displays)
    # # Open .csv files.
    columns = ['condition_nb', 'start_display_nb', 'end_display_nb']
    csv_file = open_csv_file(csv_path, columns=columns)
    csv_files = {
        stimulus_nb: open_csv_file(csv_paths[stimulus_nb], columns=columns)
        for stimulus_nb in stimulus_nbs
    }
    # # Add adaptation.
    bin_frame_nb = bin_frame_nbs_dict[None]  # i.e. default frame (grey)
    for _ in range(0, nb_displays_during_adaptation):
        vec_file.append(bin_frame_nb)
    for _, merged_trial in tqdm.tqdm(merged_trials_dataframe.iterrows()):
        stimulus_nb = merged_trial['stimulus_nb']
        stimulus_condition_nb = merged_trial['condition_nb']
        stimulus_bin_frame_nb = merged_trial['bin_frame_nb']
        # Add flash.
        start_display_nb = vec_file.get_display_nb() + 1
        bin_frame_nb = bin_frame_nbs_dict[stimulus_nb][stimulus_bin_frame_nb]
        for _ in range(0, nb_displays_per_flash):
            vec_file.append(bin_frame_nb)
        end_display_nb = vec_file.get_display_nb()
        csv_file.append(condition_nb=stimulus_nb, start_display_nb=start_display_nb, end_display_nb=end_display_nb)
        csv_files[stimulus_nb].append(condition_nb=stimulus_condition_nb, start_display_nb=start_display_nb, end_display_nb=end_display_nb)
        # Add inter flash.
        bin_frame_nb = bin_frame_nbs_dict[None]
        for _ in range(0, nb_displays_per_inter_flash):
            vec_file.append(bin_frame_nb)
    # Close .csv files.
    csv_file.close()
    for csv_file in csv_files.values():
        csv_file.close()
    # Close .vec file.
    vec_file.close()
    # ...
    print("End of .vec file creation.")

    return
