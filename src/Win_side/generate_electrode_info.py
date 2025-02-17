
import os
import json
from config.config import *


print(f"Repo dir: {REPO_DIR} from generate_electrode_info.py")


def generate_electrode_info(testmode):

    """
    Prompts the user for electrode number and hyperparameters estimation
    (!) in the natural image dataset frame of reference (!).
    
    Saves the info to a JSON file.

    Returns:
        electrode_info (dict): Dict
            'electrode_number' (int): The chosen electrode number.

            'threshold_multiplier' : Sets the amplitude of the peaks of that electrode    

            'RF_x_center' - eps_0x: Estimated rf center x-coordinate.
            'RF_y_center' - eps_0y: Estimated rf center y-coordinate.            
            'RF_size'     - beta  : Estimated rf size.

            'localker_amplitude'  : hyperparameter Amp, amplitude of the localker
            'localker_smoothness' : hyperparameter rho, smoothness in the localker
            'acosker_sigma_0'     : hyperparameter sigma_0 of the acosker

    """

    electrode_info = {}
    print(f'Generating electrode info:')
    if testmode:
        print('generate_electrode_info in TEST mode, values generated from config.py - Not saving to file')
        electrode_info['electrode_number']     = ch_id
        electrode_info['threshold_multiplier'] = threshold_multiplier_init

        electrode_info['RF_x_center'] = eps_0x_init
        electrode_info['RF_y_center'] = eps_0y_init
        electrode_info['RF_size']     = beta_init

    else:    
        electrode_info['electrode_number'] = int(input('Enter the chosen electrode number: ', ))
        electrode_info['threshold_multiplier'] = float(input('Enter the threshold multiplier: '))

        electrode_info['RF_x_center'] = float(input(f'Hyperparameter eps_0x\n   Enter the estimated rf center x-coordinate - in the {nat_img_px_nb}px by {nat_img_px_nb}px images set reference: '))
        electrode_info['RF_y_center'] = float(input(f'Hyperparameter eps_0y\n   Enter the estimated rf center y-coordinate - in the {nat_img_px_nb}px by {nat_img_px_nb}px images set reference: '))
        electrode_info['RF_size'] = float(input(f'Hyperparameter beta\n   Enter the estimated rf size - in the {nat_img_px_nb}px by {nat_img_px_nb}px images set reference: '))


    electrode_info['localker_smoothness'] = rho_init
    electrode_info['localker_amplitude']  = Amp_init
    electrode_info['acosker_sigma_0']     = sigma_0_init


    print(f'Generating electrode info:')
    if testmode: print('TEST is on')
    for key, value in electrode_info.items():
        if type(value) == float:
            print(f'   {key}: {value:.2f}')
        else:
            print(f'   {key}: {value:.0f}')


    if not testmode:
        # check if file exists and save the electrode info
        # if it exists add a numbe rto the file name
        electrode_info_filename = 'electrode_info.json'
        electrode_info_file_path = REPO_DIR / 'data' / 'electrode_info' / electrode_info_filename


        if os.path.exists(electrode_info_file_path):
            print(f'File {electrode_info_filename} already exists. Saving as a new file.')
            count_files = len([name for name in os.listdir(os.path.join(REPO_DIR, 'data')) if name.startswith('electrode_info') and name.endswith('.json')])
            electrode_info_filename = f'electrode_info_{count_files+1}.json'
            electrode_info_file_path = REPO_DIR / 'data' / electrode_info_filename

        with open(electrode_info_file_path, 'w') as f:
            json.dump(electrode_info, f)
        
    return electrode_info


if __name__ == '__main__':

    electrode_info = generate_electrode_info(testmode)
