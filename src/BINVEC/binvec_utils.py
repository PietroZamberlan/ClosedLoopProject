''''
Functions needed to generate BIN and VEC files for the DMD
'''
import numpy as np
from tqdm import tqdm
import os
import shutil


from config.config import *
from src import main_utils
from src.BINVEC.bin_manipulation import BinFile
# from src.BINVEC.pystim.images.png import load as load_png_image
# from src.BINVEC.pystim.io.bin import open_file as open_bin_file

def generate_bin_file( nat_img_tuple, chosen_idxs ):
    '''
    Generates the bin file corresponding to the chosen_idx references 
    in the natural image dataset.

    nat_img_tuple (tuple): Tuple containing the natural image dataset as torch tensors
     -> Gets transformed into a numpy array for BinFile to work
    
    
    '''
    # region _______ Create a npy file with the images _______
    assert isinstance(nat_img_tuple[0] , np.ndarray), "The natural image dataset should be a numpy array"
    img_dataset = nat_img_tuple[0]

    img_seq   = img_dataset[chosen_idxs][:,:,:,0]

    file_name = f'{session_name}_bin_file_init'
    bin_pathname = bin_path / file_name
    torch.save( img_seq, bin_pathname ) 
    # endregion

    bin_file = BinFile(path=bin_pathname, 
                       nb_images=img_seq.shape[0], 
                       frame_xsize=n_px_side_init, 
                       frame_ysize=n_px_side_init,
                       reverse=False, 
                       mode='w')

    # Add the chosen nat imgs to the bin file
    for ref in tqdm(range(0, img_seq.shape[0])):
        temp_im = img_seq[ref, :, :]
        bin_file.append(temp_im)

    bin_file.close()

    # TO check which image each fream is in debug console:
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.imshoq(bin_file.read_frame(0))
    # plt.show(block=False)

def generate_vec_file_updated(active_img_ids, rndm_img_ids, 
                              n_gray_trgs, n_img_trgs, n_ending_gray_trgs, 
                              save_file=True):
    """
    Generate the VEC file for the chosen image IDs and the random image IDs,
    with the following structure:
    0 {total_frames} 0 0 0
    for _ in range(n_active_imgs):
        0 0 0 0 0                   [n_gray_trgs lines]
        0 {ith_chosen_img_id} 0 0 0 [n_img_trgs lines]
        0 0 0 0 0                   [max_grey_trgs lines]
        0 {rndm_img_id} 0 0 0       [n_img_trgs lines]
    0 0 0 0 0                   [n_gray_trgs lines]

    If rndm_img_ids is None, the function will generate the VEC file for the active_img_ids only.

    Parameters:
    active_img_ids (int): The image IDs.
    rndm_img_ids  (int): The random image IDs.
    n_gray_trgs (int): The number of lines representing the STARTING gray image.
    n_ending_gray_trgs (int): The number of lines representing the ENDING gray image.
    n_img_trgs (int): The number of lines representing triggers of the natural image be it active or random

    Returns:
        file_content (str): The content of the VEC file.
    """

    n_active_imgs = active_img_ids.shape[0]
    n_rndm_imgs   = rndm_img_ids.shape[0]    # either ==0 or ==n_active_imgs
    if not ((n_active_imgs == n_rndm_imgs) or (n_rndm_imgs == 0)):
        raise ValueError(f"The number of active ({n_active_imgs}) and random ({n_rndm_imgs}) images must either same or n_active and zero")

    n_loops       = n_active_imgs
    n_loop_frames =  n_gray_trgs+n_img_trgs
    n_loop_frames += n_gray_trgs+n_img_trgs if n_rndm_imgs != 0 else 0
    n_frames_tot  = n_loops * n_loop_frames + n_ending_gray_trgs

    lines = []
    # Write the lines
    lines.append(f"0 {n_frames_tot} 0 0 0\n")
    for _ in range(n_loops):
        for _ in range(n_gray_trgs):       lines.append(f"0 0 0 0 0\n")
        for img_id in active_img_ids:        lines.append(f"0 {img_id} 0 0 0\n")
        if n_rndm_imgs != 0:            
            for _ in range(n_gray_trgs):   lines.append(f"0 0 0 0 0\n")  
            for rndm_img_id in rndm_img_ids: lines.append(f"0 {rndm_img_id} 0 0 0\n")
    for _ in range(n_ending_gray_trgs):        lines.append(f"0 0 0 0 0\n")
    file_content = ''.join(lines)

    if save_file:
        if n_rndm_imgs == 0:
            file_name = f'VEC_start_model_{n_active_imgs}_imgs'
        else:
            file_name = f'VEC_img_id_{active_img_ids[0]}'
        # Session name is in the vec_path in config.py
        save_vec(file_content, dir_path=vec_path, file_name=file_name)
              
    return file_content

def save_vec( vec_content, dir_path, file_name):

    if not os.path.exists( dir_path ):
        os.makedirs(dir_path)
    else:
        answer = input(f"Directory {dir_path} already exists. Overwrite? [y/N]: ")
        if answer.strip().lower().startswith('y'):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        else:
            raise ValueError(f"Directory {dir_path} already exists. Exiting...")

    with open(dir_path / file_name, 'w') as file: 
        file.write(vec_content)

if __name__ == "__main__":

    print(" Generating the bin file ( lanched as MAIN )...")

    # Upload the natural image dataset
    nat_img_tuple = main_utils.upload_natural_image_dataset( dataset_path=img_dataset_path, astensor=False )

    # Upload the electrode information
    electrode_info = main_utils.upload_electrode_info( 
        electrode_info_path, print_info=True, testmode = testmode )

    # Set up the start_model given the electrode information
    start_model = main_utils.model_from_electrode_info( electrode_info, *nat_img_tuple )# dict of tensors

    generate_bin_file( nat_img_tuple, chosen_idxs = start_model['fit_parameters']['in_use_idx'], )