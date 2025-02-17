''''
Functions needed to generate BIN and VEC files for the DMD
'''
import numpy as np
from tqdm import tqdm
import os
import shutil
import matplotlib.pyplot as plt


from config.config import *
from src import main_utils
from src.BINVEC.bin_manipulation import BinFile
# from src.BINVEC.pystim.images.png import load as load_png_image
# from src.BINVEC.pystim.io.bin import open_file as open_bin_file

def generate_bin_file( nat_img_tuple ):
    '''
    Generates the bin file corresponding to the chosen_idx references 
    in the natural image dataset.

    nat_img_tuple (tuple): Tuple containing the natural image dataset as np arra ( for BinFile to work)
    '''
    # region _______ Create a npy file with the images _______
    assert isinstance(nat_img_tuple[0] , np.ndarray), "The natural image dataset should be a numpy array"

    img_seq_train = nat_img_tuple[0][...,0]
    img_seq_test  = nat_img_tuple[1][...,0]

    img_seq = np.concatenate([img_seq_train, img_seq_test], axis=0)



    np.save( f'{bin_pathname}_array', img_seq ) 
    # endregion

    bin_file = BinFile(path=bin_pathname, 
                       nb_images=img_seq.shape[0] + 1, # We need one more for the gray frame [idx 0] 
                       frame_xsize=n_px_side_init, 
                       frame_ysize=n_px_side_init,
                       reverse=False, 
                       mode='w')

    # Add a gray frame as the first frame [idx 0]
    gray_frame = np.ones([n_px_side_init,n_px_side_init], dtype=np.uint8)*128
    bin_file.append(gray_frame)

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

def generate_vec_file(active_img_ids, rndm_img_ids, 
                              n_gray_trgs, n_img_trgs, n_ending_gray_trgs, 
                              save_file=True, testmode=False):
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

    The idx references the image id IN THE BIN file, which is organised as:
    - idx 0: gray frame
    - idx [1 to tot_imgs_dataset]: all the dataset images.
    We are basically generating a single BIN file and changing only the VEC one ( fastest way to do)

    Parameters:

        active_img_ids (int): The image IDs with respect to the BIN file
    
        rndm_img_ids  (int): The random image IDs with respect to the BIN file
        
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
    for loop_n in range(n_loops):
        img_id = active_img_ids[loop_n]
        for _ in range(n_gray_trgs):         lines.append(f"0 0 0 0 0\n")
        for _ in range(n_img_trgs):          lines.append(f"0 {img_id} 0 0 0\n")
        if n_rndm_imgs != 0:    
            rndm_img_id = rndm_img_ids[loop_n]        
            for _ in range(n_gray_trgs):     lines.append(f"0 0 0 0 0\n")  
            for _ in range(n_img_trgs):      lines.append(f"0 {rndm_img_id} 0 0 0\n")
    for _ in range(n_ending_gray_trgs):      lines.append(f"0 0 0 0 0\n")
    file_content = ''.join(lines)

    if save_file:
        if n_rndm_imgs == 0:
            file_name = f'{session_name}_VEC_start_model_{n_active_imgs}_imgs'
        else:
            file_name = f'{session_name}_VEC_img_id_{active_img_ids[0]}'
        # Session name is in the vec_path in config.py
        save_vec(file_content, dir_path=vec_path, file_name=file_name, testmode=testmode)

        return file_content, vec_path / file_name

    return file_content

def save_vec( vec_content, dir_path, file_name, testmode=False):

    '''
    Save vec file, ask for permission to everwrite if we are not in testmode
    '''
    if not os.path.exists( dir_path ):
        os.makedirs(dir_path)
    else:
        if testmode:
            print(f"Overwriting the directory {dir_path} - save_vec is in in TEST mode")
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        else:
            answer = input(f"Directory {dir_path} already exists. Overwrite? [y/N]: ")
            if answer.strip().lower().startswith('y'):
                print(f"Overwriting the directory {dir_path}")
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
            else:
                raise ValueError(f"Directory {dir_path} already exists. No overwriting access. Exiting...")

    with open(dir_path / file_name, 'w') as file: 
        file.write(vec_content)

def test_vec_img_correspondence( vec_pathname, nat_img_tuple, idx_in_vec, idx_to_compare, show_images=True ): 
    '''
    Test that the images selected by the indexes in the VEC correspond
    to the 

    Args:

        idx_in_vec (1D array): The indexes present in the vec file, in the order they where saved in the Bin

        idx_to_compare (int):  The position of the idx in idx_in_vec to compare
    '''
    bin_file = BinFile(path=bin_pathname, 
                       nb_images=0, 
                       frame_xsize=n_px_side_init, frame_ysize=n_px_side_init, 
                       reverse=False, mode='r')
    
    nb_images = bin_file._nb_images
    frames_in_bin = []
    for i in range(nb_images):
        frames_in_bin.append( bin_file.read_frame(i) )
    bin_file.close()

    train_img_dataset    = nat_img_tuple[0]
    test_img_dataset     = nat_img_tuple[1]
    combined_img_dataset = np.concatenate([train_img_dataset, test_img_dataset], axis=0)

    # Index in the dataset .
    dataset_idx = idx_in_vec[idx_to_compare]
    image_from_dataset = combined_img_dataset[dataset_idx, :, :] # [0,255]
    image_from_bin     = frames_in_bin[dataset_idx + 1 ]         # [0,1] ( it comes from bin )

    # Index of the img block of lines in the vec file
    vec_idx            = torch.where( idx_in_vec == dataset_idx )[0] 

    # The img idx taken as the vec file says
    img_idx_from_vec  = get_bin_idx_from_vec( vec_pathname, vec_idx ) 
    # Image taken with the index on the vec file
    image_from_vec    = combined_img_dataset[img_idx_from_vec] # [0,255]

    if show_images:
        plt.close('all')
        plt.ion()
        plt.subplot(1,3,1)
        plt.imshow( image_from_dataset, cmap='gray', label='Img from Dataset' )
        plt.subplot(1,3,2)
        plt.imshow( image_from_bin, cmap='gray', vmin=0, vmax=1, label='Img from Bin' )
        plt.subplot(1,3,3)
        plt.imshow( image_from_vec, cmap='gray', label='Img from Vec' )
        plt.show(block=True)

    assert img_idx_from_vec == dataset_idx, "The image index taken from the vec file does not correspond to the one in the dataset"

    return

def get_bin_idx_from_vec( vec_pathname, index ):
    '''
    Retrieves the bin index that the vec says it corresponds to index=index of the dataset.

    NB: It only works to check vec files generated for the start_model, meaning files
        that dont encode an active-random image pair.
    '''
    vec_file  = open(vec_pathname, 'r')
    vec_lines = vec_file.readlines()
    vec_file.close()

    # Remove line 0 ( its the header)
    vec_lines = vec_lines[1:]
    # The vec file is organized in blocks of gray+nat triggers. 
    block_height = n_gray_trgs + n_img_trgs
    # Find the block index containing gray trgs + trigs for the selected nat img [index]
    img_block_idx = index
    # Get the block
    vec_block = vec_lines[img_block_idx*block_height : (img_block_idx+1)*block_height]
    # Get the first nat img line
    first_img_line = vec_block[n_gray_trgs]
    # Return the index saved in the string after a '0' and a ' ' characters
    return int(first_img_line.split('0 ')[1])


if __name__ == "__main__":

    print(" Generating the bin file ( lanched as MAIN )...")

    # Upload the natural image dataset
    nat_img_tuple = main_utils.upload_natural_image_dataset( dataset_path=img_dataset_path, astensor=False )

    # Upload the electrode information
    electrode_info = main_utils.upload_electrode_info( 
        electrode_info_path, print_info=True, testmode = testmode )

    # Set up the start_model given the electrode information
    start_model = main_utils.model_from_electrode_info( electrode_info, *nat_img_tuple )# dict of tensors

    # - Generates a VEC file for the first 50 random images to show using the DMD
    vec_file = generate_vec_file_updated(
                active_img_ids = start_model['fit_parameters']['in_use_idx'],
                rndm_img_ids   = torch.empty(0),
                n_gray_trgs    = n_gray_trgs,
                n_img_trgs     = n_img_trgs,
                n_ending_gray_trgs = n_ending_gray_trgs,
                save_file=True,
                testmode = testmode )

    generate_bin_file( nat_img_tuple, chosen_idxs = start_model['fit_parameters']['in_use_idx'], )

    vec_file = generate_vec_file_updated(
            active_img_ids = start_model['fit_parameters']['in_use_idx'],
            rndm_img_ids   = torch.empty(0),
            n_gray_trgs    = n_gray_trgs,
            n_img_trgs     = n_img_trgs,
            n_ending_gray_trgs = n_ending_gray_trgs,
            save_file=True,
            testmode = testmode )

    bin_file_name = f'{session_name}_bin_file_init'
    bin_pathname  = bin_path / bin_file_name
    vec_file_name = f'{session_name}_VEC_start_model_{ntrain_init}_imgs'
    vec_pathname  = vec_path / vec_file_name

    test_bin_vec_correspondence( bin_pathname, vec_pathname )











