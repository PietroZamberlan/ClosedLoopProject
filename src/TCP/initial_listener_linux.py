# region _______ Imports __________
import numpy as np

# Import the configuration file
from config.config import *
from src.TCP.tcp_utils import *
from src import main_utils
from src.BINVEC.binvec_utils import *

# endregion 
def initial_listener_linux( electrode_info ):
    '''
    This funciton trains the initial model given the electrode estimation for
    hyperparameters.

    - Connects to Windows machine via TCP
        - One socket for sending the VEC file
        - One socket for receiving the responses
        - One socket to send the DMD off command once the whole squence has been received

    - Generates a VEC file for the first 50 random images to show using the DMD
    - Saves the used_img_idxs in the model dict
    - Sends the VEC file to the initial_run_MEA_and_DMD.py script running on Win
    - Receives all responses
    - Using the responses, trains a GP model

    Args:
        electrode_info (dict): Dictionary containing the electrode information

    Returns:
        initial_model (dict): Dictionary containing the initial model fit infos
    
    '''

    # Set up the context and the sockets
    context, pull_socket_packets, req_socket_vec, req_socket_dmd = setup_lin_side_sockets()
    print("Init linux server is running and waiting for data stream...")

    # Upload the natural image dataset
    nat_img_tuple = main_utils.upload_natural_image_dataset( dataset_path=img_dataset_path, astensor=False )

    # Set up the start_model given the electrode information
    start_model = main_utils.model_from_electrode_info( electrode_info, *nat_img_tuple )# dict of tensors

    # Plot the chosen RF on the checkerboard STA 
    GP_utils.plot_hyperparams_on_STA( start_model, STA=None, ax=None )

    # Generate the vec file for the starting 50 images
    vec_file, vec_pathname = generate_vec_file(
                active_img_ids = start_model['fit_parameters']['in_use_idx'],
                rndm_img_ids   = torch.empty(0),
                n_gray_trgs    = n_gray_trgs,
                n_img_trgs     = n_img_trgs,
                n_ending_gray_trgs = n_ending_gray_trgs,
                save_file=True,
                testmode=testmode ) 
    








    return

if __name__ == "__main__":

    print(" Starting listener_linux.py as main...")

    electrode_info = main_utils.upload_electrode_info( 
        electrode_info_path, print_info=True, testmode = testmode )
    
    initial_listener_linux( electrode_info )