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
    print("Init linux server is running ...")

    # Set up the thread variables
    threadict = setup_thread_vars() # Dictionary containing the threads and events

    try:
        # Upload the natural image dataset
        nat_img_tuple = main_utils.upload_natural_image_dataset(
            dataset_path=img_dataset_path, astensor=False )

        # Set up the start_model given the electrode information
        start_model = main_utils.model_from_electrode_info(
            electrode_info, *nat_img_tuple )# dict of tensors

        # Plot the chosen RF on the checkerboard STA 
        GP_utils.plot_hyperparams_on_STA(
            start_model, STA=None, ax=None )
        
        generate_send_wait_vec( 
            start_model, threadict, req_socket_vec, n_gray_trgs, n_img_trgs, n_end_gray_trgs )

    except KeyboardInterrupt:
        print('Key Interrups')
        threadict['global_stop_event'].set()

    finally:
        if not threadict['global_stop_event'].is_set():
            threadict['global_stop_event'].set()

        time.sleep(0.2)
        threadict['global_stop_event'].set()     
        print('\nClosing pull socket...')
        pull_socket_packets.setsockopt(zmq.LINGER, 0)
        pull_socket_packets.close()
        print('Closing vec req socket...')
        req_socket_vec.setsockopt(zmq.LINGER, 0)
        req_socket_vec.close()
        print('Closing dmd req socket...')
        req_socket_dmd.setsockopt(zmq.LINGER, 0)
        req_socket_dmd.close()

        
            
        print('Closing context...')
        context.term()
        # print('Joining threads...')
        # computation_thread.join()

        print('Checking for exceptions in queue...')
        if not threadict['exceptions_q'].empty():
            raise threadict['exceptions_q'].get()
    return


if __name__ == "__main__":

    print(" Starting listener_linux.py as main...")

    electrode_info = main_utils.upload_electrode_info( 
        electrode_info_path, print_info=True, testmode = testmode )
    
    initial_listener_linux( electrode_info )