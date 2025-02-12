

def upload_electrode_info(electrode_info_path):

    '''
    Uploads the electrode_info_file from the given path set in the config file.
    
    '''

    pass


def initial_listener_linux( electrode_info ):
    '''
    This funciton trains the initial model given the electrode estimation for
    hyperparameters.

    - Connects to Windows machine via TCP
        - One socket for sending the VEC file
        - One socket for receiving the responses
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

    pass