from config.config          import *
from src       import main_utils
# from src.TCP   import listener_linux, initial_listener_linux
# from utils     import GPutils


def main():
    '''
    main.py is the driver script that executes all of the necessary actions in order:

    We assume that we have a picke file containing:
    - The best electrode chosen
    - An estimate for its hyperparameters

    The script will:
    - Upload the electrode informations from the file
    # - Initiating zmq TCP connection to the Windows machine,

    initial_listener_linux.py will:
        - Connect to Windows machine via TCP
        - Generate a VEC file for the first 50 random images to show using the DMD
        - Send the VEC file.
        - Collect the responses to these 50 images and using them to train a GP with a dedicated TCP - GP script
        Return: current_model

    Then:

    listener_linux.py will:
    
    - Iteratively:
        - Estimate best utility from current model
        - Generate VEC file for that image ( DMD utility)
        - Send the created VEC to the DMD and show it
        - Collect response and count triggers
        - Train GP with the updated training set
    Return: final_model - When all 300 images have been shown

    '''
    # Upload electrode information file
    print("MAIN - Uploading electrode information file")
    electrode_info = main_utils.upload_electrode_info( electrode_info_path, print_info=True )

    # Initial listener_linux.py process
    print("MAIN - Launching initial listener ")
    initial_model = initial_listener_linux( electrode_info )        

    # Launch listener_linux.py process
    print("MAIN - Launching listener_linux.py process")
    final_model = listener_linux( initial_model )



if __name__ == "__main__":

    main()


