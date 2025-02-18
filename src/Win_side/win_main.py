from config.config import *
from src.Win_side.active_run_MEA_DMD import active_run_MEA_DMD
from src.Win_side.init_run_MEA_DMD import init_run_MEA_DMD
from src.Win_side.gen_and_save_electrode_info import gen_and_save_electrode_info
from src.Win_side.win_utils import ask_to_continue

def main():
    '''
    This main.py will execute all the actions needed from the Windows side needed
    after the definition of STA center and width.
    ( That part is taken care of by the pipeline scripts. )
    
    The main actions are:

        0.1
        - Visualize the chosen electrode spike shapes and selected threshold multiplier
        - Ask for confirmation for the selected threshold multiplier and continue
    
        1.
        - Generating the pikled electrode_info file and save it. 
        - Wait for it to be transfered to the Linux computer, prompt the continuation of program.
    
        2.
        - Launch init_run_MEA_DMD.py. 
            This starts the recording of the MEA
            Waits for the VEC file from the Linux side
            Waits for command to executes DMD process with VEC file order of images from Linux side
            Waits for command to stop the DMD process from Linux side

        3.
        - Launch active_run_MEA_DMD.py
            Starts the MEA recording and sending of packets
            Waits for the first VEC file and confirms
            Starts the DMD process 
            Waits for DMD stop event and confirms
            Loop until NO VEC is received and timeout is reached

    '''
    # Visualize the chosen electrode spike shapes and selected threshold multiplier
    # ...
    
    # Generate the electrode info and save it to electrode_info_path
    electrode_info = gen_and_save_electrode_info(testmode=testmode)

    # Wait for the electrode info to be transfered to the Linux computer
    if not ask_to_continue(testmode): return

    # Launche the Windows side MEA recording and DMD process for the initial model fit
    init_run_MEA_DMD()

    # Launch the Windows side listener
    # active_run_MEA_DMD()

    return

if __name__ == "__main__":

    main()
