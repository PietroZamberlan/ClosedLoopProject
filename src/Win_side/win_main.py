from config.config import *
from src.Win_side.active_run_MEA_DMD import active_run_MEA_DMD

def main():
    '''
    This main.py will execute all the actions needed from the Windows side needed
    after the definition of STA center and width.
    ( That part is taken care of by the pipeline scripts. )
    
    The main actions are:

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

    '''

    # Launch the Windows side listener
    active_run_MEA_DMD()

    return

if __name__ == "__main__":

    main()
