import subprocess
import time
import os
import zmq
import threading
import queue
import sys

# Import the configuration file
from config.config import *
from win_utils import *


def init_run_MEA_DMD():
    '''
    This function executes the MEA-connected side, first part of the experiment.

    1.
    Starts  MEA recording, and sending packets.
        Setup sockets for communication
        Start MEA recording ( the ORT subprocess  )
        Wait for signal to allow start of DMD 

    2.
    Waits for the VEC file from the Linux side
        Setup threading variables
        Start listening for DMD off signal
        Receive
        Confirm VEC reception

    3.
    Waits for command to executes DMD process
        Launch DMD process ( it project images using the DMD in the order specified in the VEC file )
    
    4.
    Waits for command to stop the DMD process from Linux side
        Stop the DMD process
    
    '''
    # Threading variables
    threadict, DMD_off_listening_thread, \
        DMD_thread, vec_receiver_confirmer_thread \
            = setup_win_thread_vars()

    # Socket variables
    context, rep_socket_vec, rep_socket_dmd = None, None, None

    try:
        # 1.
        # Setup the sockets to receive VEC and DMD off messages
        context, rep_socket_vec, rep_socket_dmd = setup_win_side_sockets()

        # Setup and run MEA recording (ORT process)
        ort_reader_params = ["-ip", LINUX_IP, "--port", PULL_SOCKET_PACKETS_PORT, 
                            "--buffer_size", f'{buffer_size}', '--threshold_multiplier', f'{threshold_multiplier_init}', 
                            "--filename", raw_data_file_path]
        # Handle to log file has to be defined in long lived scope to avoid function returning and closing file
        log_file_ort = open(ort_reader_start_output_pathname, "w")

        # Launch MEA and wait for file to allow DMD start
        ort_process = launch_ort_process( ORT_READER_PATH, ort_reader_params, 
                                        log_file_ort=log_file_ort, testmode=testmode)

        wait_for_signal_file_to_start_DMD(ort_process)

        # 2.
        # Launch DMD off cmd receiver listening thread
        DMD_off_listening_thread = launch_dmd_off_receiver(
            rep_socket_dmd, threadict, timeout_dmd_off_rcv_phase1 )

        # Wait for VEC file 
        vec_receiver_confirmer_thread = wait_for_VEC_file(
            rep_socket_vec, threadict, timeout_vec_phase1, vec_pathname_dmd_source_start )
        if threadict['global_stop_event'].is_set(): return

        # 3.
        # Start DMD projector    
        log_file_DMD = open(dmd_start_output_pathname, "w")

        exe_params = [pietro_dir_DMD, bin_number, vec_number, frame_rate, advanced_f, n_frames_LUT]
        input_data_DMD = "\n".join(exe_params)+"\n"

        DMD_thread = launch_DMD_process_thread( 
            input_data_DMD, threadict, log_file_DMD,
            testmode=testmode )

        # 4.
        # Wait for DMD stop command
        # while threadict['dmd'] \
            # not threadict['global_stop_event']


    except KeyboardInterrupt:
        print("Key Interrupt")
        threadict['global_stop_event'].set()

    finally:
        if not threadict['global_stop_event'].is_set():
            threadict['global_stop_event'].set()

        # Then join them
        time.sleep(0.2)
        join_treads([DMD_off_listening_thread, DMD_thread, vec_receiver_confirmer_thread])
        
        # Terminate subprocesses
        terminate_DMD_queue(threadict['process_queue_DMD'])
        terminate_ort_process(ort_process, log_file_ort)
        
        # Close the VEC and DMD sockets
        close_sockets([rep_socket_vec, rep_socket_dmd])

        # Clean up the signal file if it exists
        if os.path.exists(signal_file):
            os.remove(signal_file)
            print("Signal file cleanup complete.")

        print('Checking for exceptions in queue...')
        if not threadict['exceptions_q'].empty():
            raise threadict['exceptions_q'].get()
    return


if __name__ == "__main__":

    init_run_MEA_DMD()















