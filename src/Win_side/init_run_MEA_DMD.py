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

    2.
    Waits for the VEC file from the Linux side
        Confirm VEC reception

    Waits for command to executes DMD process
        Launch DMD process ( it project images using the DMD in the order specified in the VEC file )
    
    Waits for command to stop the DMD process from Linux side
        Stop the DMD process
    
    '''

    def setup_win_side_sockets():
        '''
        Sets up the VEC reception and DMD process control sockets.

        The socket for packet sending is set up in the ORT subprocess.
        '''


        # Listening socket
        context     = zmq.Context()

        rep_socket_vec = context.socket(zmq.REP)
        rep_socket_vec.bind(f"tcp://0.0.0.0:{REQ_SOCKET_VEC_PORT}")

        rep_socket_dmd = context.socket(zmq.REP)
        rep_socket_dmd.bind(f"tcp://0.0.0.0:{REQ_SOCKET_DMD_PORT}")

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('   Windows client is running...')

        return context, rep_socket_vec, rep_socket_dmd

    # Setup the sockets to receive VEC and DMD off messages
    context, rep_socket_vec, rep_socket_dmd = setup_win_side_sockets()

    def launch_ort_process_updated( ORT_READER_PATH, ort_reader_params, testmode ):
        '''Launches the ort reader process to acquire data from the MEA device.
            
            Sleeps for 2.5 the main thread to wait for ort_reader to get to the point 
            at which it writes the signal file or fails cause no device is connected.
            
        Args:
            ORT_READER_PATH: The path to the Python script that launches the acquisition
            ort_reader_params: The parameters to pass to the Python script'''
        # Run the Python script with the parameters
        if testmode:
            print("Launching MEA acquisition in TEST mode...")
        else:
            print("Launching MEA acquisition...")
        
        with open(ort_reader_start_output_pathname, "w") as log_file_ort:
            # We set the -u flag to avoid buffering the output ( it would stop the OnChannelData to print the data in real time )
            ort_process = subprocess.Popen(["python", '-u', ORT_READER_PATH] + ort_reader_params + (["-T"] if testmode else []), 
                                        stdout=log_file_ort, stderr=log_file_ort, text=True)
            print("Acquisition is running...")

            def flush_log():
                while ort_process.poll() is None:
                    log_file_ort.flush()
                    time.sleep(1)

            flush_thread = threading.Thread(target=flush_log, daemon=True)
            flush_thread.start()

            time.sleep(.5) # the ort process takes some time to get to the point in which it writes the signal file, so we wait a bit instead of printing a lot of "waiting for signal file"

        return ort_process

    # ort reader parameters
    ort_reader_params = ["-ip", LINUX_IP, "--port", PULL_SOCKET_PACKETS_PORT, 
                        "--buffer_size", f'{buffer_size}', '--threshold_multiplier', f'{threshold_multiplier_init}', 
                        "--filename", raw_data_file_path]

    ort_process = launch_ort_process_updated( ORT_READER_PATH, ort_reader_params, testmode=testmode)

    return


if __name__ == "__main__":

    init_run_MEA_DMD()















