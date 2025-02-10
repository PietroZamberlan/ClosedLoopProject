# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:05:09 2024

Script to simulate the simplest form of experiment:
    
    - Connection is established with the Linux client
    - Loop:
        - One Gray-Image-Gray triplet is shown by te DMD
        - DMD is shut off ( triggers stop )
    - Connection is closed gracefully from the server side    
"""

import subprocess
import time
import os
import zmq
import threading
import queue
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir    = os.path.join(current_dir, '..\\..\\')
sys.path.insert(0, os.path.abspath(repo_dir))
print(f"Repo dir: {repo_dir}")

# Import the configuration file
from config.config import *
from win_utils import *


# Threading variables
vec_received_confirmed_event = threading.Event()
global_stop_event            = threading.Event()
dmd_off_event                = threading.Event()
# Wait for authorization to overwrite the vec file. If this event is not set
# it means
allow_vec_changes_event    = threading.Event()
process_queue_DMD          = queue.Queue()

# Threads variables. 
# We need them to be able to join them in the finally if they were never defined
vec_receiver_confirmer_thread    = None
DMD_off_listening_thread         = None
DMD_thread                       = None

exe_params = [pietro_dir_DMD, bin_number, vec_number, frame_rate, advanced_f, n_frames_LUT]
input_data_DMD = "\n".join(exe_params)+"\n"
signal_file = "signal_file.txt"

# ort reader parameters
ort_reader_params = ["-ip", LINUX_IP, "--port", PULL_SOCKET_PACKETS_PORT, 
                     "--buffer", buffer_size, "--filename", raw_data_file_path]

# Listening socket
context     = zmq.Context()

rep_socket_vec = context.socket(zmq.REP)
rep_socket_vec.bind(f"tcp://0.0.0.0:{REQ_SOCKET_VEC_PORT}")

rep_socket_dmd = context.socket(zmq.REP)
rep_socket_dmd.bind(f"tcp://0.0.0.0:{REQ_SOCKET_DMD_PORT}")

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('   Windows client is running...')

# Opening file to redirect output of ort process

with open(f"{Win_side_path}output_ort_reader.log", "w") as log_file_ort, \
    open(f"{Win_side_path}output_DMD.log", "w") as log_file_DMD:
        try:

            ort_process = launch_ort_process(ORT_READER_PATH, ort_reader_params, log_file_ort, )

            wait_for_signal_file_to_start_DMD(ort_process)

            counter=0
            while True:
                
                print(f'===========[ {counter} ]=============')

                # Launch DMD off receiver listening thread
                DMD_off_listening_thread = launch_dmd_off_receiver()
            
                # Launch VEC receiver and confirmer thread
                vec_receiver_confirmer_thread = launch_vec_receiver_confirmer()

                # Start DMD projector                
                # DMD_thread = launch_DMD_process_thread()

                print('Threads launched - Waiting for VEC...')
                while ( not vec_received_confirmed_event.is_set() ):
                    if global_stop_event.is_set():
                        raise CustomException("GlobalStopEvent: Main thread stopped by global stop event")
                    pass
                else:
                    print('Confirmed reception of VEC - Main thread can continue') 
                    counter += 1
                    continue 
                
                # print('No VEC received - Main thread will stop-- ?')


                # print("Stopping communication threads ( VEC and DMD )")
                # global_stop_event.set()
                # print("Joining communication thread")
                # vec_receiver_confirmer_thread.join()
                # print("Joining DMD listened thread")
                # DMD_off_listening_thread.join()
                # print("Joining DMD thread")
                # DMD_thread.join()   
        
                
                #break
                # TO absolulety check. I an not closing the socket where i am waiting for
                # the response. This means that if some vec file is in the queue, 
                # I might get it in the next loop
                #time.sleep(3)
                #break
                #continue
       
        except KeyboardInterrupt:
            print("Key Interrupt")    
        finally:
            # If it has not be set by a thread, set the global stop event to stop the threads
            if not global_stop_event.is_set(): 
                print("Stopping threads ")
                global_stop_event.set()

            # Then join them
            time.sleep(0.2)
            join_treads([DMD_off_listening_thread, DMD_thread, vec_receiver_confirmer_thread])
            
            # Terminate subprocesses
            terminate_DMD_queue(process_queue_DMD)
            terminate_ort_process(ort_process)
            
            # Close the sockets
            print('Closing VEC dedicated socket...')
            close_socket(rep_socket_vec)
            print('Closing DMD dedicated socket...')
            close_socket(rep_socket_dmd)

            # Clean up the signal file if it exists
            if os.path.exists(signal_file):
                os.remove(signal_file)
                print("Signal file cleanup complete.")
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    