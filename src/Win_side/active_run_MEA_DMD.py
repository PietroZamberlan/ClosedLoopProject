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

# Import the configuration file
from config.config import *
from win_utils import *

def active_run_MEA_DMD():
    # Threading variables
    '''
        threadict = {
        'vec_received_confirmed_event': vec_received_confirmed_event,
        'global_stop_event':            global_stop_event,
        'dmd_off_event':                dmd_off_event,
        'allow_vec_changes_event':      allow_vec_changes_event,
        'process_queue_DMD':            process_queue_DMD,
        'exceptions_q':                 exceptions_q
    }
    
    '''
    # Threads variables. 
    threadict, DMD_off_listening_thread, \
        DMD_thread, vec_receiver_confirmer_thread \
            = setup_win_thread_vars()

    # Temporary def of variables until all threads use threadict
    vec_received_confirmed_event = threadict['vec_received_confirmed_event']
    global_stop_event            = threadict['global_stop_event']
    dmd_off_event                = threadict['dmd_off_event']
    allow_vec_changes_event      = threadict['allow_vec_changes_event']
    process_queue_DMD            = threadict['process_queue_DMD']
    exceptions_q                 = threadict['exceptions_q']




    exe_params = [pietro_dir_DMD, bin_number, vec_number, frame_rate, advanced_f, n_frames_LUT]
    input_data_DMD = "\n".join(exe_params)+"\n"

    # ort reader parameters
    ort_reader_params = ["-ip", LINUX_IP, "--port", PULL_SOCKET_PACKETS_PORT, 
                        "--buffer_size", f'{buffer_size}', '--threshold_multiplier', f'{threshold_multiplier_init}', 
                        "--filename", raw_data_file_path]

    # Listening socket
    context     = zmq.Context()

    rep_socket_vec = context.socket(zmq.REP)
    rep_socket_vec.bind(f"tcp://0.0.0.0:{REQ_SOCKET_VEC_PORT}")

    rep_socket_dmd = context.socket(zmq.REP)
    rep_socket_dmd.bind(f"tcp://0.0.0.0:{REQ_SOCKET_DMD_PORT}")

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('   Windows client is running...')


    # Opening file to redirect output of ort process

    log_file_ort = open(ort_reader_output_pathname, "w")
    log_file_DMD = open(dmd_output_pathname, "w") 
    try:

        ort_process = launch_ort_process(
            ORT_READER_PATH, ort_reader_params, log_file_ort, 
            testmode=testmode)

        wait_for_signal_file_to_start_DMD(ort_process)

        img_pair_counter=0
        while True and not global_stop_event.is_set():
            
            print(f'===========[ {img_pair_counter} ]=============')

            # Launch DMD off receiver listening thread
            DMD_off_listening_thread = launch_dmd_off_receiver(
                rep_socket_dmd, threadict, timeout_dmd_off_rcv_phase2)
        
            # Launch VEC receiver and confirmer thread
            vec_receiver_confirmer_thread = launch_vec_receiver_confirmer(
                rep_socket_vec, threadict, timeout_vec_phase2, vec_pathname_dmd_source_active)

            # Start DMD projector                
            DMD_thread = launch_DMD_process_thread( 
                input_data_DMD, threadict, log_file_DMD,
                testmode=testmode)

            # Wait for the VEC to be received and confirmed before joining the threads and continuing
            print('Threads launched - Waiting for VEC...')
            wait_vec_start_time = time.time()
            while not vec_received_confirmed_event.is_set():
                if global_stop_event.is_set():
                    # raise CustomException("GlobalStopEvent: Main thread stopped by global stop event")
                    print("Main thread stopped by global stop event")
                    break
                pass

            else:
                print(f'Confirmed reception of VEC after {(time.time()-wait_vec_start_time):.3f} sec - Main thread can continue') 
                img_pair_counter += 1
            
                # Join the threads
                join_treads([DMD_off_listening_thread, DMD_thread, vec_receiver_confirmer_thread])
    
                continue 
            
            #break
            # TO absolulety check. I an not closing the socket where i am waiting for
            # the response. This means that if some vec file is in the queue, 
            # I might get it in the next loop
            #time.sleep(3)
            #break
            #continue

    except KeyboardInterrupt:
        print("Key Interrupt")    
        global_stop_event.set()
    finally:
        # If it has not be set by a thread, set the global stop event to stop the threads

        print(f' Global stop event is set? {global_stop_event.is_set()}')
        if not global_stop_event.is_set(): 
            print("Stopping threads ")
            global_stop_event.set()

        # Then join them
        time.sleep(0.2)
        join_treads([DMD_off_listening_thread, DMD_thread, vec_receiver_confirmer_thread])
        
        # Terminate subprocesses
        terminate_DMD_queue(process_queue_DMD)
        terminate_ort_process(ort_process, log_file_ort)
        
        # Close the sockets
        print('Closing VEC dedicated socket...')
        close_socket(rep_socket_vec)
        print('Closing DMD dedicated socket...')
        close_socket(rep_socket_dmd)

        # Clean up the signal file if it exists
        if os.path.exists(signal_file):
            os.remove(signal_file)
            print("Signal file cleanup complete.")
            
        print('Checking for exceptions in queue...')
        if not exceptions_q.empty():
            raise exceptions_q.get()
            

if __name__ == "__main__":
    
    active_run_MEA_DMD()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    