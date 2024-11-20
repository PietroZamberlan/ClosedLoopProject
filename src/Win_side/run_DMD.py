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

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        
def threaded_wait_for_vec( rep_socket, confirmation_sent_event, failure_event, stop_event ):

    wait_response_timeout_sec = 4
    
    # Poll the socket for a reply with a timeout
    poller = zmq.Poller()
    poller.register(rep_socket, zmq.POLLIN)

    start_time = time.time()    
    print(f'...Thread: Waiting for vec file, timeout {wait_response_timeout_sec} seconds')
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        #print(f'elapsed time {elapsed_time}')
        socks = dict(poller.poll(timeout=10))             
        if rep_socket in socks:
            vec_content = rep_socket.recv_string()
            print("...Thread: VEC received")
            with open("received.vec", "w") as file:
                file.write(vec_content)
                # Send confirmation back to the server
                rep_socket.send_string("VEC CONFIRMED")
                confirmation_sent_event.set()
                print('...Thread: Confirmation sent')
            return
        if elapsed_time > wait_response_timeout_sec:
            print("...Thread: Timeout expired waiting for VEC from Linux")
            failure_event.set()
            return
        else:
            pass
    else:
        print('...Thread: Stopped from outside')
        return

def threaded_DMD(DMD_exe_path, DMD_exe_dir, input_data, process_queue, log_file=None):
    try:
        DMD_process = subprocess.Popen([DMD_exe_path], cwd=DMD_exe_dir,
                                        stdin=subprocess.PIPE, stdout=log_file,
                                        stderr=log_file, text=True,
                                          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP )    
        process_queue.put(DMD_process)
        # DMD autofill
        stdout, stderr = DMD_process.communicate(input=input_data, timeout=3)
    except subprocess.TimeoutExpired:
        print('...DMD Thread: Timeout expired')
        DMD_timeout_event.set()
        return

# Threading variables
DMD_timeout_event       = threading.Event()
process_queue           = queue.Queue()
# Define the executables paths
DMD_exe_path = r"C:/Users/user/Repositories/cppalp/x64/Release/film.exe"
DMD_exe_dir = r"C:/Users/user/Repositories/cppalp/x64/Release/"
#DMD_exe_path = r"C:\Users\user\Repositories\cppalp\x64\Release\film.exe"


# DMD parameters
pietro_dir = "21"
bin_number = "0"
vec_number = "0"
frame_rate = "30"
advanced_f = "y"
n_frames_LUT ="100"

exe_params = [pietro_dir, bin_number, vec_number, frame_rate, advanced_f, n_frames_LUT]
input_data = "\n".join(exe_params)+"\n"
signal_file = "signal_file.txt"

with open("ort_reader_output.log", "w") as log_file:
    try:
        while True:
            # Start DMD from the right directory
            print('Launching DMD subprocess thread')                     
            DMD_timeout_event.clear()
            args_DMD_thread = (DMD_exe_path, DMD_exe_dir, input_data, process_queue, log_file)
            DMD_thread      = threading.Thread(target=threaded_DMD, args=args_DMD_thread) 
            DMD_thread.start()
            while not DMD_timeout_event.is_set():
                pass
            else:
                print('DMD timed out...')
                DMD_timeout_event.clear()
                
            print("Terminating DMD subprocess")
            DMD_process = process_queue.get()
            if DMD_process.poll() is None: 
                print('Terminating DMD process')
                DMD_process.terminate()
                print('DMD process terminated')
            print("Joining DMD thread")
            DMD_thread.join()           
 
            
            break
    
            
    except KeyboardInterrupt:
        print("Process interrupted")    
        
    finally:    
        print("Terminating subprocesses...")
        if not process_queue.empty():
            DMD_process = process_queue.get()
            if DMD_process.poll() is None: 
                print("Terminating DMD subprocess")
                DMD_process.terminate()
        print('DMD process terminated')
        print("Joining DMD thread")
        DMD_thread.join()

