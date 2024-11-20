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
        
def threaded_wait_dmd_off( rep_socket_dmd, dmd_off_event, stop_event):
    try:
        # Poll the socket for command with a timeout
        
        wait_response_timeout_sec = 8
        start_time = time.time()
    
        poller = zmq.Poller()
        poller.register(rep_socket_dmd, zmq.POLLIN)
        
        print(f'...DMD Waiter Thread: Waiting for command, timeout {wait_response_timeout_sec} seconds')
        
        while not stop_event.is_set():
            elapsed_time = time.time() - start_time
            socks = dict(poller.poll(timeout=100))             
            if rep_socket_dmd in socks:
                request = rep_socket_dmd.recv_string()
                rep_socket_dmd.send_string(request)
                print("...DMD Waiter Thread: Command received and confirmed")   
                dmd_off_event.set()
                return
            if elapsed_time > wait_response_timeout_sec:
                print("...DMD Waiter Thread: Timeout expired")
                dmd_off_event.set()
                break
        else:
            dmd_off_event.set()
            print(f'{start_time - time.time()}')
            print('...DMD Waiter Thread: Stopped from outside')
        print('...DMD Waiter Thread: off event set')
        return
    except:
        dmd_off_event.set()
        stop_event.set()
        print('...DMD Waiter Thread: EXCEPTION Stop and off event set')
        return

def threaded_wait_for_vec( rep_socket_vec, confirmation_sent_event, failure_event, stop_event ):
    
    if stop_event.is_set():
        print('+++++++')

    timeout_vec  = 6
    start_time   = time.time()    
    elapsed_time = 0
    # Poll the socket for a reply with a timeout
    poller = zmq.Poller()
    poller.register(rep_socket_vec, zmq.POLLIN)

    print(f'...VEC Thread: Waiting for vec file, timeout {timeout_vec} seconds')
    while not stop_event.is_set() and elapsed_time < timeout_vec:
        elapsed_time = time.time() - start_time
        socks = dict(poller.poll(timeout=10)) # milliseconds 
        if rep_socket_vec in socks:
            vec_content = rep_socket_vec.recv_string()
            print("...VEC Thread: VEC received")
            with open("received.vec", "w") as file:
                file.write(vec_content)
                # Send confirmation back to the server
                rep_socket_vec.send_string("VEC CONFIRMED")
                confirmation_sent_event.set()
                print('...VEC Thread: Confirmation sent')
                return
    else:
        if stop_event.is_set():
            print('...VEC Thread: Stopped from outside')
            failure_event.set()
            return
     
    print("...VEC Thread: Timeout expired")
    failure_event.set()
    return

def threaded_DMD(DMD_exe_path, DMD_exe_dir, input_data, process_queue, dmd_off_event, log_file=None):
    
    try:
        DMD_process = subprocess.Popen([DMD_exe_path], cwd=DMD_exe_dir,
                                       stdin=subprocess.PIPE, stdout=log_file,
                                       stderr=log_file, text=True )    
        process_queue.put(DMD_process)
        # DMD autofill, we need the process to be timed out to be able to close it
        stdout, stderr = DMD_process.communicate(input=input_data, timeout=0.1)
    except subprocess.TimeoutExpired:
        print('...DMD Thread: Timeout expired, DMD can be terminated')
            
    while not dmd_off_event.is_set():
        pass
    else:
        DMD_process.terminate()
        print('...DMD Thread process terminated')
        return
        
        
# Threading variables
confirmation_sent_event = threading.Event()
failure_event           = threading.Event() # failure to receive vec file
stop_event              = threading.Event()
dmd_off_event           = threading.Event()
process_queue           = queue.Queue()

# Define the executables paths
DMD_exe_path = r"C:/Users/user/Repositories/cppalp/x64/Release/film.exe"
DMD_exe_dir = r"C:/Users/user/Repositories/cppalp/x64/Release/"
#DMD_exe_path = r"C:\Users\user\Repositories\cppalp\x64\Release\film.exe"
ord_reader_path = r"C:/Users/user/ort/McsUsbNet_Examples/Examples/Python/ort_reader.py"

# DMD parameters
pietro_dir = "21"
bin_number = "0"
vec_number = "0"
frame_rate = "30"
advanced_f = "y"
n_frames_LUT ="15"

exe_params = [pietro_dir, bin_number, vec_number, frame_rate, advanced_f, n_frames_LUT]
input_data = "\n".join(exe_params)+"\n"
signal_file = "signal_file.txt"

# ort reader parameters
LINUX_IP = "172.17.12.179"
ord_reader_params = ["-ip", LINUX_IP, "-p", "5555", "--buffer", "1024", "toto.raw"]

# Listening socket
context     = zmq.Context()

rep_socket_vec = context.socket(zmq.REP)
rep_socket_vec.bind("tcp://0.0.0.0:5557")

rep_socket_dmd = context.socket(zmq.REP)
rep_socket_dmd.bind("tcp://0.0.0.0:5558")
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('   Windows client is running...')

# Opening file to redirect output of ort process
with open("ort_reader_output.log", "w") as log_file:
    try:
        # Run the Python script with the parameters
        print("Launching acquisition...")
        ort_process = subprocess.Popen(["python", ord_reader_path] + ord_reader_params, 
                                       stdout=log_file, stderr=log_file)
        print("Acquisition is running...")
        time.sleep(3)
        while not os.path.exists(signal_file):
            print("Waiting for the signal file to start DMD...")
            time.sleep(0.5)

        counter=0
        while True:
                
                print(f'===========[ {counter} ]=============')
                print('Launching DMD OFF listening thread')
                # Launch parallel function to listen for DMD off command
                dmd_off_event.clear()
                stop_event.clear()               
                args = (rep_socket_dmd, dmd_off_event, stop_event)
                DMD_listening_thread = threading.Thread(target=threaded_wait_dmd_off, args=args) 
                DMD_listening_thread.start()                
            
                print('Launching VEC confirmation thread')
                # Launch parallel function to wait for response
                confirmation_sent_event.clear()
                failure_event.clear()
                args = (rep_socket_vec, confirmation_sent_event, failure_event, stop_event)
                communication_thread = threading.Thread(target=threaded_wait_for_vec, args=args) 
                communication_thread.start()

                # Start DMD from the right directory
                print('Launching DMD subprocess thread')                     
                args_DMD_thread = (DMD_exe_path, DMD_exe_dir, input_data, process_queue, dmd_off_event, log_file)
                DMD_thread      = threading.Thread(target=threaded_DMD, args=args_DMD_thread) 
                DMD_thread.start()
                print('Waiting for VEC...')
                while ( not confirmation_sent_event.is_set() and 
                        not failure_event.is_set() ):
                    pass
                if failure_event.is_set():
                    print('No Vec received')
                if confirmation_sent_event.is_set():
                    print('Confirmed reception of VEC')  

                print("Stopping communication threads ( VEC and DMD ) ")
                stop_event.set()
                print("Joining communication thread")
                communication_thread.join()
                print("Joining DMD listened thread")
                DMD_listening_thread.join()
                
                print("Joining DMD thread")
                DMD_thread.join()   
        
                
                #break
                # TO absolulety check. I an not closing the socket where i am waiting for
                # the response. This means that if some vec file is in the queue, 
                # I might get it in the next loop
                #time.sleep(3)
                #break
                if counter == 5:
                    break
                else:
                    counter+=1
                    continue
        
    except KeyboardInterrupt:
        print("Key Interrupt")    
    finally:
        if not stop_event.is_set(): 
            print("Stopping communication thread")
            stop_event.set()
        if communication_thread.is_alive():
            print("Joining communication thread")
            communication_thread.join()
        
        print("Terminating subprocesses...")
        if not process_queue.empty():
            DMD_process = process_queue.get()
            if DMD_process.poll() is None: 
                print("Terminating DMD subprocess")
                DMD_process.terminate()
        print('DMD process terminated')
        print("Joining DMD thread")
        DMD_thread.join()
        print("Terminating ort subprocess")
        if ort_process.poll() is None: ort_process.terminate()
        
        print('Closing vec req socket...')
        rep_socket_vec.setsockopt(zmq.LINGER, 0)
        rep_socket_vec.close()
        print('Closing dmd req socket...')
        rep_socket_dmd.setsockopt(zmq.LINGER, 0)
        rep_socket_dmd.close()

        # Clean up the signal file if it exists
        if os.path.exists(signal_file):
            os.remove(signal_file)
        print("Cleanup complete.")
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    