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

# ort reader parameters
LINUX_IP = "172.17.12.179"
ord_reader_params = ["-ip", LINUX_IP, "-p", "5555", "--buffer", "1024", "toto.raw"]

# Listening socket
context     = zmq.Context()
#pull_socket = context.socket(zmq.PULL)
#pull_socket.bind("tcp://*:5556")
rep_socket = context.socket(zmq.REP)
rep_socket.bind(f"tcp://0.0.0.0:5557")
print('   Windows client is running...')

try:
    # Opening file to redirect output of ort process
    with open("ort_reader_output.log", "w") as log_file:
        # Run the Python script with the parameters
        print("Launching acquisition...")
        ort_process = subprocess.Popen(["python", ord_reader_path] + ord_reader_params, stdout=log_file, stderr=log_file)
        print("Acquisition is running...")
        time.sleep(3)
        signal_file = "signal_file.txt"
        while not os.path.exists(signal_file):
            print("Waiting for the signal file to start DMD...")
            time.sleep(0.5)
        # Run the executable with the parameters
        DMD_process = subprocess.Popen([DMD_exe_path], cwd=DMD_exe_dir, stdin=subprocess.PIPE, text=True )
        # Provide the input to the executable
        DMD_process.communicate(input=input_data)
        
        print("Waiting for response from Linux machine...")
        vec_content = rep_socket.recv_string()
        print(f"Received response:")
        with open("received_vec.vec", "w") as file:
            file.write(vec_content)
        # Send confirmation back to the server
        rep_socket.send_string("CONFIRMED")
        
        # Wait for both processes to complete
        #ort_process.wait()    
        #DMD_process.wait()
    
except KeyboardInterrupt:
    print("Terminating subprocesses...")
    #ort_process.terminate() 
    #DMD_process.terminate()

finally:
    # Clean up the signal file if it exists
    if os.path.exists(signal_file):
        os.remove(signal_file)
    print("Cleanup complete.")