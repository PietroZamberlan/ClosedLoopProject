# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:46:42 2024

@author: user
"""

"""
Automation script to run the DMD with film.exe
"""

import subprocess
import time
import os

# Define the executables paths
#DMD_exe_path = r"C:/Users/user/Repositories/cppalp/x64/Release/film.exe"
DMD_exe_path = r"C:\Users\user\Repositories\cppalp\x64\Release\film.exe"
DMD_exe_dir = r"C:\Users\user\Repositories\cppalp\x64\Release"

# DMD parameters
pietro_dir = "21"
bin_number = "0"
vec_number = "0"
frame_rate = "40"
advanced_f = "n"

exe_params = [pietro_dir, bin_number, vec_number, frame_rate, advanced_f]

print(f"Executable working directory: {DMD_exe_dir}")
if not os.path.exists(DMD_exe_dir):
    print("Error: The specified directory does not exist.")

try:
    # Prepare the input for the executable
    input_data = "\n".join(exe_params) + "\n"
    
    # Run the executable from the right directory
    DMD_process = subprocess.Popen([DMD_exe_path], cwd=DMD_exe_dir, stdin=subprocess.PIPE, text=True )
    
    # Provide the input to the executable
    DMD_process.communicate(input=input_data)
    
    # Wait for both processes to complete
    DMD_process.wait()    
except KeyboardInterrupt:
    print("Terminating subprocesses...")
    DMD_process.terminate()


