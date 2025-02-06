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

Win_side_path = '.\\src\\Win_side\\' #


class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def wait_for_signal_file_to_start_DMD(ort_process):
    '''Waits for signal file to be written by the ort reader process
        to start the DMD projector.
    
        Args:
            ort_process: The process object of the ort reader process ( the script that launches the acquisition from the MEA device )
        
        Raises:
            CustomException: If the ort reader process fails to start
                NoDeviceConnected: If the ort reader does not find a connected MEA device - return code 1

        
    '''
    while not os.path.exists(signal_file):
        print("Waiting for the signal file to start DMD...")
        return_code = ort_process.poll()
        if return_code == 1: # Device not connected
            raise CustomException("NoDeviceConnected: No MEA device connected")
        if return_code is None:
            pass
        time.sleep(0.5)
    else:
        print("Signal file to start DMD found...")
        return

def launch_ort_process( ord_reader_path, ord_reader_params, log_file_ort ):
    '''Launches the ort reader process to acquire data from the MEA device.
        
        Sleeps for 2.5 the main thread to wait for ort_reader to get to the point 
        at which it writes the signal file or fails cause no device is connected.
        
    Args:
        ord_reader_path: The path to the Python script that launches the acquisition
        ord_reader_params: The parameters to pass to the Python script'''
    # Run the Python script with the parameters
    print("Launching acquisition...")
    # We set the -u flag to avoid buffering the output ( it would stop the OnChannelData to print the data in real time )
    ort_process = subprocess.Popen(["python", '-u', ord_reader_path] + ord_reader_params, 
                                    stdout=log_file_ort, stderr=log_file_ort, text=True)
    print("Acquisition is running...")

    def flush_log():
        while ort_process.poll() is None:
            log_file_ort.flush()
            time.sleep(1)

    flush_thread = threading.Thread(target=flush_log)
    flush_thread.start()

    time.sleep(2.5) # the ort process takes some time to get to the point in which it writes the signal file, so we wait a bit instead of printing a lot of "waiting for signal file"

    return ort_process

def threaded_rcv_dmd_off_signal( rep_socket_dmd, dmd_off_event, global_stop_event):
    
    '''
    Threaded function to wait for the signal to turn of the DMD coming from the 
    Linux machine.
    
    Sets:
        wait_response_timeout_sec: How much time to wait before stopping the
            DMD without signal from Linux. 
            The DMD is not prasenting more than 3 images after all. Only so 
            much time should be needed.
    
    '''
    
    try:
        # Poll the socket for command with a timeout
        wait_response_timeout_sec = 8
        start_time = time.time()
        poller = zmq.Poller()
        poller.register(rep_socket_dmd, zmq.POLLIN)
        print(f'...DMD Off cmd Receiver Thread: Waiting for command, timeout {wait_response_timeout_sec} seconds')
        
        while not global_stop_event.is_set():
            elapsed_time = time.time() - start_time
            socks = dict(poller.poll(timeout=100))             
            if rep_socket_dmd in socks:
                request = rep_socket_dmd.recv_string()
                rep_socket_dmd.send_string(request)
                print("...DMD Off cmd Receiver Thread: Stop command received and confirmed")   
                dmd_off_event.set()
                return
            if elapsed_time > wait_response_timeout_sec:
                print("...DMD Off cmd Receiver Thread: Timeout expired")
                dmd_off_event.set()
                print('...DMD Off cmd Receiver Thread: off event set')
                return
        else:
            dmd_off_event.set()
            print('...DMD Off cmd Receiver Thread: Stopped from outside')
            print('...DMD Off cmd Receiver Thread: off event set')
            return
    except:
        dmd_off_event.set()
        global_stop_event.set()
        print('...DMD Off cmd Receiver Thread: EXCEPTION encountered - global_stop_event and dmd_off_event set')
        return

def threaded_wait_for_vec( rep_socket_vec, vec_received_confirmed_event, global_stop_event ):
    ''' 
    Waits for the VEC file to be received from the Linux machine.

    Once it is, it sends a confirmation, so that the Linux machine knows packets sent from
    then on could be relevant and stops discarding them.

    Sets:
        vec_received_confirmed_event: To allow the main thread to continue after VEC ha sbeen received and overwritten
                                      It also signals that confirmation has been sent to the Linux machine to stop discarding packets.
        global_stop_event:            To True if the timeout expires when waiting for the vec file 
                                      It is the Linux machine responsibility to send the VEC file ( even a random one) 
                                      if the fit is taking too long

    Returns:
        None : And only if vec file is received and confirmed, and the global_stop_event is not set                                      
    '''

    try:
        timeout_vec  = 6
        start_time   = time.time()    
        elapsed_time = 0
        # Poll the socket for a reply with a timeout
        poller = zmq.Poller()
        poller.register(rep_socket_vec, zmq.POLLIN)

        print(f'...VEC Thread: Waiting for vec file, timeout {timeout_vec} seconds')
        while not global_stop_event.is_set() and elapsed_time < timeout_vec:
            elapsed_time = time.time() - start_time
            socks = dict(poller.poll(timeout=10)) # milliseconds 
            if rep_socket_vec in socks:
                vec_content = rep_socket_vec.recv_string()
                print("...VEC Thread: VEC received")
                # Wait for authorization to overwrite the vec file. If this event is not set
                # it means the DMD is still showing the images (using the .VEC)
                allow_vec_changes_event.wait()
                with open("received.vec", "w") as file:
                    file.write(vec_content)
                    # Send confirmation back to the server
                    rep_socket_vec.send_string("VEC CONFIRMED")
                    vec_received_confirmed_event.set()
                    print('...VEC Thread: Confirmation sent')
                    return
        else:
            if global_stop_event.is_set():
                print('...VEC Thread: Stopped from outside')
                return
        # If the timeout expires, we set the global stop - 
        # It is the Linux machine responsibility to send the VEC file
        print("...VEC Thread: Timeout expired and no VEC received, setting global_stop_event")
        global_stop_event.set()
        # failure_event.set() # there was this line here, for an event that i removed sent to this function. I think its useless
        return
    except:
        global_stop_event.set()
        print('...VEC Thread: EXCEPTION encountered - global_stop_event set')
        return
    
def threaded_DMD(DMD_exe_path, DMD_exe_dir, input_data_DMD, process_queue_DMD, dmd_off_event, global_stop_event, log_file_DMD=None):
    '''
    Launches the DMD projector process with the .VEC file of images specified in input_data_DMD.

    It also listens for the dmd_off_event catched by threaded_rcv_dmd_off_signal() to terminate the process.

    Before doing so, we need to wait for the process to be timed out, this is why we set a very short
    timeout of 0.1 with .communicate().

    Sets:
      allow_vec_changes_event: to False, to prevent changes to the VEC file that vec_receiver_confirmer_thread could do    
                            Only when the DMD terminated the VEC modifications are freed.
    
    '''
    try:
        DMD_process = subprocess.Popen([DMD_exe_path], cwd=DMD_exe_dir,
                                       stdin=subprocess.PIPE, stdout=log_file_DMD,
                                       stderr=log_file_DMD, text=True )    
        process_queue_DMD.put(DMD_process)
        # DMD autofill, we need the process to be timed out to be able to close it
        stdout, stderr = DMD_process.communicate(input=input_data_DMD, timeout=0.1)
    except subprocess.TimeoutExpired:
        print('...DMD Thread: DMD was launched and can be terminated')
            
    try:        
        while not dmd_off_event.is_set():
            pass
        else:
            DMD_process.terminate()
            # Wait for authorization to overwrite the vec file. If this event is not set
            # it means
            allow_vec_changes_event.set()
            print('...DMD Thread: DMD process terminated, VEC file can be modified')
            return
        
    except:
        print('...DMD Thread: EXCEPTION encountered - global_stop_event set')
        global_stop_event.set()
        return

def launch_dmd_off_receiver():
    '''Launches the DMD Off cmd Receiver Thread
    
    Sets:
        dmd_off_event: to False, to receive the signal to turn off the DMD
        global_stop_event: to False, to allow stopping the thread from outside
    '''
    
    print('Launching DMD off cmd receiver listening thread')
    # Launch parallel function to listen for DMD off command
    dmd_off_event.clear()
    global_stop_event.clear()               
    args = (rep_socket_dmd, dmd_off_event, global_stop_event)
    DMD_off_listening_thread = threading.Thread(target=threaded_rcv_dmd_off_signal, args=args) 
    DMD_off_listening_thread.start()                
    return DMD_off_listening_thread

def launch_vec_receiver_confirmer():
    '''Launches the VEC receiver and confirmer thread

        It will wait for the VEC from Linux machine

        Once received, it will overwrite the VEC if that is allowed by the allow_vec_changes_event set
        (by the DMD thread).

        Only after overwriting it will allow the main thread to continue with vec_received_confirmed_event is set.
        (vec_received_confirmed_event also confirms that the Linux machine has stopped discarding packets)


    Sets:
        vec_received_confirmed_event: to False, this stops the main thread until the VEC is received
                                    and confirmed by threaded_wait_for_vec
        failure_event:                to False, 
        global_stop_event:            to False, to allow stopping the thread from outside
    '''

    print('Launching VEC confirmation thread')
    # Launch parallel function to wait for response
    vec_received_confirmed_event.clear()
    args = (rep_socket_vec, vec_received_confirmed_event, global_stop_event)
    vec_receiver_confirmer_thread = threading.Thread(target=threaded_wait_for_vec, args=args) 
    vec_receiver_confirmer_thread.start()
    return vec_receiver_confirmer_thread

def launch_DMD_process_thread():
    '''
    Launches the thread that starts the DMD projector.
    The thread will:
        put the process in a state in which it can be terminated
        
        wait for signal dmd_off_event set by threaded_rcv_dmd_off_signal to turn off the DMD

        set allow_vec_changes_event: to False, to prevent changes to the VEC file that vec_receiver_confirmer_thread could do    
                                    Only when the DMD terminated the VEC can be modified
    '''
    print('Launching DMD subprocess thread')
    # Wait for authorization to overwrite the vec file. If this event is not set
    # it means the DMD is still showing the images (using the .VEC)
    allow_vec_changes_event.clear()                
    args_DMD_thread = (DMD_exe_path, DMD_exe_dir, input_data_DMD, 
                    process_queue_DMD, dmd_off_event, global_stop_event, log_file_DMD)
    DMD_thread      = threading.Thread(target=threaded_DMD, args=args_DMD_thread) 
    DMD_thread.start()
    return DMD_thread

def join_treads(threads):
    for thread in threads:
        if thread is not None and thread.is_alive():
            print(f"Joining {thread.name}")
            thread.join()
        else:
            print(f"Thread {thread.name} was not alive")

def terminate_DMD_queue(queue):
    # Terminate the DMD projector subprocess, if it has not been handeled by the DMD thread
    # The process is in a queue of which its the only element.
    # We defined it like this cause its needed to be able to terminate it
    print("Terminating DMD projector subprocess...")
    if not queue.empty():
        DMD_process = queue.get()
        if DMD_process.poll() is None: 
            print("Terminating DMD subprocess")
            DMD_process.terminate()
    print('DMD process terminated')

def terminate_ort_process(ort_process):
    print("Terminating ort subprocess")
    if ort_process.poll() is None: 
        ort_process.terminate()
        log_file_ort.flush()
    print('ort process terminated')

def close_socket(socket):
    socket.setsockopt(zmq.LINGER, 0)
    socket.close()

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

# Define the executables paths
DMD_exe_path = r"C:/Users/user/Repositories/cppalp/x64/Release/film.exe"
DMD_exe_dir = r"C:/Users/user/Repositories/cppalp/x64/Release/"
#DMD_exe_path = r"C:\Users\user\Repositories\cppalp\x64\Release\film.exe"
ord_reader_path = r"C:\Users\user\ClosedLoopProject\src\Win_side\ort_reader.py"

# DMD parameters
pietro_dir = "21"
bin_number = "0"
vec_number = "0"
frame_rate = "30"
advanced_f = "y"
n_frames_LUT ="15"

exe_params = [pietro_dir, bin_number, vec_number, frame_rate, advanced_f, n_frames_LUT]
input_data_DMD = "\n".join(exe_params)+"\n"
signal_file = "signal_file.txt"

# ort reader parameters
LINUX_IP = "172.17.12.200"
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

with open(f"{Win_side_path}output_ort_reader.log", "w") as log_file_ort, \
    open(f"{Win_side_path}output_DMD.log", "w") as log_file_DMD:
        try:

            ort_process = launch_ort_process(ord_reader_path, ord_reader_params, log_file_ort, )

            wait_for_signal_file_to_start_DMD(ort_process)

            counter=0
            while True:
                
                print(f'===========[ {counter} ]=============')

                # Launch DMD off receiver listening thread
                DMD_off_listening_thread = launch_dmd_off_receiver()
            
                # Launch VEC receiver and confirmer thread
                vec_receiver_confirmer_thread = launch_vec_receiver_confirmer()

                # Start DMD projector                
                DMD_thread = launch_DMD_process_thread()

                print('Threads launched - Waiting for VEC...')
                while ( not vec_received_confirmed_event.is_set() ):
                    if global_stop_event.is_set():
                        raise CustomException("GlobalStopEvent: Main thread stopped by global stop event")
                    pass
                else:
                    print('Confirmed reception of VEC - Main thread can continue') 
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
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    