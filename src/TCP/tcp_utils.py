import zmq
import os
import sys
import torch
import threading
import time
import json
import base64
import numpy as np
import queue
import logging

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR    = os.path.join(CURRENT_DIR, '../../')
sys.path.insert(0, os.path.abspath(REPO_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE} from tcp_utils")

def count_triggers( trigger_ch_sequence, trigger_diff_threshold=2000, trigger_threshold=51000):

    '''Counts the triggers in the provided array, tipically an quired buffer
    trigger_ch_sequence: np.array - the sequence of signals aquired from the trigger channel
    trigger_diff_threshold: int - the different threshold between two sonsecutive signals in the sequence to be considered a trigger

    '''
    start = 0
    end = trigger_ch_sequence.shape[0]

    # trigger counts in the sequence
    n_triggers = 0

    latest_diffs = np.array([]) # array to store the last 10 signal differences. If any of these is under the trigger threshold, we are still treating the same trigger
    detected_trigger_idx = np.array([]) # array to store the indexes of the triggers detected
    trg_close_to_end   = False
    trg_close_to_start = False
    for j in range(start, end-1):
        last_diff = (trigger_ch_sequence[j+1] - trigger_ch_sequence[j])

        if last_diff >= trigger_diff_threshold:
            # if none of the latest 10 differences was above trigger_diff_threshold, then we have a new trigger
            if np.all(latest_diffs < trigger_diff_threshold):
                n_triggers += 1
                detected_trigger_idx = np.append(detected_trigger_idx, j)
                # if this trigger has been detected in the first or last 10 detections, flag the buffer
                if end-j < 10 :
                    # print(f" last diff was {last_diff} ")
                    trg_close_to_end = True
                elif j < 10:
                    trg_close_to_start = True

        latest_diffs = np.append(latest_diffs, last_diff)
        if latest_diffs.shape[0] > 10:
            latest_diffs = latest_diffs[1:]
    detected_trigger_idx = detected_trigger_idx.astype(int)

    return n_triggers, detected_trigger_idx, trg_close_to_end, trg_close_to_start

def update_fit(new_spike_count, print_lock):

    '''
    Update the GP variational (m,V) and likelihood (A, lambda0) parameters using the latest revceived spike count
    return the id of the new, most informative image to display next
    
    Args:
        new_spike_count (int): The number of spikes received after the image was displayed ( in the relevant time interval T )
        result_queue : the queue to store the results of the fit 
    '''
    with print_lock:
        print(f"\n...Thread: Updating the fit using {new_spike_count} spikes...")
    new_spike_count = torch.tensor(new_spike_count, device=DEVICE)
    # with print_lock:
        # print(f"\n...Thread: New_spike_count is on device: {new_spike_count.device}")
    result = new_spike_count*2
    time.sleep(1)
    return result.to('cpu')

def fit_end_queue_img(new_spike_count, img_ID_queue, threadict):
    '''
    Add the result of update_fit to the queue of the thread results
    '''
    imgID = update_fit(new_spike_count, threadict['print_lock'])
    img_ID_queue.put(imgID)
    with threadict['print_lock']:
        print(f"\n...Thread: imdID {imgID} added to the queue: {imgID}")
    threadict['fit_finished_event'].set()
    return

def generate_vec_file(chosen_img_ID, rndm_img_id, max_gray_trgs=10, max_img_trgs=10, ending_gray_trgs=10):
    """
    Generate the VEC file for the chosen image ID and the random image ID., with the following structure:
    0 {max_total_frames} 0 0 0
    0 0 0 0 0               [max_gray_trgs lines]
    0 {chosen_img_ID} 0 0 0 [max_img_trgs lines]
    0 0 0 0 0               [max_grey_trgs lines]
    0 {rndm_img_id} 0 0 0   [max_img_trgs lines]
    0 0 0 0 0               [ending_gray_trgs lines]

    Parameters:
    img_ID (int): The image ID.
    rndm_img_id (int): The random image ID.
    max_gray_trgs (int): The number of lines representing the STARTING gray image.
    ending_gray_trgs (int): The number of lines representing the ENDING gray image.
    max_img_trgs (int): The number of lines representing triggers of the natural image.
    """

    file_path = f'{REPO_DIR}/src/DMD/saved/vec_file_{chosen_img_ID}.txt'
    lines = []

    # Write the first line
    lines.append(f"0 {max_gray_trgs+max_img_trgs+max_gray_trgs+max_img_trgs+ending_gray_trgs} 0 0 0\n")
    # Write the following lines
    for _ in range(max_gray_trgs):     lines.append("0 0 0 0 0\n")
    for _ in range(max_img_trgs):      lines.append(f"0 {chosen_img_ID} 0 0 0\n")            
    for _ in range(max_gray_trgs):     lines.append("0 0 0 0 0\n")  
    for _ in range(max_img_trgs):      lines.append(f"0 {rndm_img_id} 0 0 0\n")            
    for _ in range(ending_gray_trgs):  lines.append("0 0 0 0 0\n")  

    file_content = ''.join(lines)
    with open(file_path, 'w') as file: 
        file.write(file_content)
              
    return file_content, file_path

def vec_send_and_confirm( chosen_img_ID, rndm_img_id, threadict, req_socket_vec,  max_gray_trgs=10, max_img_trgs=10, ending_gray_trgs=10):
    '''
    Generate the vec file.
    Send to the client the VEC file corresponding the the chosen image ID and wait for confirmation. When received, set the event and allow the main thread to 
    Stop discarding packets
    '''
    
    with threadict['print_lock']:
        print(f"\n...VEC Thread: Generating VEC file for image ID: {chosen_img_ID}", end="")
    vec_content, vec_path = generate_vec_file(chosen_img_ID=chosen_img_ID, rndm_img_id=rndm_img_id, max_gray_trgs=max_gray_trgs,
                                               max_img_trgs=max_img_trgs, ending_gray_trgs=ending_gray_trgs )
    with threadict['print_lock']:
        print(f"\n...VEC Thread: Sending VEC file for image ID: {chosen_img_ID}", end="")
    # Send the VEC file to the client and wait for confirmation
    req_socket_vec.send_string(vec_content)
    # Poll the socket for a reply with a timeout, basically wait for tot milliseconds for a reply
    poll_interval_vec = 100           # Milliseconds
    timeout_vec       = 3             # Seconds
    poller_vec        = zmq.Poller()
    poller_vec.register(req_socket_vec, zmq.POLLIN)
    start_time_vec = time.time()
    with threadict['print_lock']:
        print(f'\n...VEC Thread: Waiting VEC confirmation from the client, timeout in 3 seconds...', end="\n")

    while not threadict['global_stop_event'].is_set() and (time.time() - start_time_vec) < timeout_vec:    
        socks = dict(poller_vec.poll(timeout=poll_interval_vec))
        if req_socket_vec in socks:
            confirmation = req_socket_vec.recv_string()
            if confirmation == 'VEC CONFIRMED':
                threadict['vec_confirmation_event'].set()
                with threadict['print_lock']: print(f"\n...VEC Thread: Client confirmed VEC reception", end="")
                return
            else:
                logging.error(f"\n...VEC Thread: Error: The client replied with an unexpected message: {confirmation}", end="")
                threadict['vec_failure_event'].set() 
                return           
    if threadict['global_stop_event'].is_set():
        with threadict['print_lock']: 
            print(f"\n...VEC Thread: Global stop", end="")
        return
       
    print(f"\n...VEC Thread: Error: Timeout expired, the client did not confirm the VEC", end="")
    threadict['vec_failure_event'].set()        
    return

def dmd_off_send_confirm(req_socket_dmd, threadict):

    req_socket_dmd.send_string("DMD OFF")
    timeout_dmd    = 3
    with threadict['print_lock']: print(f"\n...DMD off Thread: command sent, response timeout {timeout_dmd} seconds", end="")
    poller = zmq.Poller()       
    poller.register(req_socket_dmd, zmq.POLLIN)
    start_time_dmd = time.time()

    while not threadict['global_stop_event'].is_set() and (time.time() - start_time_dmd) < timeout_dmd:    
        socks = dict(poller.poll(timeout=100))
        if req_socket_dmd in socks:
            confirmation = req_socket_dmd.recv_string() # The client replies sending back the same message
            if confirmation == "DMD OFF":
                threadict['DMD_stopped_event'].set()
                threadict['dmd_off_set_time'] = time.time()  # Record the time when the event is set
                print(f"\n...DMD off Thread: Client confirmed DMD off", end="")
                with threadict['print_lock']: print(f"\n...DMD off Thread: Client confirmed DMD off", end="")    
                return
            else: 
                with threadict['print_lock']: print(f"\n...DMD off Thread: Client replied to DMD request with unexpected message: {confirmation}", end="")        
                return
            
    if threadict['global_stop_event'].is_set():
        with threadict['print_lock']: print(f"\n...DMD off Thread: Global stop", end="")
        return

    with threadict['print_lock']: print(f"\n...DMD off Thread: Client didn't confirm DMD off, timeout", end="")        
    return

def time_since_event_set(event_set_time):
    if event_set_time is None:
        return None
    return time.time() - event_set_time

def threaded_dump(array, file_path):
    """
    Save a NumPy array to a .npy file in a separate thread.

    Parameters:
    array (numpy.ndarray): The NumPy array to save.
    file_path (str): The path to the file where the array will be saved.
    """
    def save_array():
        np.save(file_path, array)
        # print(f"Array saved to {file_path}")

    # Create and start a new thread to save the array
    thread = threading.Thread(target=save_array)
    thread.start()

