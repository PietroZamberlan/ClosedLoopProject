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

# repo_dir = '/home/idv-eqs8-pza/IDV_code/Variational_GP/Gaussian-Processes/Spatial_GP_repo'
# sys.path.insert(0, repo_dir)
from src.GP.Spatial_GP_repo import utils

class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '__ndarray__' in obj:
            data = base64.b64decode(obj['__ndarray__'])
            return np.frombuffer(data, dtype=obj['__dtype__']).reshape(obj['__shape__'])
        elif '__bytes__' in obj:
            return obj['__bytes__'].encode(obj['__encoding__'])
        return obj

WINDOWS_OLD_MACHINE_IP = '172.17.17.125'

context = zmq.Context()
# Create the publishing socket
pub_socket = context.socket(zmq.PUSH)
pub_socket.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:5556")

# Create the listening socket
# pull_socket = context.socket(zmq.STREAM)
pull_socket = context.socket(zmq.PULL)
pull_socket.bind("tcp://*:5555")

print("Linux server is running and waiting for data stream...")

GLOBAL_COUNTER = 0

def compute_and_publish(number, recept_time):
    # Example GPU operation using PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([number], device=device)
    result = tensor.cpu().numpy()  # Move result back to CPU if needed

    print(f"Sending: {result[0]}")
    # print only the first 4 digits
    pub_socket.send_string(f'{result[0]} with delay of {(time.time() - recept_time):.4f} seconds') 

def update_fit(new_spike_count):

    '''
    Update the GP variational (m,V) and likelihood (A, lambda0) parameters using the latest revceived spike count
    return the id of the new, most informative image to display next
    
    Args:
        new_spike_count (int): The number of spikes received after the image was displayed ( in the relevant time interval T )
        result_queue : the queue to store the results of the fit 
    '''
    print(f"      Updating the fit using {new_spike_count} spikes...")
    result = new_spike_count*2
    return result

def fit_and_add_imgID_to_queue(new_spike_count, img_ID_queue, finished_event):
    '''
    Add the result of update_fit to the queue of the thread results
    '''
    imgID = update_fit(new_spike_count)
    img_ID_queue.put(imgID)
    logging.info(f"Thread : imdID added to the queue: {imgID}")
    finished_event.set()

def dump_on_file(nparray, filename):
    np.save(filename, nparray)
    logging.info(f"Dumped array to {filename}")

# Example usage in a threaded function
def threaded_dump(nparray, filename):
    dump_thread = threading.Thread(target=dump_on_file, args=(nparray, filename))
    dump_thread.start()

# Thread variables
img_ID_queue = queue.Queue()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##### Receive the data stream #####

treating_img = False # Flag to indicate if the server is handling a relevant buffer
n_acquired_buffers = 0 # Number of buffers acquired after the first relevant buffer. 
n_relevant_buffers = 6 # Number of buffers to acquire after the first relevant buffer, this should depend on the buffeth time lenght and number of ms we want to keep, see below
trigger_threshold = 5.2*1e5
ch_id = 255
computation_finished_event = threading.Event() # Event to signal when the computation thread is finished
try:
    number = 0
    tot_peaks_after_img = 0
    # for i in range(1,4):
    while True:

        string_packet = pull_socket.recv_string()
        #region To use the saved packets only if the mea is not avaialble
        # with open(f'packet{i}.json', 'r') as f:
        #     string_packet = f.read()
        #endregion
            
        #region ____________________________    Unpack and check if the buffer has been sent after an image #####
        '''-___________________________
        The client is sending packets (dictionaries) with the following structure:

        {'buffer_nb': 10, 'n_peaks': 0,'peaks': {'ch_nb from 0 to 255': np.array(shape=n of peaks in buffer with 'timestamp') } }'}}
            - Unpackable using the custom Decoder class
            - buffer_nb: the number of the buffer
            - n_peaks: the number of peaks in the buffer, already computed by the client
            - peaks: dictionary containing the peaks in each channel
            -'trg_raw_data': the trigger channel raw data, unfiltered
        _____________________________'''

        packet = json.loads(string_packet, cls=Decoder)
        print(f'Packet {packet["buffer_nb"]} received')

        # print(f"Received packet {packet['buffer_nb']} with {packet['peaks']} peaks")

        '''Check if the packet is relevant, i.e. if it has been sent after an image
            to do this, check the trigger channel (127 on the MEA, so 126 here) it is above a certain threshold ( ~ 5.2*1e5)
            if it is update the peaks count for the current image
        '''


        # Dump the trigger channel data
        # threaded_dump(packet['trg_raw_data'], f'./saved/trg_ch_bf_{packet["buffer_nb"]}_2')

        # if the buffer never crosses the threshold, discard it
        
        if packet['trg_raw_data'].max() < 40000:
            n_acquired_buffers  = 0
            tot_peaks_after_img = 0
            print(f' Not relevant')
            continue
        #endregion
        print(f'Relevant')
        '''If packet is relevat, update the peaks count'''
        tot_peaks_after_img += packet['peaks'][str(ch_id)][0]
        n_acquired_buffers += 1

        '''Only when enough buffers have been received start a lateral thread to update the fit
           while the main thread continues to receive and discart the packets
           If sampling at 20kHz, with buffer size 1024 -> (20kHz/1024)*300ms/1000ms ~ 6 buffers per 300ms
           '''
        if n_acquired_buffers < n_relevant_buffers:
            print('  Waiting for more buffers...')
            continue
        print(f'   Acquired {n_acquired_buffers} buffers after an image, enough for {(20000/1024)*(1000/n_acquired_buffers):.2f}ms')

        print(f"    {tot_peaks_after_img} peaks after last image")
        n_acquired_buffers = 0

        print(f"     Starting the computation thread with {tot_peaks_after_img} peaks")
        computation_finished_event.clear()  
        computation_thread = threading.Thread(target=fit_and_add_imgID_to_queue, 
                                              args=(tot_peaks_after_img, img_ID_queue, computation_finished_event))
        computation_thread.start()

        while not computation_finished_event.is_set():
            string_packet = pull_socket.recv_string()
            packet = json.loads(string_packet, cls=Decoder)
            print("...Discarding packet while waiting for computation to finish \n")

        logging.info('Main thread: computation_thread has finished, fit updated \n')

        new_img_id = img_ID_queue.get() #retrieves next available result and removes it from the queue
        # print(f"Next image to display: {new_img_id}")
        tot_peaks_after_img = 0
        # computation_thread.join()
        # continue
        #endregion

except KeyboardInterrupt:
    print("Server is shutting down...")
finally:
    print('Closing pull socket...')
    pull_socket.setsockopt(zmq.LINGER, 0)
    pull_socket.close()
    pub_socket.setsockopt(zmq.LINGER, 0)
    print('Closing push socket...')
    pub_socket.close()
    print('Closing context...')
    context.term()
    print('Joining threads...')
    # computation_thread.join()