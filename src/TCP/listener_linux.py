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


current_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir    = os.path.join(current_dir, '../../')
sys.path.insert(0, os.path.abspath(repo_dir))

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                    print(f" last diff was {last_diff} ")
                    trg_close_to_end = True
                elif j < 10:
                    trg_close_to_start = True

        latest_diffs = np.append(latest_diffs, last_diff)
        if latest_diffs.shape[0] > 10:
            latest_diffs = latest_diffs[1:]
    detected_trigger_idx = detected_trigger_idx.astype(int)

    return n_triggers, detected_trigger_idx, trg_close_to_end, trg_close_to_start

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
    new_spike_count = torch.tensor(new_spike_count, device=DEVICE)
    print(f'      New_spike_count is on device: {new_spike_count.device}')
    result = new_spike_count*2
    return result.to('cpu')

def fit_and_add_imgID_to_queue(new_spike_count, img_ID_queue, finished_event):
    '''
    Add the result of update_fit to the queue of the thread results
    '''
    imgID = update_fit(new_spike_count)
    img_ID_queue.put(imgID)
    logging.info(f"   Thread : imdID added to the queue: {imgID}")
    finished_event.set()

def dump_on_file(nparray, filename):
    np.save(filename, nparray)
    logging.info(f"Dumped array to {filename}")

# Thread variables
img_ID_queue = queue.Queue()
# Configure logging
logging.basicConfig(level=logging.INFO, format='      %(message)s - %(asctime)s - %(levelname)s - ', datefmt='%M:%S')

##### Receive the data stream #####

count_relevant_buffs = 0 # Number of buffers acquired since the first trigger has been detected after a pause.
count_relevant_trgs  = 0 # Number of triggers detected from the first relevant buffer on

trg_close_to_end_prev_bf = False # Flag to signal if the last trigger of the previous buffer was close to the end of the buffer

ch_id = str(255)

max_gray_trgs = 35
max_img_trgs  = 35
single_nat_img_spk_train = np.array([])

# MEA acquisition parameters
buffer_size = 1024
acq_freq    = 20000
trg_threshold = 40000

computation_finished_event = threading.Event() # Event to signal when the computation thread is finished
try:
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
            - peaks: dictionary having as keys the channels and as values the indices of detected peaks 
            -'trg_raw_data': the trigger channel raw data, unfiltered
        _____________________________'''

        packet = json.loads(string_packet, cls=Decoder)

        # print(f"Received packet {packet['buffer_nb']} with {packet['peaks']} peaks")

        '''Check if the packet is relevant, i.e. if it has been sent after an image
            to do this, check the trigger channel (127 on the MEA, so 126 here) it is above a certain threshold ( ~ 5.2*1e5)
            if it is update the peaks count for the current image
        '''

        # Dump the trigger channel data
        # threaded_dump(packet['trg_raw_data'], f'./saved/trg_ch_bf_{packet["buffer_nb"]}_2')

        # if the buffer never crosses the threshold, discard it
        if packet['trg_raw_data'].max() < trg_threshold:
            count_relevant_buffs  = 0
            count_relevant_trgs   = 0

            gray_trigs_count     = 0
            print(f'Packet {packet["buffer_nb"]} received: Not relevant')
            continue
        #endregion
        print(f'Packet {packet["buffer_nb"]} received: relevant')

        '''If packet is relevat, a gray or natural image has been displayed during this buffer.
            count the triggers and check if a natural images has started being displayed'''

        n_trgs_buffer, detected_triggers_idx, trg_close_to_end, trg_close_to_start = count_triggers(packet['trg_raw_data'].astype(np.int32), trigger_diff_threshold=2000)

        if n_trgs_buffer > 0:
            if trg_close_to_end:
                print(f"   This buffer detected a trigger close to the end ")
                logging.info(f"   Trigger close to the end detected in buffer {packet['buffer_nb']}")

            if trg_close_to_start:
                print(f"   This buffer detected a trigger close to the start")
                logging.info(f"   Trigger close to the start detected in buffer {packet['buffer_nb']}")

            if (trg_close_to_end_prev_bf and trg_close_to_start):
                print(f"   This buffer detected a trigger close to the start, and the previous did so close to the end, reducing n_trgs_buffer: {n_trgs_buffer} by 1")
                logging.info(f"   Trigger number reduced by one for buffer {packet['buffer_nb']}")
                n_trgs_buffer -= 1 
                detected_triggers_idx = detected_triggers_idx[1:]

        trg_close_to_end_prev_bf = trg_close_to_end
        print(f"   {n_trgs_buffer} triggers")
               
        count_relevant_trgs  += n_trgs_buffer
        # if image is still in the gray, continue
        if count_relevant_trgs <= max_gray_trgs:
            print(f"   Still in the gray, {count_relevant_trgs} trgs <= {max_gray_trgs}, waiting for more buffers...")
            single_nat_img_spk_train = np.array([])
            continue

        # natural img triggers, surely > 0 and <= 35
        n_trgs_natural_img =  np.min([count_relevant_trgs-max_gray_trgs, max_img_trgs])
        print(f"   Natural image is being displayed since {n_trgs_natural_img} triggers")

        # grey triggers at the end of the buffer
        n_ending_gray_trgs =  np.max([count_relevant_trgs - (max_gray_trgs + n_trgs_natural_img),0])
        print(f"   This buffer had: {n_ending_gray_trgs} ending gray triggers, they will be removed from the indexes")

        # of this buffer triggers we only keep the last n_trgs_natural_img triggers, first excluding the possible ending gray triggers
        idx_natural_img_start = detected_triggers_idx[-(n_trgs_natural_img + n_ending_gray_trgs):None if n_ending_gray_trgs==0 else -n_ending_gray_trgs] 

        # If any triggers at all have been detected
        # Select for this buffer the indices of triggers corresponding to the natural image.
        if len(idx_natural_img_start) != 0:
            ch_bf_peaks_idx  = packet['peaks'][ch_id] 
            nat_img_idx_cond = (ch_bf_peaks_idx > idx_natural_img_start.min()) & (ch_bf_peaks_idx < idx_natural_img_start.max())
            # Take the peaks corresponding to the idxs of the the natural image in this buffer
            natural_peaks_buff = ch_bf_peaks_idx[ nat_img_idx_cond ]  
        # If no triggers have been detected, continue
        else:
            continue
        
        # Peaks idxs corresponding to the natural image for the relevant buffer train
        single_nat_img_spk_train = np.append(single_nat_img_spk_train, natural_peaks_buff) 
            
        # When natural triggers are enough, start the computation thread in parallel, while discarding incoming packets
        if n_trgs_natural_img >= max_img_trgs:

            n_ch_spikes = single_nat_img_spk_train.shape[0]
            print(f'   Acquired {count_relevant_trgs} trgs after the first gray, this means {(buffer_size/acq_freq)*1000*count_relevant_trgs:.2f}ms after the first trigger of the gray')
            print(f'            {n_trgs_natural_img} trgs after first trgs of nat img, this means {(buffer_size/acq_freq)*1000*n_trgs_natural_img:.2f}ms after the first trigger of the natural image')

            # Clearing thread finishing flag and starting thread
            print(f"   Starting the computation thread with {n_ch_spikes} peaks")
            computation_finished_event.clear()  
            computation_thread = threading.Thread(target=fit_and_add_imgID_to_queue, 
                                                    args=(n_ch_spikes, img_ID_queue, computation_finished_event))
            computation_thread.start()

            single_nat_img_spk_train = np.array([])
            count_relevant_trgs = 0

            while not computation_finished_event.is_set():
                string_packet = pull_socket.recv_string()
                packet = json.loads(string_packet, cls=Decoder)
                print("     ...Discarding packet while waiting for computation to finish \n")
            else:
                print("     ...Computation finished, ready to analize next buffer \n")

        else:
            print(f'   Too few natural triggers {n_trgs_natural_img} trgs < {max_img_trgs}, continue...')
            continue


        # new_img_id = img_ID_queue.get() #retrieves next available result and removes it from the queue
        # print(f"Next image to display: {new_img_id}")
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