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
# pub_socket = context.socket(zmq.PUSH)
# pub_socket.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:5556")

# Create the listening socket as a server
# pull_socket = context.socket(zmq.STREAM)
pull_socket = context.socket(zmq.PULL)
pull_socket.bind("tcp://*:5555")

# Create a REQ socket as a client
req_socket = context.socket(zmq.REQ)
req_socket.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:5557")

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
    time.sleep(8)
    logging.info(f"   Thread : imdID added to the queue: {imgID}")
    finished_event.set()

def generate_vec_file(img_ID, max_gray_trgs=10, max_img_trgs=10, ending_gray_trgs=10):
    """
    Parameters:
    img_ID (int): The image ID.
    max_gray_trgs (int): The number of lines representing the STARTING gray image.
    ending_gray_trgs (int): The number of lines representing the ENDING gray image.
    max_img_trgs (int): The number of lines representing triggers of the natural image.
    """

    file_path = f'{repo_dir}/src/DMD/saved/vec_file_{img_ID}.txt'
    lines = []

    lines.append(f"0 {max_gray_trgs+max_img_trgs+ending_gray_trgs} 0 0 0\n")
        # Write the following lines
    for _ in range(max_gray_trgs):
        lines.append("0 0 0 0 0\n")
    for _ in range(max_gray_trgs):
        lines.append(f"0 {img_ID} 0 0 0\n")            
    for _ in range(ending_gray_trgs):
        lines.append("0 0 0 0 0\n")  
    
    file_content = ''.join(lines)
    with open(file_path, 'w') as file:
        # Write the first line
        file.write(file_content)
              
    return file_content, file_path

def send_and_confirm( img_ID, finished_event, max_gray_trgs=10, max_img_trgs=10, ending_gray_trgs=10):
    '''
    Send to the client the VEC file corresponding the the chosen image ID and wait for confirmation. When received, set the event and allow the main thread to 
    Stop discarding packets
    '''
    print(f"   Sending new VEC file corresponding to image ID: {img_ID}")

    vec_content, vec_path = generate_vec_file(img_ID=img_ID, max_gray_trgs=max_gray_trgs, max_img_trgs=max_img_trgs, ending_gray_trgs=ending_gray_trgs )
    # pub_socket.send_string(vec_content)
    # 
    # Send the VEC file to the client and wait for confirmation
    req_socket.send_string(vec_content)
    print(f"   VEC file sent to the client")
    confirmation = req_socket.recv_string()
    if confirmation == 'CONFIRMED':
        new_img_id_sent_event.set()
        logging.info(f"\n   Image ID sent and confirmed by the client")
    else:
        logging.error(f"\n   Error: The client did not confirm the reception of the new image ID")



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


# Thread variables
img_ID_queue = queue.Queue()
computation_finished_event = threading.Event() # Event to signal when the computation thread is finished
new_img_id_sent_event = threading.Event()


# Configure logging
logging.basicConfig(level=logging.INFO, format='      %(message)s - %(asctime)s - %(levelname)s - ', datefmt='%M:%S')

##### Receive the data stream #####

count_relevant_buffs = 0 # Number of buffers acquired since the first trigger has been detected after a pause.
n_trgs_tot     = 0 # Number of triggers detected from the first relevant buffer on
n_trgs_img_tot = 0

trg_close_to_end_prev_bf = False # Flag to signal if the last trigger of the previous buffer was close to the end of the buffer

ch_id = str(255)

max_gray_trgs = 10
max_img_trgs  = 10
ending_gray_trgs = 20
single_nat_img_spk_train = np.array([])
index_diff_avg = np.array([])


# MEA acquisition parameters
buffer_size     = 1024
acq_freq        = 20000
trigger_freq    = 30
trg_threshold   = 40000
buffer_per_second = acq_freq/buffer_size
ntrgs_per_buffer  = trigger_freq/buffer_per_second
# The expected difference between the indexes of the triggers, in the last nb buffers the average difference should be around this value
exp_trgs_idx_diff = buffer_size/ntrgs_per_buffer

# Flag to signal that we need to send back the new image ID to the client
computing_and_sending = False
# Number of buffers to wait before checking if we missed any triggers
nb = 10

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
        # threaded_dump(packet['trg_raw_data'], f'{repo_dir}/src/TCP/saved/trg_ch_bf_{packet["buffer_nb"]}_issueDMD1')

        # if the buffer never crosses the threshold, discard it
        if packet['trg_raw_data'].max() < trg_threshold:
            count_relevant_buffs = 0
            n_trgs_tot           = 0
            gray_trigs_count     = 0
            detected_triggers_idx_tot = np.array([])
            detected_triggers_idx     = np.array([])
            print(f'Packet {packet["buffer_nb"]} received: Not relevant', end="\r")
            # print(" " * 80, end="\r")  # Clear the line
            continue
        #endregion
        print(f'\nPacket {packet["buffer_nb"]} received: relevant')
        count_relevant_buffs += 1

        '''If packet is relevat, a gray or natural image has been displayed during this buffer.
            count the triggers and check if a natural images has started being displayed'''

        n_trgs_buffer, detected_triggers_idx, trg_close_to_end, trg_close_to_start = count_triggers(packet['trg_raw_data'].astype(np.int32), trigger_diff_threshold=2000)

        if n_trgs_buffer == 0:
            print(f"\n   No triggers detected in buffer {packet['buffer_nb']}, continue...\r", end="")
            continue

        detected_triggers_idx_tot = np.append(detected_triggers_idx_tot, detected_triggers_idx + count_relevant_buffs*buffer_size)
        # Every nb buffers with triggers, check that we missed none
        if count_relevant_buffs % nb == 0 and count_relevant_buffs > 0:

            last_idxs = detected_triggers_idx_tot[-10:]

            index_diff_avg_prev = np.mean(last_idxs[1:] - last_idxs[:-1]) if last_idxs.shape[0] > 1 else 0

            if np.abs(index_diff_avg_prev-exp_trgs_idx_diff) > np.abs(exp_trgs_idx_diff*0.01):
                print(f"\n   Warning: The average difference between the indexes of the last {last_idxs.shape[0]}: {index_diff_avg_prev} triggers is different from the expected: {exp_trgs_idx_diff}" 
                      f"\n   by more than 1%")
                logging.warning(f"   Warning: The average difference between the indexes of the triggers has changed by more than 1% from the previous {nb} buffers,"
                                f"   a trigger might have been lost")

        #region _________ Edge cases: triggers close to the start or end of the buffer _________    
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
        #endregion
        
        trg_close_to_end_prev_bf = trg_close_to_end
        print(f"   {n_trgs_buffer} triggers")
               
        n_trgs_tot  += n_trgs_buffer
        print(f"   Total triggers detected: {n_trgs_tot}")

        # if image is still in the gray, continue
        if n_trgs_tot <= max_gray_trgs:
            print(f"   Still in the gray, {n_trgs_tot} trgs <= {max_gray_trgs}, waiting for more buffers...")
            single_nat_img_spk_train = np.array([])
            continue

        # else: first gray has finished, start counting the natural image triggers

        #region ________Possible initial and ending gray triggers removal________
        # n of natutal img in this buffer is the number of triggers minus the _amount of triggers might have been missing to reach the max_gray_trgs_
        # this quantity is positive if this trigger was the one getting over the max_gray_trgs
        # otherwise it is negative

        # number of triggers of the current buffer that have been used to reach the max_gray_trgs
        n_trigs_tot_prev      = n_trgs_tot - n_trgs_buffer
        n_trgs_spent_for_gray = max_gray_trgs - n_trigs_tot_prev
        # If none of the current buffer triggers where part of the gray, n_starting_gray_trgs = 0
        if n_trgs_spent_for_gray <= 0: 
            n_starting_gray_trgs = 0
        else:
            n_starting_gray_trgs = n_trgs_spent_for_gray

        # Remove possible starting gray triggers from counters and indices array
        n_trgs_img      = n_trgs_buffer - n_starting_gray_trgs
        n_trgs_img_tot += n_trgs_img
        idx_natural_img_start = detected_triggers_idx[-n_trgs_img:] 

        n_trgs_already_gray =  n_trgs_img_tot - max_img_trgs
        if n_trgs_already_gray > 0:
            n_ending_gray_trgs = n_trgs_already_gray
        else:
            n_ending_gray_trgs = 0

        # Remove possible ending gray triggers
        n_trgs_img     -= n_ending_gray_trgs
        n_trgs_img_tot -= n_ending_gray_trgs
        
        idx_natural_img_start = idx_natural_img_start[:None if n_ending_gray_trgs==0 else -n_ending_gray_trgs]         
        #endregion

        if n_ending_gray_trgs > 0:
            print(f"   This buffer had: {n_ending_gray_trgs} ending gray triggers, they have been removed from the indexes")
            n_ending_gray_trgs = 0
        # Check if any triggers are left after removing the gray triggers ( they should )
        # Select for this buffer the indices of triggers corresponding to the natural image.
        if len(idx_natural_img_start) != 0:
            ch_bf_peaks_idx  = packet['peaks'][ch_id] 
            nat_img_idx_condition = (ch_bf_peaks_idx >= idx_natural_img_start.min()) & (ch_bf_peaks_idx <= idx_natural_img_start.max())
            # Take the peaks corresponding to the idxs of the the natural image in this buffer
            natural_peaks_buff = ch_bf_peaks_idx[ nat_img_idx_condition ]  
        else:
            print(f"   All triggers in this buffer where gray... ? continue...")
            continue

        # Peaks idxs corresponding to the natural image for the relevant buffer train
        single_nat_img_spk_train = np.append(single_nat_img_spk_train, natural_peaks_buff) 

        if n_trgs_img_tot < max_img_trgs:
            print(f"   Too few natural triggers, {n_trgs_img_tot} trgs <= {max_img_trgs}, waiting for more buffers...\r", end="")
            continue
        if n_trgs_img_tot > max_img_trgs:
            print(f"   Best image is being computed...\r", end="")
            continue
        # Natural triggers are enough. If they are too many the flag will skip the computation
        else:
            computing_img_ID = True
            n_ch_spikes = single_nat_img_spk_train.shape[0]
            print(f'   Acquired {n_trgs_tot} trgs after the first gray, this means {(buffer_size/acq_freq)*1000*n_trgs_tot:.2f}ms after the first trigger of the gray')
            print(f'            {n_trgs_img_tot} trgs after first trgs of nat img, this means {(buffer_size/acq_freq)*1000*n_trgs_img:.2f}ms after the first trigger of the natural image')

            # Clearing thread finishing flag and starting thread
            print(f"   Starting the computation thread with {n_ch_spikes} peaks")
            computation_finished_event.clear()  
            computation_thread = threading.Thread(target=fit_and_add_imgID_to_queue, 
                                                    args=(n_ch_spikes, img_ID_queue, computation_finished_event))
            computation_thread.start()
            single_nat_img_spk_train = np.array([])
            n_trgs_tot     = 0
            n_trgs_img_tot = 0

            while not computation_finished_event.is_set():
                string_packet = pull_socket.recv_string()
                packet = json.loads(string_packet, cls=Decoder)
                print(f"     ...Discarding packet {packet['buffer_nb']} while waiting for computation to finish", end="\r")
      
            # New ID has been chose, send it and receive confirmation, through a dedicated thread that will set the event when confirmation 
            # for a written VEC file is recived from the client
            new_img_id = img_ID_queue.get() #retrieves next available result and removes it from the queue

            new_img_id_sent_event.clear()
            args = (new_img_id, new_img_id_sent_event)
            kwargs = {'max_gray_trgs': max_gray_trgs, 'max_img_trgs': max_img_trgs, 'ending_gray_trgs': ending_gray_trgs}
            communication_thread = threading.Thread(target=send_and_confirm, args=args, kwargs=kwargs) 
            communication_thread.start() 

            while not new_img_id_sent_event.is_set():
                string_packet = pull_socket.recv_string()
                packet = json.loads(string_packet, cls=Decoder)
                print(f"\n     ...Discarding packet {packet['buffer_nb']} while waiting for client to receive new ID", end="\r")

            print("\n     ...Computation finished, ready to send output to client \n")
            #     send_ID_flag = True


        # We should get here only if the computation thread has finished and we are ready to send the new image ID to the client
        # or if something has failed

        # Get the result of the computation thread and send it to the client
        # print(f"Next image to display: {new_img_id}")
        # computation_thread.join()
        # continue
        #endregion

except KeyboardInterrupt:
    print("\n\nServer is shutting down...")

finally:
    print('Closing pull socket...')
    pull_socket.setsockopt(zmq.LINGER, 0)
    pull_socket.close()
    print('Closing req socket...')
    req_socket.close()



    # print('Closing push socket...')
    # pub_socket.setsockopt(zmq.LINGER, 0)
    # pub_socket.close()
    print('Closing context...')
    context.term()
    print('Joining threads...')
    # computation_thread.join()