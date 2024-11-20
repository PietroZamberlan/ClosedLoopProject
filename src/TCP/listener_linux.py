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
from src.TCP.tcp_utils import count_triggers, update_fit, fit_end_queue_img, generate_vec_file, vec_send_and_confirm, dmd_off_send_confirm, time_since_event_set, threaded_dump

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
req_socket_vec = context.socket(zmq.REQ)
req_socket_vec.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:5557")

# Create a REQ socket as a client
req_socket_dmd = context.socket(zmq.REQ)
req_socket_dmd.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:5558")

print("Linux server is running and waiting for data stream...")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thread variables
img_ID_queue = queue.Queue()
fit_finished_event         = threading.Event() # Event to signal when the computation thread is finished
vec_confirmation_event     = threading.Event()
vec_failure_event          = threading.Event()

DMD_stopped_event          = threading.Event()
global_stop_event          = threading.Event()
print_lock                 = threading.Lock()
set_time_lock              = threading.Lock()

# Note that locks and events are mutable objects so when modified in a function they are modified in the global scope
# other objects are immutable so I need to always access them through the dictionary
threadict = {
    "fit_finished_event": fit_finished_event,
    "vec_confirmation_event": vec_confirmation_event,
    "vec_failure_event": vec_failure_event,
    "DMD_stopped_event": DMD_stopped_event,
    "global_stop_event": global_stop_event,
    "print_lock": print_lock,
    "set_time_lock": set_time_lock,
    "dmd_off_set_time": None,
}

#region _______________ Variables _______________

# Configure logging
logging.basicConfig(level=logging.INFO, format='      %(message)s - %(asctime)s-%(levelname)s - ', datefmt='%M:%S')

##### Receive the data stream #####
count_relevant_buffs = 0 # Number of buffers acquired since the first trigger has been detected after a pause.
n_trgs_tot       = 0 # Number of triggers detected from the first relevant buffer on
n_trgs_img_tot   = 0
gray_trigs_count = 0
detected_triggers_idx_tot = np.array([])
detected_triggers_idx     = np.array([])
start_char = ""

trg_close_to_end_prev_bf = False # Flag to signal if the last trigger of the previous buffer was close to the end of the buffer

# DMD triggers parametrs
nb = 10 # Number of buffers to wait before checking if we missed any triggers
max_gray_trgs    = 10
max_img_trgs     = 10
ending_gray_trgs = 20
index_diff_avg   = np.array([])
single_nat_img_spk_train = np.array([])
min_time_dmd_off = 3.5    # Minimum time from the confirmation of being off to be sure that it is
max_time_dmd_off = 7      # Maximum time from the confirmation of being off to be sure that it is not starting again ( The windows server is stopping )


# MEA acquisition parameters
buffer_size     = 1024
acq_freq        = 20000
trigger_freq    = 30
trg_threshold   = 40000
buffer_per_second = acq_freq/buffer_size
ntrgs_per_buffer  = trigger_freq/buffer_per_second
# The expected difference between the indexes of the triggers, in the last nb buffers the average difference should be around this value
exp_trgs_idx_diff = buffer_size/ntrgs_per_buffer

# Acquisition channel ( chosen unit on the MEA )
# ch_id = str(255)
ch_id = str(22)
n_dataset_tot = 80
n_dataset     = 70
random_list_id = np.random.randint(0, n_dataset, 10)
chosen_list_id = np.array([])

# Loop variables
loop_counter       = 0   # counter for the segments of gray+chosen_image+gray+rndm_image+gray
loop_counter_prev  = -1
while_counter      = 0
packet_counter     = 0
poll_interval_main = 100
main_timeout       = 2
print_once         = True # Flag to be used ona loop to print a message only once
start_times = {'global_start': time.time()}
start_times['last_received_packet'] = start_times['global_start']
start_times['while_start']          = start_times['global_start']
start_times['last_rel_packet']      = start_times['global_start']
previous_was_if = False # Flag to signal if the last print was an if statement in the case we are not receiving packets

#endregion

try:
    i=0
    while not global_stop_event.is_set():
        ''' The client is sending packets (dictionaries) with the following structure:
        {'buffer_nb': 10, 'n_peaks': 0,'peaks': {'ch_nb from 0 to 255': np.array(shape=n of peaks in buffer with 'timestamp') } }'}}
            - Unpackable using the custom Decoder class
            - buffer_nb: the number of the buffer
            - n_peaks: the number of peaks in the buffer, already computed by the client
            - peaks: dictionary having as keys the channels and as values the indices of detected peaks 
            -'trg_raw_data': the trigger channel raw data, unfiltered
        _____________________________'''
        # Print the loop counter every time it changes ( every time a new image is displayed )
        if loop_counter > loop_counter_prev:
            print(f"\nPrevious loop took: {time.time()-start_times['while_start']:.3f} seconds from the while start")
            print(f"Previous loop took: {time.time()-start_times['last_rel_packet']:.3f} seconds from the last relevant packet")
            print(f"================================================[ {loop_counter} ]================================================")
            loop_counter_prev = loop_counter

        while_counter += 1
        start_times['while_start'] = time.time()
        poller_main = zmq.Poller()
        poller_main.register(pull_socket, zmq.POLLIN)
        socks_main = dict(poller_main.poll(timeout=poll_interval_main))  # with a poller this while can keep going even if the stream stops
        if pull_socket not in socks_main:
            elapsed_time = time.time() - start_times['last_received_packet']
            print('' if previous_was_if else '\n', end="")            
            print(f"Server has not received packets in the last {(elapsed_time):.3f} seconds...i={i}",end="\r")
            i+=1
            previous_was_if = True
            if elapsed_time > main_timeout and loop_counter > 0:
                global_stop_event.set()
            continue
        else:
            # Receive and decode packet
            string_packet  = pull_socket.recv_string()
            packet = json.loads(string_packet, cls=Decoder)
            if while_counter == 0:
                packet_counter = packet['buffer_nb']
            else:
                packet_counter += 1
            #region _________ Check no packet got lost _________
            if packet['buffer_nb'] != packet_counter:
                # If a packet got lost, restart the loop and the DMD on the other side, then update the packet counter
                # The only way I have for now to restart the DMD is to wait for its timeouts to go off
                packet_counter = packet['buffer_nb']
                continue
            #endregion _________________________________________
            start_times['last_received_packet'] = time.time()
            print('\n' if previous_was_if else '', end="")
            i=0 
            previous_was_if = False



        #region To use the saved packets only if the mea is not avaialble, or dump data
        # with open(f'packet{i}.json', 'r') as f:
        #     string_packet = f.read()
        # Dump the trigger channel datarepo_dirrepo_dir
        threaded_dump(packet['trg_raw_data'], f'{repo_dir}/src/TCP/saved/alpha_tests/trg_ch_bf_{packet["buffer_nb"]}_1')
        #endregion
            
        #region _________ Check if packet is relevant
        # To do this, check the trigger channel (127 on the MEA, so 126 here) it is above a certain threshold ( ~ 5.2*1e5)

        # If the buffer never crosses the threshold, discard it
        if packet['trg_raw_data'].max() < trg_threshold:
            count_relevant_buffs = 0
            n_trgs_tot           = 0
            gray_trigs_count     = 0
            detected_triggers_idx_tot = np.array([])
            detected_triggers_idx     = np.array([])
            print(f'Packet {packet["buffer_nb"]:>5} received: Not relevant',) #end="\r")
            start_char = "\n"
            continue

        print(f'{start_char}Packet {packet["buffer_nb"]:>5} received: Relevant', end="")
        start_times['last_rel_packet'] = time.time()
        count_relevant_buffs += 1
        start_char = ""
        #endregion

        #region _________ Count the triggers and check if a natural images has started being displayed , run minimal sanity check on the timing of the detected triggers ______
        n_trgs_buffer, detected_triggers_idx, trg_close_to_end, trg_close_to_start = count_triggers(packet['trg_raw_data'].astype(np.int32), trigger_diff_threshold=2000)

        if n_trgs_buffer == 0:
            print(f"\nNo triggers detected in buffer {packet['buffer_nb']}, continue...\n")
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
        #endregion

        #region _________ Edge cases: triggers close to the start or end of the buffer _________    
        if n_trgs_buffer > 0:
            if trg_close_to_end:
                logging.info(f"Trigger close to the end detected in buffer {packet['buffer_nb']}")

            if trg_close_to_start:
                logging.info(f"Trigger close to the start detected in buffer {packet['buffer_nb']}")

            if (trg_close_to_end_prev_bf and trg_close_to_start):
                logging.info(f"Buffer {packet['buffer_nb']} detected a trigger close to the start, and the previous did so close to the end, reducing n_trgs_buffer: {n_trgs_buffer} by 1")
                logging.info(f"Trigger number reduced by one for buffer {packet['buffer_nb']}")
                n_trgs_buffer -= 1 
                detected_triggers_idx = detected_triggers_idx[1:]
        #endregion
        
        trg_close_to_end_prev_bf = trg_close_to_end
        print(f" triggers :{n_trgs_buffer:>3},", end='' )
               
        n_trgs_tot  += n_trgs_buffer
        print(f" TOT triggers detected: {n_trgs_tot:>3}.", end='')

        # if image is still in the gray, continue
        if n_trgs_tot <= max_gray_trgs:
            print(f" Gray   : {n_trgs_tot:>2} trgs <= {max_gray_trgs:>2}, waiting...")
            single_nat_img_spk_train = np.array([])
            continue

        # else: first gray has finished, start counting the natural image triggers

        #region ________ Possible initial and ending gray triggers removal________
        # n of natutal img in this buffer is the number of total triggers minus the _amount of triggers might have been missing to reach the max_gray_trgs, in the buffer_
        # this quantity is positive if this trigger was the one getting over the max_gray_trgs
        # otherwise it is negative

        # number of triggers of the current buffer that have been used to reach the max_gray_trgs
        n_trigs_tot_prev      = n_trgs_tot - n_trgs_buffer        # previous count of total triggers
        n_trgs_spent_for_gray = max_gray_trgs - n_trigs_tot_prev  # triggers of this buffer that have been used to reach the max_gray_trgs
        # If none of the current buffer triggers where part of the gray, n_starting_gray_trgs = 0
        if n_trgs_spent_for_gray <= 0: 
            n_starting_gray_trgs = 0
        else:
            n_starting_gray_trgs = n_trgs_spent_for_gray          # triggers of this buffer that have been used to reach the max_gray_trgs

        # Remove possible starting gray triggers from counters and indices array
        n_trgs_img      = n_trgs_buffer - n_starting_gray_trgs    # n of natural img triggers in this buffer
        n_trgs_img_tot += n_trgs_img
        idx_natural_img_start = detected_triggers_idx[-n_trgs_img:] 

        # now do the same for the ending gray triggers. This buffer might be at the end of the natural image, and already have some gray triggrs
        n_trgs_already_gray =  n_trgs_img_tot - max_img_trgs
        if n_trgs_already_gray > 0:
            n_ending_gray_trgs = n_trgs_already_gray
        else:
            n_ending_gray_trgs = 0

        # Remove possible ending gray triggers
        n_trgs_img     -= n_ending_gray_trgs                       # n of natural img triggers in this buffer
        n_trgs_img_tot -= n_ending_gray_trgs                       # n of natural img triggers until now
        idx_natural_img_start = idx_natural_img_start[:None if n_ending_gray_trgs==0 else -n_ending_gray_trgs]         
        # warn 
        if n_ending_gray_trgs > 0 or n_starting_gray_trgs > 0:
            print(f" Ending gray: {n_ending_gray_trgs}, Starting gray: {n_starting_gray_trgs}, Removed", end='')

        # Check if any triggers are left after removing the gray triggers ( they should ), unless some very weird parameters have been used.
        # Select for this buffer the indices of triggers corresponding to the natural image.
        if len(idx_natural_img_start) != 0:
            ch_bf_peaks_idx  = packet['peaks'][ch_id]             # get the detected spikes in the channel/unit we care about
            nat_img_idx_condition = (ch_bf_peaks_idx >= idx_natural_img_start.min()) & (ch_bf_peaks_idx <= idx_natural_img_start.max())
            # Take the peaks corresponding to the idxs of the the natural image in this buffer
            natural_peaks_buff = ch_bf_peaks_idx[ nat_img_idx_condition ]  
        else:
            print(f"   All triggers in this buffer where gray... ? continue...")
            continue
        #endregion ________________________________________________________________

        # Peaks idxs corresponding to the natural image for the relevant buffer train
        single_nat_img_spk_train = np.append(single_nat_img_spk_train, natural_peaks_buff) 

        if n_trgs_img_tot < max_img_trgs:
            print(f" Natural: {n_trgs_img_tot:>2} trgs <= {max_img_trgs}, waiting...",)
            continue
        if n_trgs_img_tot > max_img_trgs:
            print(f"   Best image is being computed...",)
            continue
        # Natural triggers are enough
        if n_trgs_img_tot == max_img_trgs:

            n_ch_spikes = single_nat_img_spk_train.shape[0]

            #region ________ Send the threaded DMD off command and wait for confirmation ________
            DMD_stopped_event.clear()
            args = (req_socket_dmd, threadict)
            communication_thread = threading.Thread(target=dmd_off_send_confirm, args=args) 
            communication_thread.start()
            #endregion _______________________________________________________________

            #region ________ Fit the GP and add the new image ID to the queue ___________________
            print(f"\nStarting the computation thread with {n_ch_spikes} peaks")

            fit_finished_event.clear()  
            computation_thread = threading.Thread(target=fit_end_queue_img, 
                                                    args=(n_ch_spikes, img_ID_queue, threadict))
            computation_thread.start()
            single_nat_img_spk_train = np.array([])
            n_trgs_tot     = 0
            n_trgs_img_tot = 0

            poller_fit = zmq.Poller()
            poller_fit.register(pull_socket, zmq.POLLIN)
            p=0
            while not fit_finished_event.is_set() and not global_stop_event.is_set():
                socks = dict(poller_fit.poll(timeout=poll_interval_main))  # with a poller this while can keep going even if the stream stops
                if pull_socket in socks:
                    string_packet = pull_socket.recv_string()
                    packet = json.loads(string_packet, cls=Decoder)
                    with print_lock:
                        print(f"\rDiscarding up to packet {packet['buffer_nb']} while waiting for computation to finish...", end="")
                else: 
                    pass
            if global_stop_event.is_set():
                    continue
            computation_thread.join()
            with print_lock:
                print(f"\nGP fit completed, image ID chosen...")
            #endregion ________________________________________________________________

            chosen_img_id = img_ID_queue.get() #retrieves next available result and removes it from the queue
            rndm_img_id   = random_list_id[loop_counter]

            #region ________ Send the VEC file and wait for confirmation ________
            # New ID has been chosen, send it and receive confirmation, through a dedicated thread that will set the event when confirmation 
            # for a written VEC file is recived from the client
            vec_confirmation_event.clear()
            args = (chosen_img_id, rndm_img_id, threadict, req_socket_vec)
            kwargs = {'max_gray_trgs': max_gray_trgs, 'max_img_trgs': max_img_trgs, 'ending_gray_trgs': ending_gray_trgs}
            communication_thread = threading.Thread(target=vec_send_and_confirm, args=args, kwargs=kwargs) 
            communication_thread.start()

            poller = zmq.Poller()
            poller.register(pull_socket, zmq.POLLIN) 
            while not vec_confirmation_event.is_set() and not vec_failure_event.is_set():
                socks = dict(poller.poll(timeout=5))
                start_time = time.time()
                if pull_socket in socks:
                    string_packet = pull_socket.recv_string()
                    packet = json.loads(string_packet, cls=Decoder)
                    with print_lock:
                        print(f"\rDiscarding up to packet {packet['buffer_nb']} while waiting for client to confirm new VEC...", end="")
                else:
                    # logging.info('Client has not sent packets in the last 5 milliseconds...')#, end="\r")
                    pass # we dont break here, we let the thread break
                    
            if vec_failure_event.is_set():
                print(f"\nClient did not confirm reception of VEC, socket will not be able to send next one, closing")
                global_stop_event.set()
                continue

            if not DMD_stopped_event.is_set():
                print(f"\nClient did not yet confirm DMD off")

            if vec_confirmation_event.is_set():
                print(f"\nClient confirmed reception of VEC")

            dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
            while dmd_off_time < min_time_dmd_off:
                dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
                print(f"\rWaiting since {(dmd_off_time):.2f} for DMD to really turn off...", end="")
                print_once = False
                pass
            print_once = True

            dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
            if dmd_off_time > max_time_dmd_off:
                print(f"Too long, dmd off time: {(dmd_off_time):.2f} > {max_time_dmd_off} seconds, server is shutting down...")
                dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
                global_stop_event.set()
                continue
            #endregion _______________________________________________________________

        if loop_counter == n_dataset:
            print("All images displayed, server is shutting down...")
            global_stop_event.set()
            break
        else:
            loop_counter+=1
            continue
        #endregion ________________________________________________________________

        # We should get here only if the computation thread has finished and we are ready to send the new image ID to the client
        # or if something has failed

        # Get the result of the computation thread and send it to the client
        # print(f"Next image to display: {chosen_img_id}")
        # computation_thread.join()
        # continue

    else:
        print("\nStop event set, server is shutting down...")
        pass
except KeyboardInterrupt:
    print("\n\nKeyInterrupted: Server is shutting down...")
    global_stop_event.set() 
    print("Stop event set...")    

finally:
    global_stop_event.set()     
    print('\nClosing pull socket...')
    pull_socket.setsockopt(zmq.LINGER, 0)
    pull_socket.close()
    print('Closing vec req socket...')
    req_socket_vec.setsockopt(zmq.LINGER, 0)
    req_socket_vec.close()
    print('Closing dmd req socket...')
    req_socket_dmd.setsockopt(zmq.LINGER, 0)
    req_socket_dmd.close()

    
        
    print('Closing context...')
    context.term()
    # print('Joining threads...')
    # computation_thread.join()