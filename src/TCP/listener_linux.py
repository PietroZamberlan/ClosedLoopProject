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
import importlib.util

import matplotlib.pyplot as plt
import matplotlib.animation as animation


current_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir    = os.path.join(current_dir, '../../')
sys.path.insert(0, os.path.abspath(repo_dir))

# Import the configuration file
from config.config import WINDOWS_OLD_MACHINE_IP, PULL_SOCKET_PORT, REQ_SOCKET_VEC_PORT, REQ_SOCKET_DMD_PORT

# Load the Gaussian Processes module ( it has a - in the name so we need importlib)
# from Gaussian-Processes.Spatial_GP_repo import utils

utils_spec = importlib.util.spec_from_file_location(
    "utils",
    os.path.join(repo_dir, "Gaussian-Processes/Spatial_GP_repo/utils.py")
)
utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils)

from src.TCP.tcp_utils import count_triggers, update_fit, threaded_fit_end_queue_img, generate_vec_file, threaded_vec_send_and_confirm, threaded_sender_dmd_off_signal, time_since_event_set, threaded_dump

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

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def launch_dmd_off_sender():
    '''
    Launches the thread that sends the DMD off command to Windows and waits for the confirmation
    
    The thread will set the DMD_stopped_event to True when the confirmation is received

    Sets:
        - DMD_stopped_event: to False, to allow the thread to set it True
    '''
    DMD_stopped_event.clear()
    args = (req_socket_dmd, threadict)
    DMD_off_sender_thread = threading.Thread(target=threaded_sender_dmd_off_signal, args=args) 
    DMD_off_sender_thread.start()
    return DMD_off_sender_thread

def launch_computation_thread():
    '''
    Launches the thread that fits the GP and adds the new image ID to the queue
    '''

    print(f"\nStarting the computation thread with {n_ch_spikes} peaks")
    fit_finished_event.clear()  
    computation_thread = threading.Thread(target=threaded_fit_end_queue_img, 
                                            args=(n_ch_spikes, img_ID_queue, threadict))
    computation_thread.start()
    return computation_thread

def wait_for_computation_and_discard():
    '''
    Waits for the computation to finish or for the global stop event to be set
    while discarding all incoming packages from the pull socket:

    '''
    poller_fit = zmq.Poller()
    poller_fit.register(pull_socket_packets, zmq.POLLIN)
    # p=0
    while not fit_finished_event.is_set() and not global_stop_event.is_set():
        socks = dict(poller_fit.poll(timeout=poll_interval_main))  # with a poller this while can keep going even if the stream stops
        if pull_socket_packets in socks:
            string_packet = pull_socket_packets.recv_string()
            packet = json.loads(string_packet, cls=Decoder)
            with print_lock:
                print(f"\rDiscarding up to packet {packet['buffer_nb']} while waiting for computation to finish...", end="")
        else: 
            pass
    if fit_finished_event.is_set():
        return
    elif global_stop_event.is_set():
        raise CustomException("Global stop event set")
    else:
        raise CustomException("Unknown error - fit_finished_event not set and global_stop_event not set")

def launch_vec_sender():
    '''
    Launches the thread that generates and sends the VEC file to Windows, and waits for the confirmation.

    Sets:
        vec_confirmation_event: to False, to allow the thread to set it True            
    '''
    
    vec_confirmation_event.clear()
    args = (chosen_img_id, rndm_img_id, threadict, req_socket_vec)
    kwargs = {'max_gray_trgs': max_gray_trgs, 'max_img_trgs': max_img_trgs, 'ending_gray_trgs': ending_gray_trgs}
    vec_sender_thread = threading.Thread(target=threaded_vec_send_and_confirm, args=args, kwargs=kwargs) 
    vec_sender_thread.start()
    return vec_sender_thread

def wait_for_vec_confirmation():
    '''
    Waits for the VEC reception confirmation from Windows or for the global stop event to be set

    '''
    poller = zmq.Poller()
    poller.register(pull_socket_packets, zmq.POLLIN) 
    while not vec_confirmation_event.is_set() and not vec_failure_event.is_set() and not global_stop_event.is_set():
        socks = dict(poller.poll(timeout=5))
        start_time = time.time()
        if pull_socket_packets in socks:
            string_packet = pull_socket_packets.recv_string()
            packet = json.loads(string_packet, cls=Decoder)
            with print_lock:
                print(f"\rDiscarding up to packet {packet['buffer_nb']} while waiting for client to confirm new VEC...", end="")
        else:
            # logging.info('Client has not sent packets in the last 5 milliseconds...')#, end="\r")
            pass # we dont break here, we let the thread break
    if vec_confirmation_event.is_set():
        return
    elif global_stop_event.is_set():
        raise CustomException("Global stop event set")
    else:
        raise CustomException("vec_failure_event error - vec_confirmation_event not set and global_stop_event not set")            

def update_plot(packet, ch_id, ax, fig, lines, count_relevant_buffs, plot_counter_for_image, max_lines=10):
    """
    Update the plot with the received packet data.
    The plot window (default width 100) is centered around the maximum value in the data.
    
    Args:
        packet (dict): Dictionary containing the received data in packet['data'].
        ch_id (int): Channel id used to index into the data.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        fig (matplotlib.figure.Figure): The figure object containing the axes.
        lines (list): List of existing Line2D objects to be updated/faded.
        count_relevant_buffs (int): Counter for relevant buffers.
        plot_counter_for_image (int): Counter for the number of plots rendered.
        max_lines (int): Maximum number of lines to keep on the plot (default: 10).
        fade_step (float): Decrease in alpha for each iteration (default: 0.15). # removed
    
    Returns:
        int: The updated plot_counter_for_image.
    """
    # Extract the desired segment from the received packet.
    segment = packet['data'][:, ch_id]
    
    # Compute the window parameters to center around the maximum spike.
    window_size = 100
    half_window = window_size // 2
    center_idx = np.argmax(segment)
    
    # Ensure the window is properly adjusted if the spike is near the edges.
    start_segment_idx = center_idx - half_window
    if start_segment_idx < 0:
        start_segment_idx = 0
    end_segment_idx = start_segment_idx + window_size
    if end_segment_idx > segment.shape[0]:
        end_segment_idx = segment.shape[0]
        start_segment_idx = max(end_segment_idx - window_size, 0)
    
    # Slice the segment for plotting.
    reduced_segment = segment[start_segment_idx:end_segment_idx]
    
    # Generate x-values corresponding to the segment indices.
    # x_values = np.arange(start_segment_idx, end_segment_idx)
    x_values = np.arange(0, window_size)
    # Plot the new line with full opacity.
    new_line, = ax.plot(x_values, reduced_segment, alpha=1.0, color='k')
    lines.append(new_line)
    
    # Fade out older lines (exclude the newly added one).
    fade_step = min(1/len(lines), 1/max_lines)
    for line in lines[:-1]:
        current_alpha = line.get_alpha()
        new_alpha = current_alpha - fade_step
        if new_alpha > 0:
            line.set_alpha(new_alpha)
        else:
            line.remove()
            lines.remove(line)
    
    # Remove the oldest lines if there are too many.
    while len(lines) > max_lines:
        old_line = lines.pop(0)
        old_line.remove()
    
    # Update the axis and title.
    ax.set_xlim(0, window_size)
    ax.set_ylim(34500, 35500)
    # ax.relim()
    # ax.autoscale_view()
    ax.set_title(f"Relevant buffer number {count_relevant_buffs} - {plot_counter_for_image}th plot. ")
    
    # Update the axis and title.
    ax.relim()
    ax.autoscale_view()
    ax.set_title(f"Relevant buffer number {count_relevant_buffs} - {plot_counter_for_image}th plot")
    
    # Increment the plot counter.
    # plot_counter_for_image += 1
    
    # Save the figure to the same PDF file.
    # fig.savefig("plot.pdf")

    # Increment the plot counter.
    plot_counter_for_image += 1
    
    # Redraw the figure.
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()
    
    return plot_counter_for_image


context = zmq.Context()
# Create the publishing socket
# pub_socket = context.socket(zmq.PUSH)
# pub_socket.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:5556")

# Create the listening socket as a server
# pull_socket_packets = context.socket(zmq.STREAM)
pull_socket_packets = context.socket(zmq.PULL)
pull_socket_packets.bind(f"tcp://*:{PULL_SOCKET_PORT}")

# Create a REQ socket as a client
req_socket_vec = context.socket(zmq.REQ)
req_socket_vec.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:{REQ_SOCKET_VEC_PORT}")

# Create a REQ socket as a client
req_socket_dmd = context.socket(zmq.REQ)
req_socket_dmd.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:{REQ_SOCKET_DMD_PORT}")

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
# other objects are immutable so I need to always access them through the dictionary to modify them in a function and have the changes reflected in the global scope
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

plot_data_queue = queue.Queue()

# Simulate incoming packet
# packet = {}
# ch_id = 53
# buffer_nb = 43
# file_path = os.path.join(current_dir, f'saved/ch_{ch_id}/')

# data_path              = os.path.join(file_path, f'data_ch_{ch_id}_bf_{buffer_nb}.npy')
# packet['data']         = np.load(data_path).astype(np.int32)
# packet['data']         = np.tile(packet['data'], (256, 1))
# trgs_path              = os.path.join(file_path, f'trg_ch_bf_{buffer_nb}.npy')
# packet['trg_raw_data'] = np.load(trgs_path).astype(np.int32)
# packet['buffer_nb']    = buffer_nb

# Variables for plotting
fig, ax = plt.subplots()
line, = ax.plot([], [])
plt.ion()

color_index = 0

# plot_counter_for_image = 0 # For counting how many times we plotted the buffers in a relevant buffer streak

lines = []

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

trg_close_to_end_flag_prev_bf = False # Flag to signal if the last trigger of the previous buffer was close to the end of the buffer

# DMD triggers parametrs
nb_bff_wait = 10 # Number of buffers to wait before checking if we missed any triggers
max_gray_trgs    = 10
max_img_trgs     = 160
ending_gray_trgs = 20
index_diff_avg   = np.array([])
single_nat_img_spk_train = np.array([])
min_time_dmd_off = 3.5    # Min time to wait from the DMD off confirmation to be sure its really off
max_time_dmd_off = 7      # Maximum time from the confirmation of being off to be sure that it is not starting again ( The windows server is stopping )


# MEA acquisition parameters
buffer_size     = 1024
acq_freq        = 20000
trigger_freq    = 30
trg_threshold   = 40000
buffer_per_second = acq_freq/buffer_size
ntrgs_per_buffer  = trigger_freq/buffer_per_second
# The expected difference between the indexes of the triggers, in the last nb_bff_wait buffers the average difference should be around this value
exp_trgs_idx_diff = buffer_size/ntrgs_per_buffer

# Acquisition channel ( chosen unit on the MEA )
# ch_id = str(255)
# ch_id = str(22)
# ch_id = str(52)
# ch_id = 52
# ch_id = 126
ch_id = 53
n_dataset_tot = 80
n_dataset     = 70
random_list_id = np.random.randint(0, n_dataset, 10)
chosen_list_id = np.array([])

# Loop variables
pair_img_counter   = 0      # Counter for the segments of gray+chosen_image+gray+rndm_image+gray
loop_counter_prev  = -1
poll_interval_main = 100    # Every 
main_counter       = -1     # Number of loops listening from the server. Its not the bf_number because some of them are just discarded
main_timeout       = 2      # Max time Linux client waits for a packet before shutting down
start_times = {'global_start': time.time()}
start_times['last_received_packet'] = start_times['global_start']
start_times['while_start']          = start_times['global_start']
start_times['last_rel_packet']      = start_times['global_start']
previous_was_if = False # Flag to signal if the last print was an if statement in the case we are not receiving packets

#endregion

try:
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
        main_counter += 1 

        if pair_img_counter > loop_counter_prev:
            print(f"\nPrevious loop took: {time.time()-start_times['while_start']:.3f} seconds from the while start")
            print(f"Previous loop took: {time.time()-start_times['last_rel_packet']:.3f} seconds from the last relevant packet")
            print(f"================================================[ {pair_img_counter} Img ]================================================")
            loop_counter_prev = pair_img_counter

        start_times['while_start'] = time.time()

        # region _________ Receive and decode packet ________

        
        # for _ in range(30):

        # stime = time.time()

        # string_packet = pull_socket_packets.recv_string()
        # packet        = json.loads(string_packet, cls=Decoder)

        # received_time = time.time()
        # send_time_server = packet['send_time']


        # elapsed_time = time.time() - stime
        # print(f"\nPacket received in: {elapsed_time:.3f} seconds", end="\n")

        # continue

        poller_main = zmq.Poller()
        poller_main.register(pull_socket_packets, zmq.POLLIN)
        socks_main = dict(poller_main.poll(timeout=poll_interval_main))  # with a poller this while can keep going even if the stream stops
        if pull_socket_packets not in socks_main:
            elapsed_time = time.time() - start_times['last_received_packet']
            print('' if previous_was_if else '\n', end="")            
            print(f"Server has not received packets in the last {(elapsed_time):.3f} seconds...",end="\r")
            previous_was_if = True
            if elapsed_time > main_timeout and pair_img_counter > 0:
                global_stop_event.set()
            continue
        else:
            # Receive and decode packet
            string_packet  = pull_socket_packets.recv_string()
            packet         = json.loads(string_packet, cls=Decoder)
            
            start_times['last_received_packet'] = time.time()
            print('\n' if previous_was_if else '', end="")
            previous_was_if = False

        # TODO: Check no packets got lost
        # endregion

        # region _________ Simulate incoming packet reception ________
        # simulated_packet_interval = 1000e-3
        # time_since_last_packet = time.time() - start_times['last_received_packet']
        # if time_since_last_packet < simulated_packet_interval:
        #     time.sleep(simulated_packet_interval - time_since_last_packet)

        # # Update the last received packet time
        # start_times['last_received_packet'] = time.time()
        # print('\n' if previous_was_if else '', end="")
        # previous_was_if = False
        # endregion

        # region _________ Save or dump data _________

        # Create and start a new thread to save the array
        # thread = threading.Thread(target=threaded_dump, args=(packet['data'][:,ch_id], 
        #                                                       f'{repo_dir}/src/TCP/saved/alpha_tests/ch_{ch_id}/data_ch_{ch_id}_bf_{packet["buffer_nb"]}' ))
        # thread.start()
        # thread = threading.Thread(target=threaded_dump, args=(packet['trg_raw_data'], 
        #                                                       f'{repo_dir}/src/TCP/saved/alpha_tests/ch_{ch_id}/trg_ch_{ch_id}_bf_{packet["buffer_nb"]}' ))
        # thread.start()

        # with open(f'packet{i}.json', 'r') as f:
        #     string_packet = f.read()

        # endregion
            
        # region _________ Check if packet is relevant and proceed if it is ________

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


            plot_counter_for_image = 0 # For counting how many times we plotted the buffers in a relevant buffer streak
            continue
        
        # else:
        print(f'{start_char}Packet {packet["buffer_nb"]:>5} received: Relevant', end="")
        start_times['last_rel_packet'] = time.time()
        count_relevant_buffs += 1
        start_char = ""
        #endregion

        # region _________ Count the triggers and check if a natural images has started being displayed  ______
        n_trgs_buffer, detected_triggers_idx, trg_close_to_end_flag, trg_close_to_start_flag = count_triggers(packet['trg_raw_data'].astype(np.int32), 
                                                                                                              trigger_diff_threshold=2000)

        if n_trgs_buffer == 0:
            print(f"\nNo triggers detected in buffer {packet['buffer_nb']}, continue...\n")
            continue
        
        # Make a list of all the detected triggers indexes ( they might be repeted since they start from 0 every buffer )
        detected_triggers_idx_tot = np.append(detected_triggers_idx_tot, detected_triggers_idx + count_relevant_buffs*buffer_size)
        
        # endregion

        # region _________ Run sanity check on the timing of the detected triggers ________
        # Every nb_bff_wait buffers with triggers, check that we missed none
        if count_relevant_buffs % nb_bff_wait == 0 and count_relevant_buffs > 0:

            nb_idx_to_compare = 10
            # if detected_triggers_idx.shape[0] < nb_idx_to_compare:
        #         print(f"\n   Warning: Not enough triggers detected in buffer {packet['buffer_nb']} (last buffer) to compare with the last {nb_idx_to_compare} ")

            last_idxs = detected_triggers_idx_tot[-nb_idx_to_compare:]

            index_diff_avg_prev = np.mean(last_idxs[1:] - last_idxs[:-1]) if last_idxs.shape[0] > 1 else 0

        #     if last_idxs.shape[0] > 1:
        #         print(f"\n   Shape of last indexes: {last_idxs.shape[0]}")

            if np.abs(index_diff_avg_prev-exp_trgs_idx_diff) > np.abs(exp_trgs_idx_diff*0.01):
                print(f"\n   Warning: The average difference between the indexes of the last {last_idxs.shape[0]}: {index_diff_avg_prev} triggers is different from the expected: {exp_trgs_idx_diff}" 
                      f"\n   by more than 1%")
                logging.warning(f"   Warning: The average difference between the indexes of the triggers has changed by more than 1% from the previous {nb_bff_wait} buffers,"
                                f"   a trigger might have been lost")
        # endregion

        # region _________ Plot the received signal around the detected triggers _________

        plot_counter_for_image = update_plot(packet, ch_id, ax, fig, lines, count_relevant_buffs, plot_counter_for_image, max_lines=100)

        # endregion

        # region _________ Edge cases: triggers close to the start or end of the buffer _________    
        if n_trgs_buffer > 0:
            # if trg_close_to_end_flag:
                # logging.info(f"Trigger close to the end detected in buffer {packet['buffer_nb']}")
            # if trg_close_to_start_flag:
                # logging.info(f"Trigger close to the start detected in buffer {packet['buffer_nb']}")

            if (trg_close_to_end_flag_prev_bf and trg_close_to_start_flag):
                logging.info(f"Buffer {packet['buffer_nb']} detected a trigger close to the start, and the previous did so close to the end, reducing n_trgs_buffer: {n_trgs_buffer} by 1")
                logging.info(f"Trigger number reduced by one for buffer {packet['buffer_nb']}")
                n_trgs_buffer -= 1 
                detected_triggers_idx = detected_triggers_idx[1:]
        
        trg_close_to_end_flag_prev_bf = trg_close_to_end_flag
        print(f" triggers :{n_trgs_buffer:>3},", end='' )
        #endregion
        
        n_trgs_tot  += n_trgs_buffer

        if n_trgs_tot != detected_triggers_idx_tot.shape[0]:
            raise ValueError(f"\n   Error: n_trgs_tot: {n_trgs_tot} != detected_triggers_idx_tot.shape[0]: {detected_triggers_idx_tot.shape[0]} During buffer {packet['buffer_nb']}")

        print(f" TOT triggers detected: {n_trgs_tot:>3}.", end='')

        # if image is still in the gray, continue
        if n_trgs_tot <= max_gray_trgs:
            print(f" Gray   : {n_trgs_tot:>2} trgs <= {max_gray_trgs:>2}, waiting...")
            single_nat_img_spk_train = np.array([])
            continue

        # else: first gray has finished, start counting the natural image triggers

        # region ________ Possible initial and ending gray triggers removal________
        # n of natural img in this buffer is the number of total triggers minus the _amount of triggers might have been missing to reach the max_gray_trgs, in the buffer_
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
            ch_bf_peaks_idx  = packet['peaks'][str(ch_id)]             # get the detected spikes in the channel/unit we care about
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

        # Enough natural triggers presented
        # region _________ Start the fit and wait while discarding packets ________
        # if n_trgs_img_tot == max_img_trgs:

        n_ch_spikes = single_nat_img_spk_train.shape[0]

        # Send the threaded DMD off command and wait for confirmation 
        DMD_off_sender_thread = launch_dmd_off_sender()

        # Fit the GP and add the new image ID to the queue
        computation_thread = launch_computation_thread()

        # Reset the variables for the next image
        single_nat_img_spk_train = np.array([])
        n_trgs_tot     = 0
        n_trgs_img_tot = 0

        # Wait for the computation to finish while discarding incoming packets
        wait_for_computation_and_discard()

        # if wait_for_computation_and_discard() returns, join the computation thread
        computation_thread.join()
        with print_lock: print(f"\nGP fit completed, image ID chosen...")
        # endregion 

        # region ________ Get ID, send VEC file and wait for confirmation while discarding packets ________

        # retrieve next available result and removes it from the queue
        chosen_img_id = img_ID_queue.get() 
        rndm_img_id   = random_list_id[pair_img_counter]

        # New ID has been chosen, send it as a VEC file and receive confirmation through a dedicated thread        
        vec_sender_thread = launch_vec_sender()

        # Wait for the VEC reception confirmation from Windows while discarding packets
        wait_for_vec_confirmation()
        # endregion 

        # region ________ Wait for the DMD to turn off or shutdown ________
        while DMD_stopped_event.is_set():
            print(f"\nClient did not yet confirm DMD off")
            time.sleep(1)

        dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
        while dmd_off_time < min_time_dmd_off:
            dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
            print(f"\rWaiting {(dmd_off_time):.2f} since confirmation for DMD to really turn off...", end="")
            pass

        dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
        if dmd_off_time > max_time_dmd_off:
            print(f"Too long, DMD off time: {(dmd_off_time):.2f} > {max_time_dmd_off} seconds, server is shutting down...")
            dmd_off_time = time_since_event_set(threadict['dmd_off_set_time'])
            global_stop_event.set()
            continue
        # endregion 

        if pair_img_counter == n_dataset:
            print("All images displayed, server is shutting down...")
            global_stop_event.set()
            break
        else:
            pair_img_counter+=1
            continue
        # endregion

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
    pull_socket_packets.setsockopt(zmq.LINGER, 0)
    pull_socket_packets.close()
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