import zmq
import threading
import time
import json
import base64
import numpy as np
import queue
import matplotlib.pyplot as plt


# Import the configuration file
from config.config import *
from gaussian_processes.Spatial_GP_repo import utils as GP_utils
import src.BINVEC.binvec_utils as binvec_utils

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE} from tcp_utils")

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

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

class ImagePacket:
    def __init__(self):
        # Used only when the buffer contains the end of an images and the beginning of the next one
        self.gray_trgs_prev_buffer = 0 
        self.images        = []
        self.spike_counts  = np.array([])

        self.add_new_image()



    class LastImage:
        '''
        Current image refers to the pair gray+natural image being shown, 
        for a total of n_gray_trgs_init + n_img_trgs_init triggers.
        '''
        def __init__(self, parent):
            self.parent = parent
            self.reset(parent)
        def reset(self, parent):
            self.n_trigs       = 0
            self.gray_trgs     = parent.gray_trgs_prev_buffer
            self.nat_trgs      = 0

            self.trgs_idx      = np.array([]) # all the triggers indexes for current gray-nat image pair
            self.pks_idx       = np.array([]) # all the peaks idxs for current gray-nat image pair
            self.treated_buffs = 0

            self.spike_count   = 0

        def update_trgs_idx(self, bff_trgs_idx):
            '''
            Add to the list of indexes of the triggers, the new triggers indexes
            '''
            np.append(self.trgs_idx, self.treated_buffs*buffer_size + bff_trgs_idx )

        def get_gray_nat_trgs_in_packet(self, n_trgs_current_buffer):
            '''
            Count the triggers corresponding to gray or nat images in the buffer/packet.

            If the curring image trgs count is less than the expected total trggers for the 
            pair, the number of nat and gray triggers is returned.

            A flag is also returned True if the buffer went above the expected number of
            triggers for the pair. In this case the gray triggers should not be added to 
            current image count

            '''

            # max possible gray or natural triggers is n_trgs_current_buffer
            # min possible gray or natural triggers is 0
            # if the current buffer is going above one gray+img pair
            gray_trgs_new_img_flag = False
            if self.n_trigs > n_gray_trgs_init + n_img_trgs_init:
                # The buffer contains some natural triggers first,and then gray triggers
                n_gray_trgs_buffer = self.n_trigs - (n_gray_trgs_init + n_img_trgs_init)
                n_nat_trgs_buffer  = n_trgs_current_buffer - n_gray_trgs_buffer

                # In this case, only the natural triggers should go and increament the 
                # current image counts
                gray_trgs_new_img_flag = True

                assert self.gray_trgs == n_gray_trgs_init, f"Gray triggers count is {self.gray_trgs} and should be {n_gray_trgs_init}"


            # if the current buffer not going above one gray+img pair    
            if self.n_trigs <= n_gray_trgs_init:
                # All triggers must be gray
                n_gray_trgs_buffer = n_trgs_current_buffer 
                n_nat_trgs_buffer  = 0

            else:
                # The buffer contains some natural triggers too
                prev_n_trgs_image  = self.n_trigs - n_trgs_current_buffer 
                if prev_n_trgs_image >= n_gray_trgs_init:
                    # No triggers where gray
                    n_gray_trgs_buffer = 0
                    n_nat_trgs_buffer  = n_trgs_current_buffer
                else:
                    n_gray_trgs_buffer = n_gray_trgs_init - prev_n_trgs_image
                    n_nat_trgs_buffer  = n_trgs_current_buffer - n_gray_trgs_buffer
            
            return n_gray_trgs_buffer, n_nat_trgs_buffer, gray_trgs_new_img_flag

        def update_n_gray_nat_trgs(self, n_trgs_current_buffer):
            '''
            Update the number of gray and nat triggers in the current buffer

            We also update
            '''
            gray_trgs, nat_trgs, new_img_flag = self.get_gray_nat_trgs_in_packet(
                n_trgs_current_buffer)

            self.nat_trgs  += nat_trgs
            if not new_img_flag:
                # Normal case, buffer was below the tot expected number of triggers for the pair
                self.gray_trgs += gray_trgs
                self.gray_trgs_prev_buffer = gray_trgs 
            else:
                self.gray_trgs_prev_buffer = gray_trgs
            
            assert self.nat_trgs == n_img_trgs_init, f"Natural triggers count is {self.nat_trgs} and should be {n_img_trgs_init}"
            
            return

        def update_peaks_idx(self, peaks_idx_current_buffer):
            '''
            Add to the list of indexes of the peaks, the new peaks indexes
            '''
            np.append(self.pks_idx, self.parent.current_image.treated_buffs*buffer_size + peaks_idx_current_buffer )

        def get_spike_count(self):
            assert self.spike_count <= self.pks_idx.shape[0]

            # gray_trgs_idx = self.trgs_idx[:self.gray_trgs]
            nat_trgs_idx  = self.trgs_idx[-self.nat_trgs:]

            assert nat_trgs_idx.shape[0] == n_img_trgs_init, f"Natural triggers count is {nat_trgs_idx.shape[0]} and should be {n_img_trgs_init}"

            # Realistic window for the spike count
            # realistic_window_start_ms  = 30
            # realistic_window_end_ms    = 350
            # # How many triggers after the trigger correpsonding to first natural trigger
            # # the realistic window starts
            # start_window_idx = 0 
            # # How long, in triggers, the realistic window lasts
            # realistic_window_end_idx = 0
            # condition = ( self.pks_idx > self.nat_trgs.min() + start_window_idx \
                # & (self.pks_idx < self.nat_trgs.min() + realistic_window_end_idx) )
            condition = ( self.pks_idx > nat_trgs_idx.min() )

            spikes_in_window = self.pks_idx[condition]

            spike_count = spikes_in_window.shape[0]

            return spike_count

        def set_spike_count(self):
            '''
            Gets the spike count for the current image based on the indexes of the 
            natural image triggers and sets it to the current image object
            
            '''            
            self.spike_count = self.get_spike_count()

            return 


            def get_indices_from_seconds(times_seconds,  sampling_rate=20000):
                """
                Convert time points in seconds to sample indices.
                
                Args:
                    times_seconds: array-like of time points in seconds
                    sampling_rate: sampling frequency in Hz (default 20000)
                    max_idx: maximum index value (length of the data)
                
                Returns:
                    numpy array of integer indices
                """
                # Convert to numpy array if not already
                times = np.asarray(times_seconds)
                
                # Convert to indices with proper rounding
                indices = np.round(times * sampling_rate).astype(np.int64) # sure of the type?
                    
                return indices

            realistic_window_start_idx = get_indices_from_seconds(
                times_seconds=realistic_window_start_ms/1000, 
                sampling_rate=acq_freq)
            realistic_window_end_idx   = get_indices_from_seconds(
                times_seconds=realistic_window_end_ms/1000, 
                sampling_rate=acq_freq)

    def add_new_image(self):
        
        self.last_image = self.LastImage(self)
        self.images.append(self.last_image)
        self.image_counter = len(self.images)


    def get_spike_counts(self):
        '''
        Get the spike counts for all the images in the buffer
        '''
        for img in self.images:
            self.spike_counts = np.append(self.spike_counts, img.spike_count)
        return self.spike_counts
    

def count_triggers( trigger_ch_sequence, trigger_diff_threshold):

    '''Counts the triggers in the provided array, tipically an aquired buffer.
    trigger_ch_sequence:    np.array - the complete sequence of signals aquired from the trigger channel
    trigger_diff_threshold: int      - the difference threshold between two consecutive signals in the sequence to be considered a trigger

    Returns:
    n_triggers:             int      - the number of triggers detected in the sequence
    detected_trigger_idx:   np.array - the indexes of the triggers detected in the sequence

    '''
    
    start = 0
    end   = trigger_ch_sequence.shape[0]

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

def threaded_fit_end_queue_img(new_spike_count, current_img_id, current_model, threadict, ):
    '''
    Updates the fit with the last image spike count, estimate the most useful new image and adds its ID to the queue

    Args:
        new_spike_count (int): The number of spikes for the last image being presented

        init_model (dict): The initial model parameters ( the previous fit)

    Sets:
        fit_finished_event: The event to signal that the fit has finished and the new image ID is in the queue
    '''
    updated_model = update_model(new_spike_count, current_img_id, current_model, threadict['print_lock'])

    img_id        = find_most_useful_img(updated_model, threadict['print_lock'])

    threadict['img_id_queue'].put(img_id)
    with threadict['print_lock']:
        print(f"\n...Fit Thread: imdID {img_id} added to the queue: {img_id}", end="\n")
    threadict['fit_finished_event'].set()
    return

def threaded_vec_send_and_confirm( threadict, req_socket_vec, generate_vec, **kwargs):
    '''
    Thread that sends the provided VEC file to the client and waits for the confirmation

    If generate_vec is True, generates a VEC with che chosen active image ID and random image ID with the provided parameters

    If generate_vec is False, sends the provided VEC file

    Sends to the client the VEC file corresponding the the chosen image ID and wait for confirmation. 
    When received, set the event and allow the main thread to stop discarding packets

    Sets:
        vec_confirmation_event: to True when the client confirms the VEC file reception

        # vec_failure_event:    to True when the client does not confirm the VEC file reception


    '''

    try:
        if generate_vec:
            rndm_img_id     = kwargs.get("rndm_img_id")
            chosen_img_id   = kwargs.get("chosen_img_id")
            n_gray_trgs     = kwargs.get("n_gray_trgs")
            n_img_trgs      = kwargs.get("n_img_trgs")
            n_end_gray_trgs = kwargs.get("n_end_gray_trgs")

            with threadict['print_lock']: 
                print(f"\n...VEC Thread: Generating VEC file for image ID: {chosen_img_id}", end="\n")
            vec_content, vec_path = binvec_utils.generate_vec_file(
                chosen_img_id, rndm_img_id, 
                n_gray_trgs, n_img_trgs, 
                n_end_gray_trgs, 
                )
            with threadict['print_lock']: 
                print(f"\n...VEC Thread: Sending VEC file for image ID: {chosen_img_id}", end="\n")

        if not generate_vec:
            vec_content = kwargs.get("vec_content")

        # Send the VEC file to the client and wait for confirmation
        req_socket_vec.send_string(vec_content)
        # Poll the socket for a reply with a timeout, basically wait for tot milliseconds for a reply
        poll_interval_vec = 100           # Milliseconds
        timeout_vec       = timeout_vec_reception             # Seconds
        poller_vec        = zmq.Poller()
        poller_vec.register(req_socket_vec, zmq.POLLIN)
        start_time_vec = time.time()
        with threadict['print_lock']:
            print(f'\n...VEC Thread: Waiting VEC confirmation from the client, timeout in {timeout_vec} seconds...', end="\n")

        while not threadict['global_stop_event'].is_set() and (time.time() - start_time_vec) < timeout_vec:    
            socks = dict(poller_vec.poll(timeout=poll_interval_vec))
            if req_socket_vec in socks:
                confirmation = req_socket_vec.recv_string()
                if confirmation == 'VEC CONFIRMED':
                    threadict['vec_confirmation_event'].set()
                    with threadict['print_lock']: 
                        print(f"\n...VEC Thread: Client confirmed VEC reception", end="\n")
                    return
                else:
                    print(f"\n...VEC Thread: Error: The client replied with an unexpected message: {confirmation}", end="\n")
                    # threadict['vec_failure_event'].set() 
                    return           
        if threadict['global_stop_event'].is_set():
            with threadict['print_lock']: print(f"\n...VEC Thread: Global stop from outside", end="\n")
            return
        else:
            with threadict['print_lock']: 
                print(f"\n...VEC Thread: Error: Timeout expired without VEC reception confirmation - setting Global stop", 
                    end="\n")
            # threadict['vec_failure_event'].set() 
            threadict['global_stop_event'].set()
            threadict['exceptions_q'].put(CustomException("Timeout expired without VEC reception confirmation"))  
            return
    except Exception as e:
        with threadict['print_lock']:
            print(f"\n...VEC Thread: Unexpected Error: {e}", end="\n")
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(e)
        return

def launch_vec_sender( threadict, req_socket_vec, generate_vec, **kwargs):
    '''
    Launches the thread that generates and sends the VEC file to Windows, and waits for the confirmation.

    Sets:
        vec_confirmation_event: to False, to allow the thread to set it True            
    '''
    if generate_vec:
        rndm_img_id   = kwargs.get("rndm_img_id")
        chosen_img_id = kwargs.get("chosen_img_id")
        n_gray_trgs = kwargs.get("n_gray_trgs")
        n_img_trgs  = kwargs.get("n_img_trgs")
        end_gray_trgs = kwargs.get("n_end_gray_trgs")
        kwargs = {'rndm_img_id':rndm_img_id, 'chosen_img_id':chosen_img_id,
                  'n_gray_trgs': n_gray_trgs, 'n_img_trgs': n_img_trgs, 
                  'n_end_gray_trgs': n_end_gray_trgs}
        
    if not generate_vec:
        vec_content = kwargs.get("vec_content")
        kwargs = {'vec_content': vec_content}
        
    threadict['vec_confirmation_event'].clear()
    args = ( threadict, req_socket_vec, generate_vec )
    vec_sender_thread = threading.Thread(target=threaded_vec_send_and_confirm, args=args, kwargs=kwargs) 
    vec_sender_thread.start()
    return vec_sender_thread

def wait_for_vec_confirmation(pull_socket_packets, threadict):
    '''
    Waits for the VEC reception confirmation from Windows or for the global stop event to be set.

    # While waiting for vec_confirmation_event or vec_failure_event or global_stop_event, all incoming packets are discarded

    Raises:
        CustomException: If the global_stop_event is set
        # CustomException: If the vec_failure_event is set - do we need this event?

    Returns:
        None: If the vec_confirmation_event is set.

    '''
    poller = zmq.Poller()
    poller.register(pull_socket_packets, zmq.POLLIN) 
    while not threadict['vec_confirmation_event'].is_set() and not threadict['vec_failure_event'].is_set() and not threadict['global_stop_event'].is_set():
        socks = dict(poller.poll(timeout=5))
        start_time = time.time()
        if pull_socket_packets in socks:
            string_packet = pull_socket_packets.recv_string()
            packet = json.loads(string_packet, cls=Decoder)
            with threadict['print_lock']:
                print(f"\rDiscarding up to packet {packet['buffer_nb']} while waiting for client to confirm new VEC...", end="")
        else:
            # logging.info('Client has not sent packets in the last 5 milliseconds...')#, end="\r")
            pass # we dont break here, we let the thread break
    if threadict['vec_confirmation_event'].is_set():
        return
    elif threadict['global_stop_event'].is_set():
        raise CustomException("Global stop event set")
    else:
        raise CustomException("vec_failure_event error - vec_confirmation_event not set and global_stop_event not set")            

def generate_send_wait_vec( start_model, threadict, req_socket_vec, n_gray_trgs, n_img_trgs, n_end_gray_trgs ):
    '''
    Used in phase 1 for initial model.

    Generates the Vec with start_model indexes parameters, 

    Launches the vec_sender_thread to send vec and expect confirmation,

    Sleeps until the confirmation event is set or the global_stop_event is set.

    If Exception raises vec_sender_thread, it's added to threadict['exceptions_q'], and
        global_stop_event is set. so it returns

    '''
    # Generate the vec file for the starting 50 images
    vec_content, vec_pathname = binvec_utils.generate_vec_file(
                active_img_ids = start_model['fit_parameters']['in_use_idx'],
                rndm_img_ids   = torch.empty(0),
                n_gray_trgs    = n_gray_trgs,
                n_img_trgs     = n_img_trgs,
                n_end_gray_trgs = n_end_gray_trgs,
                ) 
    # Send the vec file
    vec_sender_thread = launch_vec_sender(
                threadict, 
                req_socket_vec,
                generate_vec = False,
                vec_content  = vec_content,) # kwarg

    while not (threadict['vec_confirmation_event'].is_set() \
            or threadict['global_stop_event'].is_set()):
        time.sleep(1)
    return

def threaded_sender_dmd_off_signal(req_socket_dmd, threadict):
    '''
    Sends a signal to the Windows machine to turn off the DMD and waits for its confirmation

    timeout_dmd_off_confirmation : How long the linux machine waits for confirmation of reception of 
        DMD off command before stopping the server
    
    '''
    req_socket_dmd.send_string("DMD OFF")
    timeout_dmd_off_confirmation = timeout_dmd_off_snd
    with threadict['print_lock']: 
        print(f"\n...DMD off Thread: command sent, response timeout {timeout_dmd_off_confirmation} seconds", 
              end="\n")
    poller = zmq.Poller()       
    poller.register(req_socket_dmd, zmq.POLLIN)
    start_time_dmd = time.time()

    while not threadict['global_stop_event'].is_set() and (time.time() - start_time_dmd) < timeout_dmd_off_confirmation:    
        socks = dict(poller.poll(timeout=100))
        if req_socket_dmd in socks:
            confirmation = req_socket_dmd.recv_string() # The client replies sending back the same message
            if confirmation == "DMD OFF":
                threadict['DMD_stopped_event'].set()
                threadict['dmd_off_set_time'] = time.time()  # Record the time when the event is set
                with threadict['print_lock']: 
                    print(f"\n...DMD off Thread: Client confirmed DMD off {(time.time() - start_time_dmd):.2f}s after request ", 
                          end="\n")    
                return
            else: 
                with threadict['print_lock']: 
                    print(f"\n...DMD off Thread: Error, client replied to DMD request with unexpected message: {confirmation}", end="\n")        
                threadict['exceptions_q'].put(CustomException("Client replied to DMD request with unexpected message"))
                return
            
    if threadict['global_stop_event'].is_set():
        with threadict['print_lock']: print(f"\n...DMD off Thread: Global stop from outside", end="\n")
        return
    else:
        with threadict['print_lock']: print(f"\n...DMD off Thread: Client didn't confirm DMD off, timeout", end="\n")  
        threadict['exceptions_q'].put(CustomException("Timeout expired without DMD off confirmation"))
        threadict['global_stop_event'].set()
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
    np.save(file_path, array)

def launch_dmd_off_sender(req_socket_dmd, threadict):
    '''
    Launches the thread that sends the DMD off command to Windows and waits for the confirmation
    
    The thread will set the DMD_stopped_event to True when the confirmation is received

    Sets:
        - DMD_stopped_event: to False, to allow the thread to set it True
    '''
    threadict['DMD_stopped_event'].clear()
    args = (req_socket_dmd, threadict)
    DMD_off_sender_thread = threading.Thread(target=threaded_sender_dmd_off_signal, args=args) 
    DMD_off_sender_thread.start()
    return DMD_off_sender_thread

def launch_computation_thread( n_ch_spikes, current_img_id, current_model, threadict):
    '''
    Launches the thread that fits the GP and adds the new image ID to the queue
    '''

    print(f"\nStarting the computation thread with {n_ch_spikes} peaks")
    threadict['fit_finished_event'].clear()  
    computation_thread = threading.Thread(target=threaded_fit_end_queue_img, 
                                            args=(n_ch_spikes, current_img_id, current_model, threadict, ))
    computation_thread.start()
    return computation_thread

def wait_for_computation_and_discard(pull_socket_packets, threadict, poll_interval_main):
    '''
    Waits for the computation to finish or for the global stop event to be set
    while discarding all incoming packages from the pull socket:

    '''
    poller_fit = zmq.Poller()
    poller_fit.register(pull_socket_packets, zmq.POLLIN)
    # p=0
    while not threadict['fit_finished_event'].is_set() and not threadict['global_stop_event'].is_set():
        socks = dict(poller_fit.poll(timeout=poll_interval_main))  # with a poller this while can keep going even if the stream stops
        if pull_socket_packets in socks:
            string_packet = pull_socket_packets.recv_string()
            packet = json.loads(string_packet, cls=Decoder)
            with threadict['print_lock']:
                print(f"\rDiscarding up to packet {packet['buffer_nb']} while waiting for computation to finish...", end="")
        else: 
            pass
    if threadict['fit_finished_event'].is_set():
        return
    elif threadict['global_stop_event'].is_set():
        raise CustomException("Global stop event set")
    else:
        raise CustomException("Unknown error - fit_finished_event not set and global_stop_event not set")

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

def print_img_pair_counter(pair_img_counter, loop_counter_prev, start_times):
    '''
        Prints 3 lines indicating which image pair is being processed by the main while loop

        It does 
    
    '''
    if pair_img_counter > loop_counter_prev:
        print(f"\nPrevious loop took: {time.time()-start_times['while_start']:.3f} seconds from the while start")
        print(f"Previous loop took: {time.time()-start_times['last_rel_packet']:.3f} seconds from the last relevant packet")
        print(f"================================================[ {pair_img_counter} Img ]================================================")
        loop_counter_prev = pair_img_counter
    return loop_counter_prev

def rcv_and_decode_packet(
        pull_socket_packets, poll_interval_main, start_times, threadict, main_timeout, pair_img_counter, prev_no_packet_flag):

    '''
    Polls the pull socket for incoming packets and decodes them if there are any.

    Returns:    
        prev_no_packet_flag: bool - True if no packet was received, False otherwise.
                            Used in print statements in main
        packet: dict - The decoded packet if one was received, None otherwise.
    
    '''

    poller_main = zmq.Poller()
    poller_main.register(pull_socket_packets, zmq.POLLIN)
    socks_main = dict(poller_main.poll(timeout=poll_interval_main))  # with a poller this while can keep going even if the stream stops
    if pull_socket_packets not in socks_main:
        elapsed_time = time.time() - start_times['last_received_packet']
        print('' if prev_no_packet_flag else '\n', end="")            
        # print(f"Server has not received packets in the last {(elapsed_time):.3f} seconds...",end="\r")
        prev_no_packet_flag = True
        if elapsed_time > main_timeout and pair_img_counter > 0:
            threadict['global_stop_event'].set()
        return prev_no_packet_flag, None

    else:
        # Receive and decode packet
        string_packet  = pull_socket_packets.recv_string()
        packet         = json.loads(string_packet, cls=Decoder)
        
        start_times['last_received_packet'] = time.time()
        print('\n' if prev_no_packet_flag else '', end="")
        prev_no_packet_flag = False
    return prev_no_packet_flag, packet

def is_relevant(packet):
    if packet['trg_raw_data'].max() < trg_threshold:
        return False
    else:
        return True

def print_relevance_of_packet(is_relevant, packet, prev_was_relevant_flag, image_pair_values, start_times):
    '''
    Prints if a buffer is relevant or not
    '''

    start_char = ""
    if not is_relevant:
        print(f'Packet {packet["buffer_nb"]:>5} received: Not relevant', end="\r")
        
        # The current packet is not relevant, reset the variables for the current image pair 
        reset_image_pairs_variables( image_pair_values )

        prev_was_relevant_flag = False

        # print('\n\n found non relevant packet\n')
    
    elif is_relevant:
        if not prev_was_relevant_flag: start_char = "\n" # otherwhise we are at the beginning of the line
        prev_was_relevant_flag = True
        
        print(f'{start_char}Packet {packet["buffer_nb"]:>5} received: Relevant', end="")
        start_times['last_rel_packet'] = time.time()
        image_pair_values['consecutive_relevant_buffs'] += 1
        # start_char = "\n"

    return prev_was_relevant_flag

def reset_image_pairs_variables( image_pair_values ):
    '''
    Resets to zero the variables that keep track of the relevant buffers and trigger counts for the current image pair
    '''

    image_pair_values['consecutive_relevant_buffs'] = 0
    image_pair_values['n_trgs_tot_in_pair']         = 0
    # gray_trigs_count     = 0
    image_pair_values['detected_triggers_idx_tot']  = np.array([])
    image_pair_values['detected_triggers_idx']      = np.array([])

    image_pair_values['plot_counter_for_image']     = 0 # For counting how many times we plotted the buffers in a relevant buffer streak

    # return consecutive_relevant_buffs, n_trgs_tot_in_pair, detected_triggers_idx_tot, detected_triggers_idx, plot_counter_for_image

def setup_lin_side_sockets():
    '''
    Sets up three sockets:
    One for receiving packets

    One for sending vec files

    One for sending a DMD off signal

    '''

    context = zmq.Context()

    # Create the listening socket as a server
    pull_socket_packets = context.socket(zmq.PULL)
    pull_socket_packets.bind(f"tcp://*:{PULL_SOCKET_PACKETS_PORT}")

    # Create a REQ socket as a client
    req_socket_vec = context.socket(zmq.REQ)
    req_socket_vec.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:{REQ_SOCKET_VEC_PORT}")

    # Create a REQ socket as a client
    req_socket_dmd = context.socket(zmq.REQ)
    req_socket_dmd.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:{REQ_SOCKET_DMD_PORT}")

    return context, pull_socket_packets, req_socket_vec, req_socket_dmd

def setup_thread_vars_linux():
    # Thread variables
    img_id_queue = queue.Queue()

    fit_finished_event         = threading.Event() # Event to signal when the computation thread is finished
    vec_confirmation_event     = threading.Event()
    # vec_failure_event          = threading.Event()
    DMD_stopped_event          = threading.Event()
    global_stop_event          = threading.Event()
    print_lock                 = threading.Lock()
    set_time_lock              = threading.Lock()
    
    exceptions_q               = queue.Queue()
    packets_q                  = queue.Queue()

    # Set up start times var
    start_times = {'global_start': time.time()}
    start_times['last_received_packet'] = start_times['global_start']
    start_times['while_start']          = start_times['global_start']
    start_times['last_rel_packet']      = start_times['global_start']

    
    # Note that locks and events are mutable objects so when modified in a function they are modified in the global scope
    # other objects are immutable so I need to always access them through the dictionary to modify them in a function and have the changes reflected in the global scope
    threadict = {
        "fit_finished_event": fit_finished_event,
        "vec_confirmation_event": vec_confirmation_event,
        # "vec_failure_event": vec_failure_event,
        "DMD_stopped_event": DMD_stopped_event,
        "global_stop_event": global_stop_event,

        "img_id_queue"     : img_id_queue,
        "exceptions_q"     : exceptions_q,
        "packets_q"        : packets_q,

        "start_times"      : start_times,
        "dmd_off_set_time" : None,

        "print_lock"       : print_lock,
        "set_time_lock"    : set_time_lock,
    }
    return threadict

def setup_plot_vars():
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    plt.ion()

    color_index = 0

    plot_counter_for_image = 0 # For counting how many times we plotted the buffers in a relevant buffer streak

    lines = []

    return fig, ax, line, lines, color_index, plot_counter_for_image

def launch_threaded_dump(packet, REPO_DIR):

    thread = threading.Thread(target=threaded_dump, args=(packet['data'][:,ch_id], 
                                                        f'{REPO_DIR}/src/TCP/saved/alpha_tests/ch_{ch_id}/data_ch_{ch_id}_bf_{packet["buffer_nb"]}' ))
    thread.start()
    thread = threading.Thread(target=threaded_dump, args=(packet['trg_raw_data'], 
                                                        f'{REPO_DIR}/src/TCP/saved/alpha_tests/ch_{ch_id}/trg_ch_{ch_id}_bf_{packet["buffer_nb"]}' ))
    thread.start()

    # with open(f'packet{i}.json', 'r') as f:
        # string_packet = f.read()

def update_image_pair_values(image_pair_values, **kwargs): 
    for key, val in kwargs.items():
        if key in image_pair_values: 
            image_pair_values[key] = val 
        else: 
            print(f"Key '{key}' not present in image_pair_values") 
            
    return image_pair_values

def threaded_rcv_and_decode_packet(
        pull_socket_packets, threadict, image_packet):

    '''
    Polls the pull socket for packets and adds them to the queue if there are any.

    Returns:
    '''
    try:

        start_times = threadict['start_times']
        poller_main = zmq.Poller()
        poller_main.register(pull_socket_packets, zmq.POLLIN)

        while not threadict['global_stop_event'].is_set():
            socks_main = dict(poller_main.poll(timeout=100))  # with a poller this while can keep going even if the stream stops
            if pull_socket_packets not in socks_main:
                elapsed_time = time.time() - start_times['last_received_packet']
                print('' if prev_no_packet_flag else '\n', end="")            
                print(f"Server has not received packets in the last {(elapsed_time):.3f} seconds...",end="\r")
                prev_no_packet_flag = True
                if elapsed_time > main_timeout_rcv_packet and image_packet.n_rcvd_imgs > 0:
                    # timeout expired and while receiving images ( not sure its correct to stop here)
                    print('\n')
                    threadict['global_stop_event'].set()
                    print(f"\n...RCV Thread: Stopped from outside")
                return

            else:
                # Receive and decode packet
                string_packet  = pull_socket_packets.recv_string()
                packet         = json.loads(string_packet, cls=Decoder)
                
                start_times['last_received_packet'] = time.time()
                # print('\n' if prev_no_packet_flag else '', end="")
                print('\n', end="")
                prev_no_packet_flag = False
                threadict['packets_q'].put(packet)

    except Exception as e:
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(e)
        with threadict['print_lock']:
            print(f"\n...RCV Thread: Unexpected Error: {e}", end="\n")
        return

def launch_threaded_rcv_and_decode( pull_socket_packets, threadict, image_packet):
    '''
    Launches the thread that receives and decodes packets from the pull socket and adds them to the queue
    '''
    print('Starting the receiving thread')
    args = (pull_socket_packets, threadict, image_packet)
    rcv_thread = threading.Thread(target=threaded_rcv_and_decode_packet, args=args) 
    rcv_thread.start()
    return rcv_thread

def count_triggers_init( trigger_ch_sequence, trigger_diff_threshold):

    '''Counts the triggers in the provided array, tipically an aquired buffer.
    trigger_ch_sequence:    np.array - the complete sequence of signals aquired from the trigger channel
    trigger_diff_threshold: int      - the difference threshold between two consecutive signals in the sequence to be considered a trigger

    Returns:
    n_triggers:             int      - the number of triggers detected in the sequence
    detected_trigger_idx:   np.array - the indexes of the triggers detected in the sequence

    '''
    
    start = 0
    end   = trigger_ch_sequence.shape[0]

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

    border_trgs = {'trgs_close_to_end': trg_close_to_end, 'trgs_close_to_start': trg_close_to_start}

    return n_triggers, detected_trigger_idx, border_trgs

def remove_trg_if_close_to_end_start( threadict, packet, n_trgs, trgs_idx, border_trgs_flag, prev_border_trgs_flag ):
    '''
    Checks if the flags prev_border_trgs_flag is signaling a triger close to the end of the buffer
    of the previous acquired buffer and a trigger close to the start of the current buffer.
    '''
    if not prev_border_trgs_flag:
        # Dictionary is empty, handle the first call case
        prev_border_trgs_flag = {'trgs_close_to_end': False, 'trgs_close_to_start': False}

    if prev_border_trgs_flag['trgs_close_to_end'] and border_trgs_flag['trgs_close_to_start']:
        with threadict['print_lock']:
            print(f'''\nBuffer {packet['buffer_nb']} detected a trigger close to the start, 
                        and the previous did so close to the end, reducing n_trgs_buffer: {n_trgs} by 1''')
            print(f"\nTrigger number reduced by one for buffer {packet['buffer_nb']}")
        n_trgs -= 1 
        trgs_idx = trgs_idx[1:]

    return n_trgs, trgs_idx, border_trgs_flag













