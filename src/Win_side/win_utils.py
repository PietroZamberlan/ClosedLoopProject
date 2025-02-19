import time
import os
import zmq
import threading
import subprocess
import queue

from config.config import *

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
            raise CustomException("OrtProcessFail: Probably NoDeviceConnected: No MEA device connected")
        if return_code is None:
            pass
        time.sleep(0.5)
    else:
        print("Signal file to start DMD found...")
        return

def launch_ort_process( ORT_READER_PATH, ort_reader_params, log_file_ort, testmode ):
    '''Launches the ort reader process to acquire data from the MEA device.
        
        Sleeps for 2.5 the main thread to wait for ort_reader to get to the point 
        at which it writes the signal file or fails cause no device is connected.
        
    Args:
        ORT_READER_PATH: The path to the Python script that launches the acquisition
        ort_reader_params: The parameters to pass to the Python script'''
    # Run the Python script with the parameters
    if testmode:
        print("Launching MEA acquisition in TEST mode...")
    else:
        print("Launching MEA acquisition...")
    # We set the -u flag to avoid buffering the output ( it would stop the OnChannelData to print the data in real time )
    ort_process = subprocess.Popen(["python", '-u', ORT_READER_PATH] + 
                                   ort_reader_params + (["-T"] if testmode else []), 
                                   stdout=log_file_ort, stderr=log_file_ort, text=True)
    print("Acquisition is running...")

    def flush_log():
        while ort_process.poll() is None:
            log_file_ort.flush()
            time.sleep(1)
        log_file_ort.flush() # one last flush when the process finishes

    flush_thread = threading.Thread(target=flush_log, )
    flush_thread.start()

    time.sleep(.5) # the ort process takes some time to get to the point in which it writes the signal file, so we wait a bit instead of printing a lot of "waiting for signal file"

    return ort_process

def threaded_rcv_dmd_off_signal( rep_socket_dmd, threadict, timeout_dmd_off_rcv):
    
    '''
    Threaded function to wait for the signal to turn of the DMD coming from the 
    Linux machine.
    

    timeout_dmd_off_rcv: How much time to wait before stopping the
            DMD without signal from Linux. 
            The DMD is not prasenting more than 3 images after all. Only so 
            much time should be needed.
    ( set in config.py )
    '''
    
    try:
        # Poll the socket for command with a timeout - set in config.py
        start_time = time.time()
        poller = zmq.Poller()
        poller.register(rep_socket_dmd, zmq.POLLIN)
        print(f'...DMD Off cmd receiver Thread: Waiting for command, timeout {timeout_dmd_off_rcv} seconds')
        
        while not threadict['global_stop_event'].is_set():
            elapsed_time = time.time() - start_time
            socks = dict(poller.poll(timeout=100)) 
            # print('\n ...DMD Off cmd receiver Thread: Waiting for command since {} seconds'.format(elapsed_time))            
            if rep_socket_dmd in socks:
                request = rep_socket_dmd.recv_string()

                rep_socket_dmd.send_string(request)
                print(f"...DMD Off cmd receiver Thread: Stop command received and confirmed {elapsed_time:.3f} seconds")   
                threadict['dmd_off_event'].set()
                return
            if elapsed_time > timeout_dmd_off_rcv:
                print("...DMD Off cmd receiver Thread: Timeout expired")
                threadict['dmd_off_event'].set()
                print('...DMD Off cmd receiver Thread: off event set')
                return
        else:
            threadict['dmd_off_event'].set()
            print('...DMD Off cmd receiver Thread: Stopped from outside')
            print('...DMD Off cmd receiver Thread: off event set')
            return
    except Exception as e:
        threadict['dmd_off_event'].set()
        threadict['global_stop_event'].set()
        print('...DMD Off cmd receiver Thread: EXCEPTION encountered - global_stop_event and dmd_off_event set')
        threadict['exceptions_q'].put(e)
        return

def threaded_wait_for_vec( rep_socket_vec, threadict, timeout_vec, vec_pathname_dmd_source):
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
        start_time   = time.time()    
        elapsed_time = 0
        # Poll the socket for a reply with a timeout
        poller = zmq.Poller()
        poller.register(rep_socket_vec, zmq.POLLIN)

        print(f'...VEC Thread: Waiting for vec file, timeout {timeout_vec} seconds')
        while not threadict['global_stop_event'].is_set() and elapsed_time < timeout_vec:
            elapsed_time = time.time() - start_time
            socks = dict(poller.poll(timeout=10)) # milliseconds 
            if rep_socket_vec in socks:
                vec_content = rep_socket_vec.recv_string()
                print("...VEC Thread: VEC received, waiting for auth to overwrite from DMD Thread")
                # Wait for authorization to overwrite the vec file. If this event is not set
                # it means the DMD is showing the images (using the .VEC)
                threadict['allow_vec_changes_event'].wait()
                with open(vec_pathname_dmd_source, "w") as file:
                    file.write(vec_content)
                    # Send confirmation back to the server
                    rep_socket_vec.send_string("VEC CONFIRMED")
                    threadict['vec_received_confirmed_event'].set()
                    print('...VEC Thread: Confirmation sent')
                    return
        else:
            if threadict['global_stop_event'].is_set():
                print('...VEC Thread: Stopped from outside')
                return
        # If the timeout expires, we set the global stop - 
        # It is the Linux machine responsibility to send the VEC file
        print("...VEC Thread: Timeout expired and no VEC received, setting global_stop_event")
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(CustomException("Timeout expired waiting for VEC file"))
        return
    except Exception as e:
        threadict['global_stop_event'].set()
        print('...VEC Thread: EXCEPTION encountered - global_stop_event set')
        threadict['exceptions_q'].put(e)
        return
    
def threaded_DMD(DMD_EXE_PATH, DMD_EXE_DIR, input_data_DMD, threadict, log_file_DMD=None, testmode=True):
    '''
    Launches the DMD projector process with the .VEC file of images specified in input_data_DMD.

    It also listens for the dmd_off_event catched by threaded_rcv_dmd_off_signal() to terminate the process.

    Before doing so, we need to wait for the process to be timed out, this is why we set a very short
    timeout of 0.1 with .communicate().

    Sets:
      allow_vec_changes_event: to False, to prevent changes to the VEC file that vec_receiver_confirmer_thread could do    
                            Only when the DMD terminated the VEC modifications are freed.
    
    '''
    
    if not testmode:
        try:
            DMD_process = subprocess.Popen([DMD_EXE_PATH], cwd=DMD_EXE_DIR,
                                        stdin=subprocess.PIPE, stdout=log_file_DMD,
                                        stderr=log_file_DMD, text=True )    
            threadict['process_queue_DMD'].put(DMD_process)
            # DMD autofill, we need the process to be timed out to be able to close it
            stdout, stderr = DMD_process.communicate(input=input_data_DMD, timeout=0.1)

            print(f'...DMD Thread: DMD was launched, waiting ...')
            DMD_process.wait()
            print(f'...DMD Thread: DMD process terminated')
        except subprocess.TimeoutExpired:
            print('...DMD Thread: DMD was launched and can be terminated')
    else:
        print('...DMD Thread: DMD was launched in TEST mode')

    try:        
        while not threadict['dmd_off_event'].is_set():

            pass
        else:
            if not testmode: DMD_process.terminate()
            # Wait for authorization to overwrite the vec file. If this event is not set
            # it means
            threadict['allow_vec_changes_event'].set()
            print('...DMD Thread: DMD process terminated, VEC file can be modified')
            return
        
    except Exception as e:

        print('...DMD Thread: EXCEPTION encountered - setting global_stop_event')
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(e)
        return

def threaded_DMD_phase1(DMD_EXE_PATH, DMD_EXE_DIR, input_data_DMD, threadict, log_file_DMD, testmode=True):
    '''
    Launches the DMD projector process with the .VEC file of images specified in input_data_DMD.

    It also listens for the dmd_off_event catched by threaded_rcv_dmd_off_signal() to terminate the process.

    Before doing so, we need to wait for the process to be timed out, this is why we set a very short
    timeout of 0.1 with .communicate().

    Sets:
      allow_vec_changes_event: to False, to prevent changes to the VEC file that vec_receiver_confirmer_thread could do    
                            Only when the DMD terminated the VEC modifications are freed.
    
    '''
    
    if not testmode:
        try:
            DMD_process = subprocess.Popen([DMD_EXE_PATH], cwd=DMD_EXE_DIR,
                                        stdin=subprocess.PIPE, stdout=log_file_DMD,
                                        stderr=log_file_DMD, text=True )    
            threadict['process_queue_DMD'].put(DMD_process)
            # DMD autofill, we need the process to be timed out to be able to close it
            stdout, stderr = DMD_process.communicate(input=input_data_DMD, timeout=0.1)
        except subprocess.TimeoutExpired:
            print('...DMD Thread: DMD was launched and can be terminated')
    else:
        print('...DMD Thread: DMD was launched in TEST mode')

    try:    
        if not testmode:
            def flush_log():
                while DMD_process.poll() is None:
                    log_file_DMD.flush()
                    time.sleep(1)
                log_file_DMD.flush() # one last flush when the process finishes

            flush_thread = threading.Thread(target=flush_log, )
            flush_thread.start()    
        while not threadict['dmd_off_event'].is_set():
            pass
        else:
            if not testmode: DMD_process.terminate()
            # Wait for authorization to overwrite the vec file. If this event is not set
            # it means
            threadict['allow_vec_changes_event'].set()
            print('...DMD Thread: DMD process terminated, VEC file can be modified')
            return
        
    except Exception as e:

        print('...DMD Thread: EXCEPTION encountered - setting global_stop_event')
        threadict['global_stop_event'].set()
        threadict['exceptions_q'].put(e)
        return

def launch_dmd_off_receiver(rep_socket_dmd, threadict, timeout_dmd_off_rcv):
    '''Launches the DMD Off cmd receiver Thread
    
    Sets:
        dmd_off_event: to False, to receive the signal to turn off the DMD
        global_stop_event: to False, to allow stopping the thread from outside
    '''
    
    print('Launching DMD off cmd receiver listening thread')
    # Launch parallel function to listen for DMD off command
    threadict['dmd_off_event'].clear()
    threadict['global_stop_event'].clear()               
    args = (rep_socket_dmd, threadict, timeout_dmd_off_rcv)
    DMD_off_listening_thread = threading.Thread(target=threaded_rcv_dmd_off_signal, args=args) 
    DMD_off_listening_thread.start()                
    return DMD_off_listening_thread

def launch_vec_receiver_confirmer(rep_socket_vec, threadict, timeout_vec, vec_pathname_dmd_source):
    '''Launches the VEC receiver and confirmer thread

        It will wait for the VEC from Linux machine

        Once received, it will overwrite the VEC if that is allowed by the allow_vec_changes_event set
        (by the DMD thread).

        Only after overwriting it will allow the main thread to continue with vec_received_confirmed_event is set.
        (vec_received_confirmed_event also confirms that the Linux machine has stopped discarding packets)

    Args:
        vec_pathname_dmd_source: the math of the vec file the DMD is using
    Sets:
        vec_received_confirmed_event: to False, this stops the main thread until the VEC is received
                                    and confirmed by threaded_wait_for_vec
        failure_event:                to False, 
        global_stop_event:            to False, to allow stopping the thread from outside
    '''

    print('Launching VEC confirmation thread')
    # Launch parallel function to wait for response
    threadict['vec_received_confirmed_event'].clear()
    args = (rep_socket_vec, threadict, timeout_vec, vec_pathname_dmd_source)
    vec_receiver_confirmer_thread = threading.Thread(target=threaded_wait_for_vec, args=args) 
    vec_receiver_confirmer_thread.start()

    return vec_receiver_confirmer_thread

def launch_DMD_process_thread(input_data_DMD, threadict, log_file_DMD, testmode=True):
    '''
    Launches the thread that starts the DMD projector.
    The thread will:
        put the process in a state in which it can be terminated
        
        wait for signal dmd_off_event set by threaded_rcv_dmd_off_signal to turn off the DMD

        set allow_vec_changes_event: to False, to prevent changes to the VEC file that vec_receiver_confirmer_thread could do    
                                    Only when the DMD terminated the VEC can be modified
    '''
    if testmode:
        print('Launching DMD subprocess thread in TEST mode')
    else:    
        print('Launching DMD subprocess thread')
    # Wait for authorization to overwrite the vec file. If this event is not set
    # it means the DMD is still showing the images (using the .VEC)
    threadict['allow_vec_changes_event'].clear()                
    args_DMD_thread = (DMD_EXE_PATH, DMD_EXE_DIR, input_data_DMD, threadict, log_file_DMD, testmode)
    
    # DMD_thread      = threading.Thread(target=threaded_DMD, args=args_DMD_thread) 
    print('Launching DMD subprocess thread PHASE 1')
    DMD_thread      = threading.Thread(target=threaded_DMD_phase1, args=args_DMD_thread) 

    DMD_thread.start()

    return DMD_thread

def join_treads(threads):
    for thread in threads:
        if thread is None:
            continue
        if thread.is_alive():
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

def terminate_ort_process(ort_process, log_file_ort):
    print("Terminating ort subprocess")
    if ort_process.poll() is None: 
        ort_process.terminate()
        log_file_ort.flush()
    print('ort process terminated')

def close_socket(socket):
    socket.setsockopt(zmq.LINGER, 0)
    socket.close()

def close_sockets(sockets):
    for socket in sockets:
        if socket is not None:
            print(f'Closing socket {socket}')
            close_socket(socket)

def generate_packet(buffer_nb):
    '''
    Generates a packet dicitonary as it would come out of the MEA.

    Args:
        buffer_nb (int): The buffer / packet number used to keep track of how many packets have been sent.
    
    '''
    
    packet = {}

    data_path              = electrode_raw_data_path / f'ch_{ch_id}' / f'data_ch_{ch_id}_bf_{buffer_nb}.npy'
    trgs_path              = electrode_raw_data_path / 'trg_ch' / f'trg_ch_bf_{buffer_nb}.npy'

    packet['data']         = np.load(data_path).astype(np.int32)
    packet['data']         = np.tile(packet['data'], (256, 1))
    packet['trg_raw_data'] = np.load(trgs_path).astype(np.int32)
    packet['buffer_nb']    = buffer_nb

    return packet

def ask_to_continue(testmode):
    if not testmode: 
        choice = input(f"Transfer electrode_info file to {electrode_info_path} on Linux machine.\nThen press 'Y' to continue.\nAnything else to cancel: ")
    else:
        print(f'User input to continue automatized by TEST mode')
        print("Continuing ...")
        return True

    if choice.lower() != 'y':
        print("Aborted by user.")
        return False
    else:
        print("Continuing ...")
        return True

def setup_win_thread_vars():
    '''
    Sets up the thread variables needed on Windows side
    '''
    vec_received_confirmed_event = threading.Event()
    global_stop_event            = threading.Event()
    dmd_off_event                = threading.Event()
    # Wait for authorization to overwrite the vec file. If this event is not set
    # it means
    allow_vec_changes_event    = threading.Event()
    process_queue_DMD          = queue.Queue()
    exceptions_q               = queue.Queue()

    threadict = {
        'vec_received_confirmed_event': vec_received_confirmed_event,
        'global_stop_event':            global_stop_event,
        'dmd_off_event':                dmd_off_event,
        'allow_vec_changes_event':      allow_vec_changes_event,
        'process_queue_DMD':            process_queue_DMD,
        'exceptions_q':                 exceptions_q
    }

    DMD_off_listening_thread, \
        DMD_thread, vec_receiver_confirmer_thread \
            = None, None, None

    return threadict, DMD_off_listening_thread, DMD_thread, vec_receiver_confirmer_thread

def setup_win_side_sockets():
    '''
    Sets up the VEC reception and DMD process control sockets.

    The socket for packet sending is set up in the ORT subprocess.
    '''


    # Listening socket
    context     = zmq.Context()

    rep_socket_vec = context.socket(zmq.REP)
    rep_socket_vec.bind(f"tcp://0.0.0.0:{REQ_SOCKET_VEC_PORT}")

    rep_socket_dmd = context.socket(zmq.REP)
    rep_socket_dmd.bind(f"tcp://0.0.0.0:{REQ_SOCKET_DMD_PORT}")

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('   Windows client is running...')

    return context, rep_socket_vec, rep_socket_dmd

def wait_vec_reception(rep_socket_vec, threadict, timeout_vec, vec_pathname_dmd_source):
    '''
    Waiter that waits for the VEC file to be received and confirmed by Win machine.

    Used in the init_run_MEA_DMD to stop the execution until VEC is received.

    It CALLS threaded_wait_for_VEC_file with launch_vec_receiver_confirmer to 
    actually executed a dedicated thread for it. 
    '''
    vec_receiver_confirmer_thread = launch_vec_receiver_confirmer(
        rep_socket_vec, threadict, timeout_vec, vec_pathname_dmd_source)
    while not (threadict['vec_received_confirmed_event'].is_set() \
               or threadict['global_stop_event'].is_set()):
        time.sleep(1)

    return vec_receiver_confirmer_thread





















