# region _______ Imports __________
import numpy as np

# Import the configuration file
from config.config import *
from src.TCP.tcp_utils import *
from src import main_utils
from src.BINVEC.binvec_utils import *

# endregion 
def initial_listener_linux( electrode_info ):
    '''
    This funciton trains the initial model given the electrode estimation for
    hyperparameters.

    - Connects to Windows machine via TCP
        - One socket for sending the VEC file
        - One socket for receiving the responses
        - One socket to send the DMD off command once the whole squence has been received

    - Generates a VEC file for the first 50 random images to show using the DMD
    - Saves the used_img_idxs in the model dict
    - Sends the VEC file to the initial_run_MEA_and_DMD.py script running on Win
    - Receives all responses
    - Using the responses, trains a GP model

    Args:
        electrode_info (dict): Dictionary containing the electrode information

    Returns:
        initial_model (dict): Dictionary containing the initial model fit infos
    
    '''

    # Set up the context and the sockets
    context, pull_socket_packets, req_socket_vec, req_socket_dmd = setup_lin_side_sockets()
    print("Init linux server is running ...")

    # Set up the thread variables
    threadict = setup_thread_vars_linux() # Dictionary containing the threads and events

    try:
        # Upload the natural image dataset
        nat_img_tuple = main_utils.upload_natural_image_dataset(
            dataset_path=img_dataset_path, astensor=False )

        # Set up the start_model given the electrode information
        start_model = main_utils.model_from_electrode_info(
            electrode_info, *nat_img_tuple )# dict of tensors

        # Plot the chosen RF on the checkerboard STA 
        GP_utils.plot_hyperparams_on_STA(
            start_model, STA=None, ax=None )
        
        # Send the VEC file and wait for confirmed reception
        generate_send_wait_vec( 
            start_model, threadict, req_socket_vec, n_gray_trgs_init, n_img_trgs_init, n_end_gray_trgs )

        # Receive response packets and count triggers

        def receive_responses_count_spikes( n_expected_images):
            '''
            Receive the packets from the Windows machine and count the triggers corresponding to each image.

            This function should be called in correspondence to only one VEC file being sent.
                (be it the initial vec or the phase 2 vec where only 2 images are shown)

            We call natural trigers the ones corresponding to natural images being shown.
            Grey triggers the others.

            1.
            Define an ImgPacket, 
                object that defines the sequence of images-packets correspondence:
                - expected number of images ( same as the vec )
                - expected number of relevant packets 
                - image_counter - number of images as detected by counting natural triggers

                - attributes relative to the current image
                - attributes relative to the current packet
                - attributes relative to the current packet/buffer ...?
                    'consecutive_relevant_buffs' : consecutive_relevant_buffs,
                    'n_trgs_tot_in_pair'         : n_trgs_tot_in_pair,
                    'n_trgs_img_tot_in_pair'     : n_trgs_img_tot_in_pair,
                    'detected_triggers_idx_tot'  : detected_triggers_idx_tot,
                    'detected_triggers_idx'      : detected_triggers_idx,
                    'plot_counter_for_image'     : plot_counter_for_image,

            2.
            Receive packets in a thread and add them to the packet queue
                - Launch threaded rcv and decode packet function _init version
                    - Poll every rvc_packet_poll_timeout seconds
                    - Break if the global stop event is set
                    - Add every received packet to queue
                                                            
            3.
            Read along the packet queue in main thread:

                ALL IMAGES IN VEC:
                while image_counter < expected number of images:
            
                    if not is_relevant(packet):
                        if never_received_relevant:
                            - continue
                        else:
                            - break ( end of vec file )

                    else:
                        ONE IMAGE
                        Here we count trgs and spikes for one image ( n_gray_trgs + n_img_trgs )

                        We use the indexes of the triggers corresponding to gray/nat image being presented
                        to select the spikes saved in the array of spikes indexes for one image. (pks_idx)

                        while ImgPacket.current_img.n_trigs==n_trigs_current_img < n_gray_trgs_init + n_img_trgs_init:
                        
                            all variables refer to _current_buffer
                            - count triggers and get
                                n_trgs, idx_trgs, flag for trgs close to end/start
                                - break if no trgs

                            - get only the indexes corresponding to natural images idx_nat_trgs    
                                to be used to select the spikes indexes for the image
                                - n_trgs_img = number of triggers in buffer corresponding to natural image
                                - idx_nat_trgs = idx_trgs[ -n_trgs_img: ]

                            - select the indexes corresponding to peaks of natural images                        
                                nat_img_pks_condition = (pks_idx >= idx_nat_trgs.min()) \
                                                        &  (pks_idx <= idx_nat_trgs.max())
                                nat_pks_idx = pks_idx[nat_img_pks_condition]

                            - add current buffer spike count (nat_pks_idx) to ImgPacket.current_img
                                ImgPacket.current_img.spike_count += nat_pks_idx.size
                            - add current trgs count to ImgPacket.current_img
                                ImgPacket.current_img.n_trigs += n_trgs_img

                            - add triggers to total count of ImgPacket (n_trgs_tot_in_pair += n_trgs_buffer) 
                                ImgPacket.n_trigs += n_trgs

                            - if image still in gray:
                                - ImgPacket.current_img.n_trigs += n_trgs_img
                                - continue
                                        
            '''
            # 1.
            image_packet = ImagePacket()

            # 2.
            launch_threaded_rcv_and_decode( pull_socket_packets, threadict, image_packet )

            # 3.
            # This while treats ALL images expected in the vec
            never_received_relevant = True
            while image_packet.image_counter < n_expected_images:
                
                packet = threadict['packet_q'].get()
                if not is_relevant( packet ):
                    if never_received_relevant: 
                        continue
                    else:
                        break # end of vec file
                else:
                    never_received_relevant = False

                    # This while treats ONE image ( a pair of gray-nat images )
                    while image_packet.current_img.n_trigs < n_gray_trgs_init + n_img_trgs_init:
                        
                        # Count trigs in packet and get flags for trgs close to end/start
                        n_trgs, trgs_idx, border_trgs_flag = count_triggers_init( 
                            trigger_ch_sequence    = packet['trg_raw_data'].astype(np.int32),
                            trigger_diff_threshold = trg_diff_threshold )
                        # Except if no triggers in relevant packet
                        if n_trgs == 0: 
                            threadict['global_stop_event'].set()
                            threadict['exceptions_q'].put( 
                                CustomException('No triggers in relevant packet, some MEA params must be wrong') )
                            return

                        # Remove a trigger if one was detected close to end and start
                        n_trgs, trgs_idx, prev_border_trgs_flag = remove_trg_if_close_to_end_start( 
                            threadict, packet, n_trgs, trgs_idx, border_trgs_flag, prev_border_trgs_flag )

                        
                        idx_nat_trgs, n_trgs_img = get_nat_triggers( trgs_idx, n_trgs, n_img_trgs_init )
                        nat_pks_idx = get_nat_pks_idx( pks_idx, idx_nat_trgs )

                        image_packet.current_img.spike_count += nat_pks_idx.size
                        image_packet.current_img.n_trigs += n_trgs_img
                        image_packet.n_trigs_tot_in_pair += n_trgs

                        if flag == 'end':
                            break
            
            return

        receive_responses_count_spikes( 
            n_expected_images=start_model['fit_parameters']['in_use_idx'].shape[0] )



        time.sleep(5) # simulate reception of responses

        # Send the DMD off command and wait for confirmation 
        DMD_off_sender_thread = launch_dmd_off_sender( req_socket_dmd, threadict)

        # Train the starting GP model with the responses
        # start_model = train_GP_with_responses( start_model, threadict )

        # Wait for DMD stop command to be received ( not necessary when receiving responses )
        while True:
            if threadict['global_stop_event'].wait(timeout=0.1):       break
            if threadict['DMD_stopped_event'].set().wait(timeout=0.1): break
            pass

    except KeyboardInterrupt:
        print('Key Interrups')
        threadict['global_stop_event'].set()

    finally:
        if not threadict['global_stop_event'].is_set():
            threadict['global_stop_event'].set()

        time.sleep(0.2)
        threadict['global_stop_event'].set()     
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

        print('Checking for exceptions in queue...')
        if not threadict['exceptions_q'].empty():
            raise threadict['exceptions_q'].get()
    return


if __name__ == "__main__":

    print(" Starting listener_linux.py as main...")

    electrode_info = main_utils.upload_electrode_info( 
        electrode_info_path, print_info=True, testmode = testmode )
    
    initial_listener_linux( electrode_info )