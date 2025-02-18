# region _______ Imports __________
import zmq
import time
import numpy as np
import logging

# Import the configuration file
from config.config import *
from src.TCP.tcp_utils import *
from gaussian_processes.Spatial_GP_repo import utils as GP_utils

# endregion 
def listener_linux( current_model ):
    
    # Set up the context and the sockets
    context, pull_socket_packets, req_socket_vec, req_socket_dmd = setup_lin_side_sockets()
    print("Linux server is running and waiting for data stream...")

    # Set up the thread variables
    threadict = setup_thread_vars() # Dictionary containing the threads and events

    # Variables for plotting
    fig, ax, line, lines, color_index, plot_counter_for_image = setup_plot_vars()

    #region _______________ Variables _______________

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='      %(message)s - %(asctime)s-%(levelname)s - ', datefmt='%M:%S')

    ##### Receive the data stream #####
    consecutive_relevant_buffs = 0 # Number of buffers acquired since the first trigger has been detected after a pause.
    n_trgs_tot_in_pair         = 0 # Number of triggers detected from the first relevant buffer on
    n_trgs_img_tot_in_pair     = 0
    # gray_trigs_count     = 0
    detected_triggers_idx_tot = np.array([])
    detected_triggers_idx     = np.array([])
    # start_char = ""
    
    trg_close_to_end_flag_prev_bf = False # Flag to signal if the last trigger of the previous buffer was close to the end of the buffer

    # DMD triggers parametrs
    nb_bff_wait      = 10 # Number of buffers to wait before checking if we missed any triggers
    index_diff_avg   = np.array([])
    single_nat_img_spk_train = np.array([])

    # MEA acquisition parameters
    buffer_per_second = acq_freq/buffer_size
    ntrgs_per_buffer  = trigger_freq/buffer_per_second
    # The expected difference between the indexes of the triggers, in the last nb_bff_wait buffers the average difference should be around this value
    exp_trgs_idx_diff = buffer_size/ntrgs_per_buffer

    # Image parameters
    random_list_id = np.random.randint(0, n_imgs_dataset, 10)
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
    prev_no_packet_flag    = False # Flag to signal if the last print was an if statement in the case we are not receiving packets
    prev_was_relevant_flag = True  # Flag to signal if the last packet was not relevant
    #endregion

    # Variables relative to a certain image pair being treated
    # image_pair_values = {
    #     'consecutive_relevant_buffs' : consecutive_relevant_buffs,
    #     'n_trgs_tot_in_pair'         : n_trgs_tot_in_pair,
    #     'n_trgs_img_tot_in_pair'     : n_trgs_img_tot_in_pair,
    #     'detected_triggers_idx_tot'  : detected_triggers_idx_tot,
    #     'detected_triggers_idx'      : detected_triggers_idx,
    #     'plot_counter_for_image'     : plot_counter_for_image,
    # }

    try:
        '''
        This while should 
        - define a new ImgPair
        - listen for packets and update ImgPair values until sufficient triggers are detected
        - fit the GP and send the image ID to the client via VEC
        - wait for the client to confirm VEC reception
        '''

        while not threadict['global_stop_event'].is_set():
            ''' The client is sending packets (dictionaries) with the following structure:
            {'buffer_nb': 10, 'n_peaks': 0,'peaks': {'ch_nb from 0 to 255': np.array(shape=n of peaks in buffer with 'timestamp') } }'}}
                - Unpackable using the custom Decoder class
                - buffer_nb: the number of the buffer
                - n_peaks: the number of peaks in the buffer, already computed by the client
                - peaks: dictionary having as keys the channels and as values the indices of detected peaks 
                -'trg_raw_data': the trigger channel raw data, unfiltered
            _____________________________'''

            start_times['while_start'] = time.time()
            image_pair_values = {
                'consecutive_relevant_buffs' : consecutive_relevant_buffs,
                'n_trgs_tot_in_pair'         : n_trgs_tot_in_pair,
                'n_trgs_img_tot_in_pair'     : n_trgs_img_tot_in_pair,
                'detected_triggers_idx_tot'  : detected_triggers_idx_tot,
                'detected_triggers_idx'      : detected_triggers_idx,
                'plot_counter_for_image'     : plot_counter_for_image,
            }
            

        # region _________ Receive packets until sufficent natural triggers are detected ________

            # Print the loop counter every time it changes ( every time a new image pair is treated )
            loop_counter_prev = print_img_pair_counter(pair_img_counter, loop_counter_prev, start_times)

            # region _________ Receive and decode packet ________
            prev_no_packet_flag, packet = rcv_and_decode_packet(
                pull_socket_packets, poll_interval_main, start_times, threadict, main_timeout, pair_img_counter, prev_no_packet_flag)
            if packet is None:
                continue

            # TODO: Check no packets got lost
            # endregion

            # region _________ Save or dump data _________
            # Create and start a new thread to save the array
            # launch_threaded_dump(packet, repo_dir)
            # endregion    

            # region _________ Check if packet is relevant and proceed if it is ________

            # To do this, check the trigger channel (127 on the MEA, so 126 here) it is above a certain threshold ( ~ 5.2*1e5)
            # If the buffer never crosses the threshold, discard it

            isrelevant            = is_relevant(packet)
            prev_was_relevant_flag = print_relevance_of_packet( 
                isrelevant, packet, prev_was_relevant_flag, image_pair_values, start_times)
            
            
            consecutive_relevant_buffs = image_pair_values['consecutive_relevant_buffs']
            n_trgs_tot_in_pair         = image_pair_values['n_trgs_tot_in_pair']
            detected_triggers_idx_tot  = image_pair_values['detected_triggers_idx_tot']
            detected_triggers_idx      = image_pair_values['detected_triggers_idx']
            plot_counter_for_image     = image_pair_values['plot_counter_for_image']

            if not isrelevant: 
                continue

            #endregion

            # region _________ Count the triggers and check if a natural images has started being displayed  ______
            n_trgs_buffer, detected_triggers_idx, trg_close_to_end_flag, trg_close_to_start_flag = count_triggers(
                packet['trg_raw_data'].astype(np.int32),  trigger_diff_threshold=trg_diff_threshold)

            if n_trgs_buffer == 0:
                print(f"\nNo triggers detected in buffer {packet['buffer_nb']}, continue...\n")
                continue
                    
            # endregion

            # region _________ Run sanity check on the timing of the detected triggers ________
            # Every nb_bff_wait buffers with triggers, check that we missed none
            # if consecutive_relevant_buffs % nb_bff_wait == 0 and consecutive_relevant_buffs > 0:

            #     nb_idx_to_compare = 10
            #     # if detected_triggers_idx.shape[0] < nb_idx_to_compare:
            # #         print(f"\n   Warning: Not enough triggers detected in buffer {packet['buffer_nb']} (last buffer) to compare with the last {nb_idx_to_compare} ")

            #     last_idxs = detected_triggers_idx_tot[-nb_idx_to_compare:]

            #     index_diff_avg_prev = np.mean(last_idxs[1:] - last_idxs[:-1]) if last_idxs.shape[0] > 1 else 0

            # #     if last_idxs.shape[0] > 1:
            # #         print(f"\n   Shape of last indexes: {last_idxs.shape[0]}")

            #     if np.abs(index_diff_avg_prev-exp_trgs_idx_diff) > np.abs(exp_trgs_idx_diff*0.01):
            #         print(f"\n   Warning: The average difference between the indexes of the last {last_idxs.shape[0]}: {index_diff_avg_prev} triggers is different from the expected: {exp_trgs_idx_diff}" 
            #               f"\n   by more than 1%")
            #         logging.warning(f"   Warning: The average difference between the indexes of the triggers has changed by more than 1% from the previous {nb_bff_wait} buffers,"
            #                         f"   a trigger might have been lost")
            # endregion

            # region _________ Plot the received signal around the detected triggers _________

            # plot_counter_for_image = update_plot( 
                # packet, ch_id, ax, fig, lines, consecutive_relevant_buffs, plot_counter_for_image, max_lines=100)

            # endregion

            # region _________ Edge cases: triggers close to the start or end of the buffer _________    
            if n_trgs_buffer > 0:
                # if trg_close_to_end_flag:
                    # logging.info(f"Trigger close to the end detected in buffer {packet['buffer_nb']}")
                # if trg_close_to_start_flag:
                    # logging.info(f"Trigger close to the start detected in buffer {packet['buffer_nb']}")

                if (trg_close_to_end_flag_prev_bf and trg_close_to_start_flag):
                    with threadict['print_lock']:
                        print(f'''\nBuffer {packet['buffer_nb']} detected a trigger close to the start, 
                                    and the previous did so close to the end, reducing n_trgs_buffer: {n_trgs_buffer} by 1''')
                        print(f"\nTrigger number reduced by one for buffer {packet['buffer_nb']}")
                    n_trgs_buffer -= 1 
                    detected_triggers_idx = detected_triggers_idx[1:]
            
            trg_close_to_end_flag_prev_bf = trg_close_to_end_flag
            print(f" triggers :{n_trgs_buffer:>3},", end='' )
            #endregion
            
            n_trgs_tot_in_pair += n_trgs_buffer

            # Make a list of all the detected triggers indexes. 
            # They go over the buffer size
            detected_triggers_idx_tot = np.append(detected_triggers_idx_tot, detected_triggers_idx + consecutive_relevant_buffs*buffer_size)

            if n_trgs_tot_in_pair != detected_triggers_idx_tot.shape[0]:
                raise ValueError(f"\n   Error: n_trgs_tot_in_pair:{n_trgs_tot_in_pair} = detected_triggers_idx_tot.shape[0]: {detected_triggers_idx_tot.shape[0]} During buffer {packet['buffer_nb']}")

            print(f" TOT triggers detected: {n_trgs_tot_in_pair:}.", end='')

            # if image is still in the gray, continue
            if n_trgs_tot_in_pair <= max_gray_trgs:
                print(f" Gray   : {n_trgs_tot_in_pair:} trgs <= {max_gray_trgs:>2}, waiting...")
                single_nat_img_spk_train = np.array([])
                continue

            # else: first gray has finished, start counting the natural image triggers

            # region ________ Possible initial and ending gray triggers removal________
            # n of natural img in this buffer is the number of total triggers minus the _amount of triggers might have been missing to reach the max_gray_trgs, in the buffer_
            # this quantity is positive if this trigger was the one getting over the max_gray_trgs
            # otherwise it is negative

            # number of triggers of the current buffer that have been used to reach the max_gray_trgs
            n_trigs_tot_prev      = n_trgs_tot_in_pair - n_trgs_buffer        # previous count of total triggers
            n_trgs_spent_for_gray = max_gray_trgs - n_trigs_tot_prev  # triggers of this buffer that have been used to reach the max_gray_trgs
            # If none of the current buffer triggers where part of the gray, n_starting_gray_trgs = 0
            if n_trgs_spent_for_gray <= 0: 
                n_starting_gray_trgs = 0
            else:
                n_starting_gray_trgs = n_trgs_spent_for_gray          # triggers of this buffer that have been used to reach the max_gray_trgs

            # Remove possible starting gray triggers from counters and indices array
            n_trgs_img      = n_trgs_buffer - n_starting_gray_trgs    # n of natural img triggers in this buffer
            n_trgs_img_tot_in_pair += n_trgs_img
            idx_natural_img_start = detected_triggers_idx[-n_trgs_img:] 

            # now do the same for the ending gray triggers. This buffer might be at the end of the natural image, and already have some gray triggrs
            n_trgs_already_gray =  n_trgs_img_tot_in_pair - max_img_trgs
            if n_trgs_already_gray > 0:
                n_ending_gray_trgs = n_trgs_already_gray
            else:
                n_ending_gray_trgs = 0

            # Remove possible ending gray triggers
            n_trgs_img     -= n_ending_gray_trgs                       # n of natural img triggers in this buffer
            n_trgs_img_tot_in_pair -= n_ending_gray_trgs                       # n of natural img triggers until now
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

            if n_trgs_img_tot_in_pair < max_img_trgs:
                print(f" Natural: {n_trgs_img_tot_in_pair:>2} trgs <= {max_img_trgs}, waiting...",)
                continue

        # endregion
        
            # _____________________________ Enough natural triggers presented _____________________________

            if n_trgs_img_tot_in_pair > max_img_trgs:
                print(f"   Best image is being computed...",)
                continue

            
            # region _________ Start the fit and wait while discarding packets ________
            # if n_trgs_img_tot_in_pair == max_img_trgs:

            n_ch_spikes = single_nat_img_spk_train.shape[0]

            # Send the threaded DMD off command and wait for confirmation 
            DMD_off_sender_thread = launch_dmd_off_sender( req_socket_dmd, threadict)

            # Fit the GP and add the new image ID to the queue
            computation_thread = launch_computation_thread( n_ch_spikes, current_img_id, current_model, threadict)

            # Reset the image pair variables for the next image
            # Prev version - no image_pair_values var
            # single_nat_img_spk_train = np.array([])
            # n_trgs_tot_in_pair     = 0
            # n_trgs_img_tot_in_pair = 0

            reset_image_pairs_variables( image_pair_values )
            consecutive_relevant_buffs = image_pair_values['consecutive_relevant_buffs']
            n_trgs_tot_in_pair         = image_pair_values['n_trgs_tot_in_pair']
            detected_triggers_idx_tot  = image_pair_values['detected_triggers_idx_tot']
            detected_triggers_idx      = image_pair_values['detected_triggers_idx']
            plot_counter_for_image     = image_pair_values['plot_counter_for_image']


            # Wait for the computation to finish while discarding incoming packets
            wait_for_computation_and_discard(pull_socket_packets, threadict, poll_interval_main)

            # if wait_for_computation_and_discard() returns, join the computation thread
            computation_thread.join()
            with threadict['print_lock']: print(f"\nGP fit completed, image ID chosen...")
            # endregion 

            # region ________ Get ID, send VEC file and wait for confirmation while discarding packets ________

            # retrieve next available result and removes it from the queue
            chosen_img_id = threadict['img_ID_queue'].get() 
            rndm_img_id   = random_list_id[pair_img_counter]

            # New ID has been chosen, send it as a VEC file and receive confirmation through a dedicated thread        
            vec_sender_thread = launch_vec_sender(
                threadict, chosen_img_id, rndm_img_id, req_socket_vec, max_gray_trgs, max_img_trgs)

            # Wait for the VEC reception confirmation from Windows while discarding packets
            wait_for_vec_confirmation( pull_socket_packets, threadict)
            # endregion 

            # region ________ Wait for the DMD to turn off or shutdown ________

            while not threadict['DMD_stopped_event'].is_set():
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
                threadict['global_stop_event'].set()
                continue
            # endregion 

            if pair_img_counter == n_imgs_dataset:
                print("All images displayed, server is shutting down...")
                threadict['global_stop_event'].set()
                break
            else:
                # Only if we got to this point  it means we treated a whole image pair sequence
                pair_img_counter += 1 
                continue
            # endregion

            # Get the result of the computation thread and send it to the client
            # print(f"Next image to display: {chosen_img_id}")
            # computation_thread.join()
            # continue

        else:
            print("\nGlobal stop event set, server is shutting down...")
            pass
    except KeyboardInterrupt:
        print("\n\nKeyInterrupted: Server is shutting down...")
        threadict['global_stop_event'].set() 
        print("Stop event set...")    

    finally:
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

    return

if __name__ == "__main__":

    print(" Starting listener_linux.py as main...")
    electrode_info = {
        'best_electrode' : 0,
        'hyperparameters': [1, 1, 1]
    }
    current_model = {
        'current_img_id' : 0,
        'current_model'  : electrode_info
    }

    listener_linux( current_model )