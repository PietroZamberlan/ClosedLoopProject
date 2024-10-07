# Document for tracking TCP protocol progres and missing features

For a general list of tasks and progress, see the [TODO.md](../TODO.md) file.

For now the TCP protocol on the windows machine can: 

Receive packets from a zmq.PUSH server, sent by the Windows machine.

We keep track of the number of peacks after an image, and number of packets taht have been treated for a given image.

A string packet is received and decoded with the custom class Decoder, here the followinf informations are contined in a dictionary:

```{'buffer_nb': 10, 'n_peaks': 0,'peaks': {'ch_nb from 0 to 255': np.array(shape=n of peaks in buffer with 'timestamp') } }'}}
            - Unpackable using the custom Decoder class
            - buffer_nb: the number of the buffer
            - n_peaks: the number of peaks in the buffer, already computed by the client
            - peaks: dictionary containing the peaks in each channel
            -'trg_raw_data': the trigger channel raw data, unfiltered
```

If needed, dump the data to disk

- [ ] Implement way check if the recorded spikes are still influenced by the latest shown image.
    - [ ] If no threshold crosses are present in the packet, it's surely non relevant
    - [ ] If the threshold has been crossed at least once, the packet is relevant. Implement count_triggers().
Check if the packet is relevant, its recorded spikes are still influenced by the latest shown image.
    If no threshold crosses are present in the packet, it's surely non relevant



