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
    - [x] If no threshold crosses are present in the packet, it's surely non relevant
    - [x] If the threshold has been crossed at least once, the packet is relevant. Implement count_triggers().
    - The code exists for counting the triggers but:
        - It relies on *casting to int32* from int16, this is because it realies on the difference of the next voltage value with the current one. And sometimes this difference is too high.
        - It gets the first voltage above noise level that could be detected at sample frequency. Is it a limit?
    - [ ] Find a way to implement it without casting? Is int16 necessary? Thas is the type choosen by Pierre.
    - [x] Handle adge cases in which a trigger happens right at the end of a buffer
    - [x] Implement the counter for triggers in the gray. Images will be shown as #n of grey triggers, #n of image triggers #n of grey triggers
    - [x] Handle the case of absence of triggers in a buffer
    - [ ] Implement the counter for the 3 kinds of images that will be shown Checkerboard, Active choice, Random Choice
    - [ ] Implement anti aliasing control. Difference between triggers shouldnt be too different between each 

    Expected behaviour is counting only the " upward " spikes ( difference of more than 2000? , 5000? mV or whatever unit it is. So that I can call this counter function on a buffer. )
Check if the packet is relevant, its recorded spikes are still influenced by the latest shown image.
    If no threshold crosses are present in the packet, it's surely non relevant



