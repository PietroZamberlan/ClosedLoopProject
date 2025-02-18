import time
import os
import base64
import clr
import ctypes
import argparse
import sys
import importlib.util
import json, numpy
import sys
import numpy as np
import zmq
import scipy

from win_utils import generate_packet
from config.config import *

from System import *
clr.AddReference('System.Collections')
from System.Collections.Generic import List

from clr_array_to_numpy import asNumpyArray # This is the original way to do it, but you need to sys.path.append('path_to_McsUsbNet_Examples')

# To imoprt Mcs
clr.AddReference('C:\\Users\\user\\ClosedLoopProject\\src\\Win_side\\McsUsbNet_Examples\\McsUsbNet\\x64\\\McsUsbNet.dll')
from Mcs.Usb import CMcsUsbListNet
from Mcs.Usb import DeviceEnumNet
from Mcs.Usb import CMeaDeviceNet
from Mcs.Usb import McsBusTypeEnumNet
from Mcs.Usb import DataModeEnumNet
from Mcs.Usb import SampleSizeNet
from Mcs.Usb import SampleDstSizeNet

from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir    = os.path.join(current_dir, '..\\..\\')
print(f' current dir from ort_reader: {current_dir}')
print(f' repo dir from ort_reader: {repo_dir}')

class NoDeviceFoundError(Exception):
    def __init__(self, message):
        super().__init__('\n ' + message)
        
class Encoder(json.JSONEncoder):

    def default(self, obj):
        """Encode numpy arrays.

        See also:
            https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
        """

        if isinstance(obj, numpy.ndarray):
            # Prepare data.
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = numpy.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            # Prepare serialized object.
            ser_obj = {
                '__ndarray__': base64.b64encode(obj_data),
                '__dtype__': str(obj.dtype),
                '__shape__': obj.shape,
            }
        elif isinstance(obj, numpy.int64):
            ser_obj = int(obj)
        elif isinstance(obj, bytes):
            ser_obj = {
                '__bytes__': obj.decode('utf-8'),
                '__encoding__': 'utf-8',
            }
        else:
            try:
                ser_obj = json.JSONEncoder.default(self, obj)
            except TypeError as error:
                print("{} ({})".format(obj, type(obj)))
                raise error

        return ser_obj

def OnChannelData(x, cbHandle, numSamples):
    
    global nb_buffers
            
    if dataMode == DataModeEnumNet.Unsigned_16bit:
        data, frames_ret = device.ChannelBlock.ReadFramesUI16(0, 0, buffer_size , Int32(0));
        np_data = asNumpyArray(data, ctypes.c_uint16)
    else: # dataMode == DataModeEnumNet.Signed_32bit
        data, frames_ret = device.ChannelBlock.ReadFramesI32(0, 0, buffer_size , Int32(0));
        np_data = asNumpyArray(data, ctypes.c_int32)
    
    flatten = np_data.tobytes()
    print(f'OnChan: {nb_buffers}')
         
    num_channels = len(np_data) // frames_ret
    np_data = np_data.reshape(frames_ret, num_channels)
    
    '''Wether or not we do filtering and peak detection, the raw data for the trigger channel will be sent as well
       in a separate key in the packet dictionary'''
    trg_raw_data = np_data[:,trg_ch_id] 
    if do_filtering:
        #We copy the data to float32
        np_data = np_data.astype(np.float32)
        np_data += float(np.iinfo('int16').min)
        b, a = filter_coeff
        filtered_traces = scipy.signal.lfilter(b, a, np_data, axis=0)
    else:
        filtered_traces = np_data
    
    if peaks_only:
       
        if nb_buffers < num_init_buffers: # Initial buffers are held together to allow the filtering to work, they are not sent
            tstart = nb_buffers*buffer_size
            buffered_data[tstart:tstart+frames_ret] = filtered_traces
        elif nb_buffers == num_init_buffers:
            thresholds[:] = np.median(np.abs(buffered_data - np.median(buffered_data, 0)), 0)
            thresholds[:] *= threshold_multiplier
        else:
            peaks = {}
            n_peaks = 0
            for i in range(num_channels):
    
                peaks[i] = scipy.signal.find_peaks(-filtered_traces[:, i], 
                                                   height=thresholds[i],
                                                   distance=exclude_sweep_ms)[0]
                n_peaks += len(peaks[i])
            print(f"Found {n_peaks} peaks during the buffer")
            packet = {'buffer_nb' : nb_buffers, 'peaks' : peaks, 'n_peaks' : n_peaks, 'trg_raw_data' : trg_raw_data, 'send_time': time.time()}
            socket.send_string(json.dumps(packet, cls=Encoder))    
    else:
        # Ugly logic to not break things. Of course one could calculate peaks before and use continue keywords
        if nb_buffers < num_init_buffers: # Initial buffers are held together to allow the filtering to work, they are not sent
            tstart = nb_buffers*buffer_size
            buffered_data[tstart:tstart+frames_ret] = filtered_traces
        elif nb_buffers == num_init_buffers:
            thresholds[:] = np.median(np.abs(buffered_data - np.median(buffered_data, 0)), 0)
            thresholds[:] *= threshold_multiplier
        else:
            peaks = {}
            n_peaks = 0
            for i in range(num_channels):
    
                peaks[i] = scipy.signal.find_peaks(-filtered_traces[:, i], 
                                                   height=thresholds[i],
                                                   distance=exclude_sweep_ms)[0]
                n_peaks += len(peaks[i])
            print(f"Found {n_peaks} peaks during the buffer")
            # filtered_traces was .flatten() when sent. this might improve performance
            packet = {'buffer_nb' : nb_buffers, 'data' : filtered_traces, 'trg_raw_data' : trg_raw_data, 'peaks' : peaks, 'n_peaks' : n_peaks, 'send_time': time.time()}
            socket.send_string(json.dumps(packet, cls=Encoder)) 

    data_file.write(flatten)
    nb_buffers += 1
        
def OnError(msg, info):
    print(msg, info)

parser = argparse.ArgumentParser(
                    prog='Daemon to get peaks',
                    description='Record the MCS device to disk and send buffers via TCP')

# parser.add_argument('filename')
parser.add_argument('--filename', required=True, 
                    help='Path to the dump file for raw data')
parser.add_argument('-ip', default='127.0.0.1', dest='ip', 
                    help='IP address of the linux machine to send the data to')
parser.add_argument('-p', '--port', default=1234, dest='port', type=int,
                    help='Port number to send the data to')
parser.add_argument('-b', '--buffer_size', default=1024, dest='buffer_size', type=int,
                    help='Size of the buffer to send')
parser.add_argument('-s', '--sampling', default=20000, dest='sampling_rate', type=int,
                    help='Sampling rate of the MEA')
parser.add_argument('-t', '--threshold_multiplier', default=26, dest='threshold_multiplier', type=int, 
                    help='Multiplier of the median of the frist num_init_buffers to determine the threshold for peaks')
parser.add_argument('-n', '--num_init_buffers', default=10, dest='num_init_buffers', type=int,
                    help='Number of buffers to use for the initial threshold calculation. They are not sent.')
parser.add_argument('-e', '--exclude_sweep_ms', default=2, dest='sweep', type=int,
                    help='Time in ms to exclude peaks from each other')
parser.add_argument('--test', '-T', action='store_true', dest='testmode', 
                    help='Set test mode to true - no connection attempted to MEA')

peaks_only = False
do_filtering = False

args              = parser.parse_args()
testmodeMEA       = args.testmode
samplingrate      = args.sampling_rate
buffer_size       = args.buffer_size
num_init_buffers  = args.num_init_buffers
threshold_multiplier = args.threshold_multiplier
sweep             = args.sweep
linux_ip          = args.ip
linux_port        = args.port
rawdata_filename  = args.filename
rawdata_path      = Path(rawdata_filename)

context = zmq.Context()
#socket = context.socket(zmq.STREAM)
#socket.bind(f'tcp://{args.ip}:{args.port}')
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://{linux_ip}:{linux_port}")


### We create the file to dump the recording as a raw binary file
if rawdata_path.exists():
    os.remove(rawdata_path)
data_file = open(rawdata_filename, 'wb')

### Definition of the lowpass filter
filter_coeff = scipy.signal.iirfilter(
    3,
    200, 
    fs=samplingrate, 
    analog=False, 
    btype='highpass', 
    ftype='butter', 
    output="ba"
)

if not testmodeMEA:
    device = CMeaDeviceNet(McsBusTypeEnumNet.MCS_USB_BUS)

    deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
    DataModeToSampleSizeDict = {
        DataModeEnumNet.Unsigned_16bit : SampleSizeNet.SampleSize16Unsigned,
        DataModeEnumNet.Signed_32bit :  SampleSizeNet.SampleSize32Signed
    }

    DataModeToSampleDstSizeDict = {
        DataModeEnumNet.Unsigned_16bit : SampleDstSizeNet.SampleDstSize16,
        DataModeEnumNet.Signed_32bit :  SampleDstSizeNet.SampleDstSize32
    }

    if deviceList.Count == 0:
        # raise NoDeviceFoundError("No device is connected, or it's turned off")
        print("No device is connected, or it's turned off", file=sys.stderr)
        sys.exit(1)
    print("found %d devices" % (deviceList.Count))

    for i in range(deviceList.Count):
        listEntry = deviceList.GetUsbListEntry(i)
        print("Device: %s   Serial: %s" % (listEntry.DeviceName,listEntry.SerialNumber))

    dataMode = DataModeEnumNet.Unsigned_16bit;
    #dataMode = DataModeEnumNet.Signed_32bit;

    device.ChannelDataEvent += OnChannelData
    device.ErrorEvent += OnError

    device.Connect(deviceList.GetUsbListEntry(0))

    device.SetSamplerate(samplingrate, 1, 0);

    miliGain = device.GetGain();
    voltageRanges = device.HWInfo.GetAvailableVoltageRangesInMicroVoltAndStringsInMilliVolt(miliGain);
    for i in range(0, len(voltageRanges)):
        print("(" + str(i) + ") " + voltageRanges[i].VoltageRangeDisplayStringMilliVolt);

    device.SetVoltageRangeByIndex(0, 0);
    device.SetDataMode(dataMode, 0)
    #device.SetNumberOfChannels(256)
    device.EnableDigitalIn(Boolean(False), 0)
    device.EnableChecksum(False, 0)

    block = device.GetChannelsInBlock(0);

    print("Channels in Block: ", block)

    if dataMode == DataModeEnumNet.Unsigned_16bit:
        mChannels = device.GetChannelsInBlock(0)
    else: # dataMode == DataModeEnumNet.Signed_32bit
        mChannels = device.GetChannelsInBlock(0) // 2;
    print("Number of Channels: ", mChannels)

    global nb_buffers
    nb_buffers = 0
    trg_ch_id  = 126
    thresholds = np.zeros(mChannels, dtype=np.float32)
    exclude_sweep_ms = int(samplingrate * sweep * 1e-3)
    # This gets updated with the first buffer_size buffers acquired from the MEA so that the scpipy filtering can work
    buffered_data = np.zeros((num_init_buffers*buffer_size, mChannels), dtype=np.uint16)


    device.ChannelBlock.SetSelectedData(  
        mChannels  , buffer_size * 10, buffer_size , DataModeToSampleSizeDict[dataMode], DataModeToSampleDstSizeDict[dataMode], block)

else:
    print("Acquisition launched in TEST mode - no connection to MEA")

    # test_packet = np.load('test_packet.npy', allow_pickle=True).item()

try:
    with open("signal_file.txt", "w") as f:
        f.write("Signal to launch the executable")
        print('Signal file to launch DMD written')
    
    if not testmodeMEA:
        device.StartDacq()
        # time.sleep(20)
        while True:
            time.sleep(1)
            # print('Still alive...')
    if testmodeMEA:
        # Send simulated packets every 0.1s
        buffer_nb = starting_buffer_nb
        while True:
            time.sleep(0.11)
            packet = generate_packet(buffer_nb)
            # buffer_nb += 1
            socket.send_string(json.dumps(packet, cls=Encoder))    
            if buffer_nb == ending_buffer_nb:
                break

except KeyboardInterrupt:
    pass
    
finally:
    print("Server is shutting down...")
    print('Stopping acquisition...')
    device.StopDacq()
    print('Disconnecting MEA...')
    device.Disconnect()
    print('Closing push socket...')
    socket.setsockopt(zmq.LINGER, 0)
    socket.close()
    print('Closing context...')
    context.term()






























