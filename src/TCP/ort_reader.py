import time
import os
import base64
import clr
import ctypes
import argparse

from System import *
clr.AddReference('System.Collections')
from System.Collections.Generic import List

import json, numpy

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


from clr_array_to_numpy import asNumpyArray

clr.AddReference(os.getcwd() + '\\..\\..\\McsUsbNet\\x64\\\McsUsbNet.dll')
from Mcs.Usb import CMcsUsbListNet
from Mcs.Usb import DeviceEnumNet

parser = argparse.ArgumentParser(
                    prog='Daemon to get peaks',
                    description='Record the MCS device to disk and send buffers via TCP')

parser.add_argument('filename')
parser.add_argument('-ip', default='127.0.0.1', dest='ip')
parser.add_argument('-p', '--port', default=1234, dest='port')
parser.add_argument('-b', '--buffer', default=1024, dest='buffer_size')
parser.add_argument('-s', '--sampling', default=20000, dest='sampling_rate')
parser.add_argument('-t', '--threshold', default=6, dest='threshold')
parser.add_argument('-n', '--num_buffers', default=10, dest='num_buffers')
parser.add_argument('-e', '--exclude_sweep_ms', default=2, dest='sweep')

peaks_only = True
do_filtering = True

args = parser.parse_args()
from Mcs.Usb import CMeaDeviceNet
from Mcs.Usb import McsBusTypeEnumNet
from Mcs.Usb import DataModeEnumNet
from Mcs.Usb import SampleSizeNet
from Mcs.Usb import SampleDstSizeNet

import numpy as np
import zmq
import scipy
context = zmq.Context()
socket = context.socket(zmq.STREAM)
socket.bind(f'tcp://{args.ip}:{args.port}')
device = CMeaDeviceNet(McsBusTypeEnumNet.MCS_USB_BUS);

### We create the file to dump the recording as a raw binary file
from pathlib import Path
data_filename = Path(args.filename)
if data_filename.exists():
    import os
    os.remove(data_filename)
data_file = open(args.filename, 'wb')

### Definition of the lowpass filter
filter_coeff = scipy.signal.iirfilter(
    3,
    200, 
    fs=args.sampling_rate, 
    analog=False, 
    btype='highpass', 
    ftype='butter', 
    output="ba"
)

def OnChannelData(x, cbHandle, numSamples):
    global nb_buffers
    if dataMode == DataModeEnumNet.Unsigned_16bit:
        data, frames_ret = device.ChannelBlock.ReadFramesUI16(0, 0, callbackThreshold, Int32(0));
        np_data = asNumpyArray(data, ctypes.c_uint16)
    else: # dataMode == DataModeEnumNet.Signed_32bit
        data, frames_ret = device.ChannelBlock.ReadFramesI32(0, 0, callbackThreshold, Int32(0));
        np_data = asNumpyArray(data, ctypes.c_int32)
    
    flatten = np_data.tobytes()
    print(f'OnChan: {nb_buffers}')
         
    
    num_channels = len(np_data) // frames_ret
    np_data = np_data.reshape(frames_ret, num_channels)
    
    if do_filtering:
        #We copy the data to float32
        np_data = np_data.astype(np.float32)
        np_data += float(np.iinfo('int16').min)
        b, a = filter_coeff
        filtered_traces = scipy.signal.lfilter(b, a, np_data, axis=0)
    else:
        filtered_traces = np_data
    
    if peaks_only:
       
        if nb_buffers < args.num_buffers:
            tstart = nb_buffers*args.buffer_size
            buffered_data[tstart:tstart+frames_ret] = filtered_traces
        elif nb_buffers == args.num_buffers:
            thresholds[:] = np.median(np.abs(buffered_data - np.median(buffered_data, 0)), 0)
            thresholds[:] *= args.threshold
        else:
            peaks = {}
            n_peaks = 0
            for i in range(num_channels):
                
                peaks[i] = scipy.signal.find_peaks(-filtered_traces[:, i], 
                                                   height=thresholds[i],
                                                   distance=exclude_sweep_ms)[0]
                n_peaks += len(peaks[i])
            print(f"Found {n_peaks} peaks during the buffer")
            packet = {'nb_buffers' : nb_buffers, 'peaks' : peaks}
            socket.send_string(json.dumps(packet, cls=Encoder))    
    else:
        packet = {'nb_buffers' : nb_buffers, 'data' : filtered_traces.flatten()}
        socket.send_string(json.dumps(packet, cls=Encoder)) 

    data_file.write(flatten)
    nb_buffers += 1
        

def OnError(msg, info):
    print(msg, info)

deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
DataModeToSampleSizeDict = {
    DataModeEnumNet.Unsigned_16bit : SampleSizeNet.SampleSize16Unsigned,
    DataModeEnumNet.Signed_32bit :  SampleSizeNet.SampleSize32Signed
}

DataModeToSampleDstSizeDict = {
    DataModeEnumNet.Unsigned_16bit : SampleDstSizeNet.SampleDstSize16,
    DataModeEnumNet.Signed_32bit :  SampleDstSizeNet.SampleDstSize32
}


print("found %d devices" % (deviceList.Count))

for i in range(deviceList.Count):
    listEntry = deviceList.GetUsbListEntry(i)
    print("Device: %s   Serial: %s" % (listEntry.DeviceName,listEntry.SerialNumber))

dataMode = DataModeEnumNet.Unsigned_16bit;
#dataMode = DataModeEnumNet.Signed_32bit;

device.ChannelDataEvent += OnChannelData
device.ErrorEvent += OnError

device.Connect(deviceList.GetUsbListEntry(0))

Samplingrate = args.sampling_rate;
device.SetSamplerate(Samplingrate, 1, 0);

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
callbackThreshold = args.buffer_size


if dataMode == DataModeEnumNet.Unsigned_16bit:
    mChannels = device.GetChannelsInBlock(0)
else: # dataMode == DataModeEnumNet.Signed_32bit
    mChannels = device.GetChannelsInBlock(0) // 2;
print("Number of Channels: ", mChannels)

global nb_buffers
nb_buffers = 0
thresholds = np.zeros(mChannels, dtype=np.float32)
exclude_sweep_ms = int(args.sampling_rate * args.sweep * 1e-3)
buffered_data = np.zeros((args.num_buffers*args.buffer_size, mChannels), dtype=np.uint16)


device.ChannelBlock.SetSelectedData(mChannels, callbackThreshold * 10, callbackThreshold, DataModeToSampleSizeDict[dataMode], DataModeToSampleDstSizeDict[dataMode], block)

device.StartDacq()
time.sleep(2)
device.StopDacq()
device.Disconnect()
