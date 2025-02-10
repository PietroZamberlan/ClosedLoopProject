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

context = zmq.Context()

# Create the listening socket as a server
pull_socket_packets = context.socket(zmq.PULL)
pull_socket_packets.bind(f"tcp://*:{PULL_SOCKET_PORT}")

# Create a REQ socket as a client
req_socket_vec = context.socket(zmq.REQ)
req_socket_vec.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:{REQ_SOCKET_VEC_PORT}")

# Create a REQ socket as a client
req_socket_dmd = context.socket(zmq.REQ)
req_socket_dmd.connect(f"tcp://{WINDOWS_OLD_MACHINE_IP}:{REQ_SOCKET_DMD_PORT}")
