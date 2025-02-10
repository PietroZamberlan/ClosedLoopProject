import subprocess
import time
import os
import zmq
import threading
import queue
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir    = os.path.join(current_dir, '../../')
sys.path.insert(0, os.path.abspath(repo_dir))
Win_side_path = '.\\src\\Win_side\\' #

# Import the configuration file
from config.config import WINDOWS_OLD_MACHINE_IP, LINUX_IP, PULL_SOCKET_PORT, REQ_SOCKET_VEC_PORT, REQ_SOCKET_DMD_PORT
