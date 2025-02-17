
from pathlib import Path
import torch

# Get the directory of this file and move one level up to the repo root
current_dir = Path(__file__).resolve().parent
REPO_DIR = current_dir.parent
GP_REPO_DIR = REPO_DIR / 'gaussian_processes' / 'Spatial_GP_repo'

testmode = True
# testmode = False

# Name of a try for the whole experiment. 
# Its defined by the variables in this file
session_name = 'session_1'   

WINDOWS_OLD_MACHINE_IP   = '172.17.19.233'
LINUX_IP                 = "172.17.12.200"
PULL_SOCKET_PACKETS_PORT = '5555'
REQ_SOCKET_VEC_PORT      = '5557'
REQ_SOCKET_DMD_PORT      = '5558'

Win_side_path = '.\\src\\Win_side\\'

# Define the executables paths
DMD_EXE_PATH    = r"C:/Users/user/Repositories/cppalp/x64/Release/film.exe"
DMD_EXE_DIR     = r"C:/Users/user/Repositories/cppalp/x64/Release/"
ORT_READER_PATH = r"C:\Users\user\ClosedLoopProject\src\Win_side\ort_reader.py"

# Main parameters
experiment_data_path   = REPO_DIR / 'data'                                # Data folder that does not change
session_data_tcp_path  = REPO_DIR / 'src' / 'TCP' / 'data' / session_name # Data folder changing with sessions

img_dataset_path       = experiment_data_path / 'img_dataset' 
train_img_dataset_name = 'nat_img_train_dataset_PNAS_uint8.npy'
test_img_dataset_name  = 'nat_img_test_dataset_PNAS_uint8.npy'

electrode_info_path    = experiment_data_path  / 'electrode_info' / 'electrode_info.json'
vec_path               = session_data_tcp_path / 'vec_files' 
bin_path               = session_data_tcp_path / 'bin_files'


# Theaded functions parameters
timeout_vec         = 6 # seconds
timeout_dmd_off_rcv = 5

# DMD executable parameters
pietro_dir_DMD = "21"
bin_number     = "0"
vec_number     = "0"
frame_rate     = "30"
advanced_f     = "y"
n_frames_LUT   = "15"
raw_data_file_path = REPO_DIR / 'data' / 'raw_data.raw'

min_time_dmd_off = 3.5    # Min time to wait from the DMD off confirmation to be sure its really off
max_time_dmd_off = 7      # Maximum time from the confirmation of being off to be sure that it is not starting again ( The windows server is stopping )

# DMD triggers and image correspondence parameters
n_gray_trgs        = 60
n_img_trgs         = 60
n_ending_gray_trgs = 20


# Image dataset parameters
n_imgs_dataset   = 70  # How many images in the dataset
nat_img_px_nb    = 108

# MEA and DMD parameters
signal_file = "signal_file.txt"  # Signal file that ort_reader writes to allow DMD to start

# MEA parameters
buffer_size    = 1024
acq_freq       = 20000
trigger_freq   = 30
trg_threshold  = 40000
trg_diff_threshold = 2000
threshold_multiplier_init = 8 

# Acquisition channel ( chosen unit on the MEA )
ch_id = 53

# Initial hyperparameters
eps_0x_init  = 54    # px
eps_0y_init  = 54    # px
beta_init    = 15     # px
rho_init     = 15     # px
Amp_init     = 0.1
sigma_0_init = 0.1
# Initial link function parameters
A_init       = 0.01
lambda0_init = 1.


# CUDA parameters
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)


# Initial fit parameters
ntrain_init        = 50  
ntilde_init        = ntrain_init

ntilde_init        = 50
maxiter_init       = 100
nMstep_init        = 0   # Hyperparameters are not learnt
nEstep_init        = 20
nFparamstep_init   = 20
cellid_init        = None
n_px_side_init     = nat_img_px_nb
lr_Mstep_init      = 0
lr_Fparamstep_init = 0.1

# Dataset parameters
n_img_dataset = 3160



