


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

# DMD executable parameters
pietro_dir_DMD = "21"
bin_number     = "0"
vec_number     = "0"
frame_rate     = "30"
advanced_f     = "y"
n_frames_LUT   = "15"
raw_data_file_path = "data/raw_data.raw"

min_time_dmd_off = 3.5    # Min time to wait from the DMD off confirmation to be sure its really off
max_time_dmd_off = 7      # Maximum time from the confirmation of being off to be sure that it is not starting again ( The windows server is stopping )

# DMD triggers and image correspondence parameters
max_gray_trgs    = 10
max_img_trgs     = 160
ending_gray_trgs = 20

n_imgs_dataset   = 70  # How many images in the dataset


# MEA and DMD parameters
signal_file = "signal_file.txt"  # Signal file that ort_reader writes to allow DMD to start

# MEA parameters
buffer_size    = "1024"
acq_freq       = 20000
trigger_freq   = 30
trg_threshold  = 40000

# Acquisition channel ( chosen unit on the MEA )
ch_id = 53