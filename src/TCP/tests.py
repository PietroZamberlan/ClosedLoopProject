import json
import numpy as np
import base64


import numpy as np

import threading
import time
import logging
def update_fit(tot_peaks_after_img):

    print(f"update_fit completed {tot_peaks_after_img}")
    logging.info("Thread %s: starting", tot_peaks_after_img)
    
def dump_on_file(nparray, filename):
    np.save(filename, nparray)
    logging.info(f"Dumped array to {filename}")

# Example usage in a threaded function
def threaded_dump(nparray, filename):
    dump_thread = threading.Thread(target=dump_on_file, args=(nparray, filename))
    dump_thread.start()


threaded_dump(np.array([1,2,3]), "./saved/test.npy")
