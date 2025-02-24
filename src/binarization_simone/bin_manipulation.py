import numpy as np
import os

class BinFile:

    @classmethod
    def read_header(cls, path):

        header = {}
        with open(path, mode='rb') as input_file:
            # Read image xsize.
            image_xsize_bytes = input_file.read(2)
            header['xsize'] = int.from_bytes(image_xsize_bytes, byteorder='little')
            # Read image ysize.
            image_ysize_bytes = input_file.read(2)
            header['ysize'] = int.from_bytes(image_ysize_bytes, byteorder='little')
            # Read number of images.
            nb_images_bytes = input_file.read(2)
            header['nb_images'] = int.from_bytes(nb_images_bytes, byteorder='little')
            # Read number of bits.
            nb_bits_bytes = input_file.read(2)
            header['nb_bits'] = int.from_bytes(nb_bits_bytes, byteorder='little')

        return header

    @classmethod
    def read_nb_images(cls, path):

        header = cls.read_header(path)

        return header['nb_images']

    def __init__(self, path, frame_xsize, frame_ysize, nb_images=0, reverse=False, mode='r'):

        self._path = path
        self._reverse = reverse
        self._mode = mode

        if self._mode == 'r':
            header = self.read_header(self._path)
            self._nb_images = header['nb_images']
            self._frame_xsize = header['xsize']
            self._frame_ysize = header['ysize']
            self._nb_bits = header['nb_bits']
            self._file = open(self._path, mode='rb')
            self._frame_nb = self._nb_images - 1
        elif self._mode == 'w':
            self._nb_images = nb_images
            self._frame_xsize = frame_xsize
            self._frame_ysize = frame_ysize
            self._nb_bits = 8
            # self._file = open(self._path, mode='w+b')
            self._file = open(self._path, mode='wb')
            self._write_header()
            self._frame_nb = -1
        else:
            raise ValueError("unknown mode value: {}".format(self._mode))

        self._counter = 0

    def __len__(self):

        return self._nb_images

    def __iter__(self):

        self._counter = 0  # i.e. reinitialization

        return self

    def __next__(self):

        if self._counter < len(self):
            frame = self.read_frame(self._counter)
            self._counter += 1
        else:
            raise StopIteration

        return frame

    @property
    def _frame_shape(self):
        
        return self._frame_xsize, self._frame_ysize

    @property
    def ysize(self):
        
        return self._frame_ysize

    @property
    def xsize(self):
        return self._frame_xsize

    @property
    def nb_frames(self):
        
        return self._nb_images

    @property
    def nb_bits(self):
        
        return self._nb_bits

    def is_readable(self):
        
        return self._mode == 'r'

    def is_writeable(self):
        
        return self._mode == 'w'

    def get_frame_nb(self):
        """Get the number of the latest frame appended."""
        return self._frame_nb

    def get_frame_nbs(self):
        """Get the number of frames appended."""
        return np.arange(0, len(self))

    def read_frame_as_bytes(self, frame_nb):
        """Read frame as bytes."""
        assert self.is_readable(), "not readable"
        assert 0 <= frame_nb < len(self), frame_nb
        assert self._nb_bits == 8, self._nb_bits

        # Set file's current position.
        header_byte_size = 2 * 4
        frame_byte_size = self._frame_ysize * self._frame_xsize
        byte_offset = header_byte_size + frame_byte_size * frame_nb
        self._file.seek(byte_offset)
        # Read data from file.
        frame_bytes = self._file.read(frame_byte_size)

        return frame_bytes

    def read_frame(self, frame_nb):
        """Read, convert to float and reshape frame."""
        assert self.is_readable(), "not readable"
        assert 0 <= frame_nb < len(self), frame_nb
        assert self._nb_bits == 8, self._nb_bits

        # Set file's current position.
        header_byte_size = 2 * 4
        frame_byte_size = self._frame_ysize * self._frame_xsize
        byte_offset = header_byte_size + frame_byte_size * frame_nb
        self._file.seek(byte_offset)
        # Read data from file.
        frame_bytes = self._file.read(frame_byte_size)
        frame_data = np.fromstring(frame_bytes, dtype=np.uint8)
        # Convert data to float.
        frame_data = frame_data.astype(float)
        dinfo = np.iinfo(np.uint8)
        frame_data = frame_data / float(dinfo.max - dinfo.min + 1)
        # Reshape data.
        shape = (self._frame_xsize, self._frame_ysize)
        frame_data = np.reshape(frame_data, shape)

        return frame_data

    def _write_header(self):

        header_list = [
            self._frame_xsize,
            self._frame_ysize,
            self._nb_images,
            self._nb_bits,
        ]
        print("header_list: {}".format(header_list))
        header_array = np.array(header_list, dtype=np.int16)
        header_bytes = header_array.tobytes()
        self._file.write(header_bytes)

        return

    def append(self, frame):
        
        if isinstance(frame, bytes):

            assert len(frame) == self._frame_ysize * self._frame_xsize, len(frame)

            self._file.write(frame)

        else:

            assert frame.dtype == np.uint8, "frame.dtype: {}".format(frame.dtype)

            if self._reverse:
                frame = np.iinfo(np.uint8).max - frame
            frame_bytes = frame.tobytes()
            self._file.write(frame_bytes)

        self._frame_nb += 1

        return

    def flush(self):

        os.fsync(self._file.fileno())  # force write

        return

    def close(self):

        self.flush()
        self._file.close()

        return
