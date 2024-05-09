import numpy as np
import ctypes as ct
import os
import struct

class Waveform:
    # Class for extracting data from binary LabVIEW waveform files
    # Example usage:
    #
    # For the Nth waveform in a file:
    #
    # wvf = Waveform.read_labview_waveform('path/to/file',N)
    #
    # For waveform arrays:
    # wvf = Waveform.read_labview_waveform('path/to/file')

    def __init__(self, data:np.ndarray, time:np.ndarray, dt:float = None):
        self.data = data
        self.time = time
        self.dt = dt

    @classmethod
    def read_labview_waveform(cls,file_path:str, N:int = 0):
        # Extracts the Nth waveform from file

        file_path = file_path.replace("/","\\") # Fix file path for windows

        if os.path.isfile(file_path):
            # Loads DLL with labview functions
            if (struct.calcsize("P")*8)==64:
                wvf_read_dll = ct.CDLL(os.path.dirname(__file__)+"/wvfRead64.dll")
            else:
                wvf_read_dll = ct.CDLL(os.path.dirname(__file__)+"/wvfRead.dll")

            path = bytes(file_path, 'utf-8')

            # Reads the length of the Nth waveform in the file
            wvf_read_dll.read_waveform_length.argtypes = [ct.c_char_p, ct.c_uint32]
            wvf_read_dll.read_waveform_length.restype = ct.c_int32
            length = wvf_read_dll.read_waveform_length(ct.c_char_p(path), ct.c_uint32(N))

            # Initialize an array of doubles of the addequate length
            data = (ct.c_double * length)(*range(length))
            time_scale = ct.c_double()

            # Reads the waveform data
            wvf_read_dll.read_waveform_data.argtypes = [ct.c_char_p,  ct.c_uint32, ct.POINTER(ct.c_double), ct.c_int32, ct.POINTER(ct.c_double)]
            wvf_read_dll.read_waveform_data.restype = None
            wvf_read_dll.read_waveform_data(ct.c_char_p(path), ct.c_uint32(N), data, length, ct.pointer(time_scale))
            time_scale = time_scale.value
            data = np.array(data[:])
            time = np.array([ind*time_scale for ind in range(length)])
            return cls(data,time,time_scale)
        else:
            raise Exception("File not found: " + file_path)

    @classmethod
    def read_array_labview_waveform(cls,file_path):
        # Extracts all waveforms from file

        file_path = file_path.replace("/","\\") # Fix file path for windows

        if os.path.isfile(file_path):
            # Loads DLL with labview functions
            if (struct.calcsize("P")*8)==64:
                wvf_read_dll = ct.CDLL(os.path.dirname(__file__)+"/wvfRead64.dll")
            else:
                wvf_read_dll = ct.CDLL(os.path.dirname(__file__)+"/wvfRead.dll")
                
            path = bytes(file_path, 'utf-8')

            # Reads the number of waveforms in the file
            wvf_read_dll.read_waveform_number.argtypes = [ct.c_char_p]
            wvf_read_dll.read_waveform_number.restype = ct.c_int32
            N = wvf_read_dll.read_waveform_number(ct.c_char_p(path))

            # Reads the length of each waveform
            length = (ct.c_int32 * N)(*range(N))
            wvf_read_dll.read_all_waveform_length.argtypes = [ct.c_char_p, ct.c_int32, ct.POINTER(ct.c_int32)]
            wvf_read_dll.read_all_waveform_length.restype = None
            wvf_read_dll.read_all_waveform_length(ct.c_char_p(path), N, length)
            length = length[:]

            # Reads data from all waveforms in file
            dt_all = (ct.c_double * N)(*range(N))
            data_all = (ct.c_double * np.sum(length))(*range(np.sum(length)))
            wvf_read_dll.read_all_waveform_data.argtypes = [ct.c_char_p, ct.c_int32, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32]
            wvf_read_dll.read_all_waveform_data.restype = None
            wvf_read_dll.read_all_waveform_data(ct.c_char_p(path), N, dt_all, data_all, ct.c_int32(np.sum(length)))

            ArrayWaveforms = []

            iStart = 0
            for ind in range(N):
                iEnd = iStart + length[ind] - 1

                time_scale = dt_all[ind]
                time = np.array([lind*time_scale for lind in range(length[ind])])
                data = np.array(data_all[iStart:iEnd])

                wvf = cls(data,time,time_scale)
                ArrayWaveforms.append(wvf)
                iStart = iEnd + 1

            return ArrayWaveforms
        else:
            raise Exception("File not found: " + file_path)