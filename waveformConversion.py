import numpy as np
import struct
import ctypes as ct
import os

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

        # Reads the length of the Nth waveform in the file
        wvf_read_dll = ct.CDLL(os.path.dirname(__file__)+"/wvfRead.dll")
        path = bytes(file_path, 'utf-8')
        wvf_read_dll.read_waveform_length.argtypes = [ct.c_char_p, ct.c_uint32]
        wvf_read_dll.read_waveform_length.restype = ct.c_int32
        length = wvf_read_dll.read_waveform_length(ct.c_char_p(path), ct.c_uint32(N))

        # Initialize an array of doubles of the addequate length
        data = (ct.c_double * length)(*range(length))
        time_scale = ct.c_double()

        # Reads the waveform data
        wvf_read_dll.read_waveform_data.argtypes = [ct.c_char_p,  ct.c_uint32, ct.POINTER(ct.c_double), ct.c_int32, ct.POINTER(ct.c_double)]
        wvf_read_dll.read_waveform_data.restype = None
        wvf_read_dll.read_waveform_data(ct.c_char_p(path), ct.c_uint32(N), data, ct.pointer(time_scale), length)
        time_scale = time_scale.value
        data = np.array(data[:])
        time = np.array([ind*time_scale for ind in range(length.value)])
        return cls(data,time,time_scale)

    @classmethod
    def read_array_labview_waveform(cls,file_path):
        # Extracts all waveforms from file

        with open(file_path, 'rb') as file:
            file.seek(0x23A)

            num_arrays_bytes = file.read(4) # Read the number of waveforms in the array
            num_arrays = struct.unpack('>I', num_arrays_bytes)[0]

            # Skip the start time
            file.seek(file.tell()+16)

            ArrayWaveforms = []

            for _ in range(num_arrays):
                # Read the sampling time
                time_scale_bytes = file.read(8)
                time_scale = struct.unpack('>d', time_scale_bytes)[0]

                # Read number of array elements in waveform
                num_points_bytes = file.read(4)
                num_points = struct.unpack('>I', num_points_bytes)[0]

                # Initialize the data and time values of the waveform
                data = np.zeros(num_points)
                time = np.zeros(num_points)


                for k in range(num_points):
                    data_bytes = file.read(8) # Read data value
                    data[k] = struct.unpack('>d', data_bytes)[0] #Unpack datapoints
                    time[k] = k * time_scale
                
                wvf = cls(data,time,time_scale)
                ArrayWaveforms.append(wvf)

                file.seek(file.tell()+21)

                for _ in range(struct.unpack('>I', num_points_bytes)[0]): # Number of waveform attributes
                    file.seek(file.tell()+54) # Ignore all waveform attributes

        return ArrayWaveforms
    
a = Waveform.read_labview_waveform("D:/Dados - Thaler/Documentos/Amaciamento/Ensaios Brutos/Unidade C1/A_2022_10_10/vibracao/vib1.dat",1)

print(type(a))
