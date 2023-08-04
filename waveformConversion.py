import numpy as np
import struct

class Waveform:
    # Class for extracting data from binary LabVIEW waveform files
    # Example usage:
    #
    # For the Nth waveform in a file:
    #
    # wvf = Waveform.read_labview_waveform('path\to\file',N)
    #
    # For waveform arrays:
    # wvf = Waveform.read_labview_waveform('path\to\file',0)

    def __init__(self, data:np.ndarray, time:np.ndarray, dt:float = None):
        self.data = data
        self.time = time
        self.dt = dt

    @classmethod
    def read_labview_waveform(cls,file_path:str, N:int = 0):
        # Extracts the Nth waveform from file

        with open(file_path, 'rb') as file:
            file.seek(0x23A)

            num_arrays_bytes = file.read(4) # Read the number of waveforms in the array
            num_arrays = struct.unpack('>I', num_arrays_bytes)[0]

            # Skip the start time
            file.seek(file.tell()+16)

            for k1 in range(num_arrays):

                if k1 == N: # If is the Nth array
                    # Read the sampling time
                    time_scale_bytes = file.read(8)
                    time_scale = struct.unpack('>d', time_scale_bytes)[0]

                    # Read number of array elements in waveform
                    num_points_bytes = file.read(4)
                    num_points = struct.unpack('>I', num_points_bytes)[0]

                    # Initialize the data and time values of the waveform
                    data = np.zeros(num_points)
                    time = np.zeros(num_points)


                    for k2 in range(num_points):
                        data_bytes = file.read(8) # Read data value
                        data[k2] = struct.unpack('>d', data_bytes)[0] #Unpack datapoints
                        time[k2] = k2 * time_scale
                    
                    return cls(data,time,time_scale)
                    
                else:
                    file.seek(file.tell()+8) # Skip sampling time
                    
                    # Read number of array elements in waveform
                    num_points_bytes = file.read(4)
                    num_points = struct.unpack('>I', num_points_bytes)[0]

                    file.seek(file.tell()+8*num_points) # Skip whole array

                file.seek(file.tell()+21)

                for _ in range(struct.unpack('>I', num_points_bytes)[0]): # Number of waveform attributes
                    file.seek(file.tell()+54) # Ignore all waveform attributes

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