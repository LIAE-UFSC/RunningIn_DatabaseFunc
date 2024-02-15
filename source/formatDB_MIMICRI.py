import h5py
import tqdm
import numpy as np

def formatDB_MIMICRI(pathIn:str, pathOut:str, mode:str = 'w'):

    # Vars and database to copy
    desiredVars = ["massFlow"]
    desiredDatabase = ["vibrationRAWLateral", "vibrationRAWLongitudinal","acousticEmissionRAW", "voltageRAW", "currentRAW"]


    with h5py.File('pathOut','a') as f_dest:
        with h5py.File('pathIn','r') as f_src:
            modelGrpIn = list(f_src.values())[0] # Returns model subgroup
            maxminHeader = modelGrpIn["maxValues"].attrs["columnNames"]
            maxValues = modelGrpIn["maxValues"][:]
            minValues = modelGrpIn["minValues"][:]
            for unitGrpIn in tqdm.tqdm(list(modelGrpIn.values()), desc = "Unidade",  position=0):
                for testGrpIn in tqdm.tqdm(list(unitGrpIn.values()), desc = " Ensaio", leave=False,  position=1):

                    if testGrpIn.name in f_dest: 
                        if mode == "w": # Overwrites dataset if it already exists in target file
                            del f_dest[testGrpIn.name]
                        else: # Ignores dataset if it already exists in target file
                            continue
                        
                    testGrpOut = f_dest.create_group(testGrpIn.name) # Create test group in target file

                    if 'runningIn' in testGrpIn.attrs: # Get running-in state
                        testGrpOut.attrs["runningIn"] = testGrpIn.attrs["runningIn"]

                    headersIn = testGrpIn["measurements"].attrs['columnNames'] # Get header for timeseries measurements
                    
                    headersOut = ["time"] # Initialize target header with time
                    measurementOut = testGrpIn["measurements"][:,headersIn.index("time")]

                    for var in desiredVars:
                        if var in headersIn:
                            # Get column data
                            dataOriginal = testGrpIn["measurements"][:,headersIn.index(var)]

                            # Get max and min
                            maxVal = maxValues[maxminHeader.index(var)]
                            minVal = minValues[maxminHeader.index(var)]

                            # Multiply by a factor of 1/max
                            dataFit = dataOriginal/maxVal

                            measurementOut = np.c_[measurementOut, dataFit]
                            headersOut.append(var)

                    # Create timeseries measurement dataset in target file
                    dMeas = testGrpOut.create_dataset("measurements", data = dataFit, compression="gzip", shuffle=True)
                    dMeas.attrs['columnNames'] = headersOut
                            
                    for measurementIn in tqdm.tqdm(list(testGrpIn.values()), desc = "    Medicao", leave=False,  position=2):
                        
                        if isinstance(measurementIn, h5py.Dataset):
                            continue # Do nothing if is a dataset instead of a group

                        measurementOut = f_dest.create_group(measurementIn.name) # Create group for datasets in target file

                        for datasetVar in desiredDatabase:
                            if datasetVar in measurementIn:
                                # Get data
                                dataOriginal = measurementIn[datasetVar][()]

                                # Get max and min
                                maxVal = maxValues[maxminHeader.index(datasetVar)]
                                minVal = minValues[maxminHeader.index(datasetVar)]

                                # Multiply by a factor of 1/max
                                dataFit = dataOriginal/maxVal

                                dSet = measurementOut.create_dataset(datasetVar, data = dataFit, compression="gzip", shuffle=True)
                                dSet.attrs["dt"] = measurementIn[datasetVar].attrs["dt"]
                        

                        