import h5py
import numpy as np
import warnings
import scipy.stats
import pandas as pd
from typing_extensions import Self
from .dbcreate import convertModel, convertFolders

class RunIn_File(h5py.File):
    # Class for running-in database in an hdf5 file
    def __init__(self, filePath, driver=None, libver=None, userblock_size=None, swmr=False,
                 rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None,
                 fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None,
                 page_buf_size=None, min_meta_keep=0, min_raw_keep=0, locking=None, **kwds):
        self.path = filePath
        self._fileh5ref = h5py.File(self.path,'r', driver, libver, userblock_size, swmr,
                 rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order,
                 fs_strategy, fs_persist, fs_threshold, fs_page_size,
                 page_buf_size, min_meta_keep, min_raw_keep, locking, **kwds)
        
        self._index = 0

        # Select model from file keys
        self._modelh5ref = [self._fileh5ref[key] for key in self._fileh5ref.keys() if isinstance(self._fileh5ref[key], h5py.Group)][0]
        
        self.model = self._modelh5ref.name[1:]

        self.units =  [self.RunIn_Unit_Reference(self,self._modelh5ref[group]) for group in self._modelh5ref.keys() if isinstance(self._modelh5ref[group], h5py.Group)]

    def __repr__(self):
        return f"Run-in model database <{self.model}> ({len(self.units)} units)"
    
    def __enter__(self):
        return self
    
    def open(self):
        return self
    
    def close(self):
        self.units = None
        self._modelh5ref = None
        self._fileh5ref.close()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type:
            print(f"Exception {exc_type} occurred with value {exc_val}")
        return True
    
    def __str__(self):
        return self.model
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index == len(self.units):
            self._index = 0
            raise StopIteration
        else:
            self._index += 1
            return self.units[self._index-1]
    
    def __getitem__(self, item):
        # Index by unit name
        if isinstance(item,str):
            for unit in self.units:
                if unit.name == item:
                    return unit
            return None
        
        # Index by number
        else:
            return self.units[item]
    
    @classmethod
    def convertModel(cls, foldersIn:list[str], fileOut:str, modelName:str, supressWarnings:bool = False) -> Self:
        convertModel(foldersIn, fileOut, modelName, supressWarnings)
        return cls(fileOut)

    @classmethod
    def convertFolders(cls, folderIn:str, folderOut:str, filePrefix = "Model", supressWarnings:bool = False) -> list[Self]:
        listFiles = convertFolders(folderIn, folderOut, filePrefix, supressWarnings)
        return [cls(file) for file in listFiles]

    def getTestDict(self):
        return {unit.name:unit.getTestNames() for unit in self.units}
    
    def getUnits(self):
        return {unit.name:unit for unit in self.units}
    
    def to_dict(self, testDict: dict = None, vars: list[str] = None, 
                    processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> pd.DataFrame:

        if testDict is None:
            testDict = self.getTestDict()

        data = []

        for (unit,tests) in testDict.items():
            # Add unit name to each dict list

            rows = self[unit].to_dict(testName = tests, vars = vars, processesdVars=processesdVars, tStart=tStart, tEnd=tEnd, indexes=indexes)
            data.extend(rows)

        return data
    
    def to_dataframe(self, testDict: dict = None, vars: list[str] = None, 
                     processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> list[dict[np.ndarray]]:
        
        if testDict is None:
            testDict = self.getTestDict()

        data = pd.DataFrame()

        for (unit,tests) in testDict.items():
            # Add unit name to each dict list

            unit_df = self[unit].to_dataframe(testName = tests, vars = vars, processesdVars=processesdVars, tStart=tStart, tEnd=tEnd, indexes=indexes)
            data = pd.concat([data,unit_df])

        return data

    class RunIn_Unit_Reference:
        """
        A class to handle a unit group inside an HDF5 file.

        Attributes
        ----------
        _h5ref : h5py.Group
            Reference to the HDF5 group.
        _h5file : h5py.File
            Reference to the HDF5 file.
        name : str
            Name of the unit group.
        model : str
            Model name from the parent object.
        tests : list[RunIn_Test_Reference]
            List of RunIn_Test_Reference objects representing the tests in the unit group.

        Methods
        -------
        getTestNames():
            Retrieves the names of all tests of the unit.
        to_dict(testName=None, vars=None, processesdVars=None, tStart=None, tEnd=None, indexes=None):
            Converts data from the unit tests into a dictionary format.
        to_dataframe(testName=None, vars=None, processesdVars=None, tStart=None, tEnd=None, indexes=None):
            Converts data from the unit tests into a pandas DataFrame.
        """
        
        def __init__(self, parent, unitGroupId:h5py.Group):
            self._h5ref = unitGroupId
            self._h5file = parent
            self.name = unitGroupId.name.split("/")[2]
            self.model = parent.model
            self._index = 0
            self.tests = [self.RunIn_Test_Reference(self,self._h5ref[group]) for group in self._h5ref.keys() if isinstance(self._h5ref[group], h5py.Group)]

        def __repr__(self):
            return self.name
        
        def __getitem__(self, item):
            # Index by test name
            if isinstance(item,str):
                for test in self.tests:
                    if test.name == item:
                        return test
                return None
            
            # Index by number
            else:
                return self.tests[item]
    
        def __str__(self):
            return self.name
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self._index == len(self.tests):
                self._index = 0
                raise StopIteration
            else:
                self._index += 1
                return self.tests[self._index-1]
            
        def getTestNames(self):
            """
            Retrieves the names of all tests of the unit.
            Returns:
                list: A list containing the names of all tests.
            """

            return [test.name for test in self.tests]
            
        def to_dict(self, testName: list[str] = None, vars: list[str] = None, 
                    processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> list[dict[np.ndarray]]:
            """
            Converts data from the unit tests into a dictionary format.

            Parameters
            ----------
            testName : list[str], optional
                List of test names to include in the dictionary. If None, all test names are included. Names not found are ignored.
            vars : list[str], optional
                List of variable names to include in the dictionary. If None, all variables are included.
            processesdVars : list[dict], optional
                List of dictionaries specifying variables and processes to apply.
                This only applies to high frequency data, which may be one of the following:
                    - current: Current of the electric motor of the compressor
                    - voltage: Voltage of the electric motor of the compressor
                    - acousticEmission: Acoustic emissions
                    - vibrationLongitudinal: Longitudinal vibration in the upper half of the compressor.
                    - vibrationRig: Rig vibration.
                    - vibrationLateral: Lateral vibration in the lower half of the compressor.
                The processing methods currently implemented are:
                    - RMS: Root mean square 
                    - Kurtosis: Kurtosis
                    - Variance: Variance
                    - Skewness: Skewness
                    - Peak: Peak value
                    - Crest: Crest factor
            tStart : float, optional
                Start time for data extraction, in seconds. If None, starts from 0 seconds, which is the instant at which the compressor was turned on.
            tEnd : float, optional
                End time for data extraction, in seconds. If None, ends at the last time point. Time is in seconds.
            indexes : list[int], optional
                List of indexes to include in the dictionary. If provided, tStart and tEnd are ignored.

            Returns
            -------
            dict[np.ndarray]
                Dictionary containing the test data.

            Example
            -------
            # Get the time; voltage; Kurtosis and RMS of the current; and variance of the lateral vibration from the first hour of testing for all tests in the unit:
            data = unit.to_dict(vars=["time","voltage"],processesdVars=[
                                    {"var":"current","process":["Kurtosis","RMS"]},
                                    {"var":"vibrationLateral","process":["Variance"]}],
                                    tEnd=3600)
            """

            if testName is None:
                testName = self.getTestNames()

            selTests = [test for test in self.tests if test.name in testName]

            data = []

            for test in selTests:
                # Add test name to each dict of list
                row = test.to_dict(vars = vars, processesdVars=processesdVars, tStart=tStart, tEnd=tEnd, indexes=indexes)
                row.update({"test":test.name, "unit":self.name})
                data.append(row)

            return data
        
        def to_dataframe(self, testName: list[str] = None, vars: list[str] = None, 
                         processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> list[dict[np.ndarray]]:
            """
            Converts data from the unit tests into a pandas DataFrame.

            Parameters
            ----------
            testName : list[str], optional
                List of test names to include in the DataFrame. If None, all test names are included. Names not found are ignored.
            vars : list[str], optional
                List of variable names to include in the DataFrame. If None, all variables are included.
            processesdVars : list[dict], optional
                List of dictionaries specifying variables and processes to apply.
                This only applies to high frequency data, which may be one of the following:
                    - current: Current of the electric motor of the compressor
                    - voltage: Voltage of the electric motor of the compressor
                    - acousticEmission: Acoustic emissions
                    - vibrationLongitudinal: Longitudinal vibration in the upper half of the compressor.
                    - vibrationRig: Rig vibration.
                    - vibrationLateral: Lateral vibration in the lower half of the compressor.
                The processing methods currently implemented are:
                    - RMS: Root mean square 
                    - Kurtosis: Kurtosis
                    - Variance: Variance
                    - Skewness: Skewness
                    - Peak: Peak value
                    - Crest: Crest factor
            tStart : float, optional
                Start time for data extraction, in seconds. If None, starts from 0 seconds, which is the instant at which the compressor was turned on.
            tEnd : float, optional
                End time for data extraction, in seconds. If None, ends at the last time point. Time is in seconds.
            indexes : list[int], optional
                List of indexes to include in the DataFrame. If provided, tStart and tEnd are ignored.

            Returns
            -------
            pd.DataFrame
                DataFrame containing the test group data.

            Example
            -------
            # Get the time; voltage; Kurtosis and RMS of the current; and variance of the lateral vibration from the first hour of testing for all tests in the unit:
            data = unit.to_dataframe(vars=["time","voltage"],processesdVars=[
                                        {"var":"current","process":["Kurtosis","RMS"]},
                                        {"var":"vibrationLateral","process":["Variance"]}],
                                        tEnd=3600)
            """
            
            if testName is None:
                testName = self.getTestNames()

            selTests = [test for test in self.tests if test.name in testName]

            data = pd.DataFrame()

            for test in selTests:
                test_df = test.to_dataframe(vars = vars, processesdVars=processesdVars, tStart=tStart, tEnd=tEnd, indexes=indexes)
                test_df["test"] = test.name
                test_df["unit"] = self.name
                data = pd.concat([data,test_df])

            return pd.DataFrame(data)

        class RunIn_Test_Reference:
            """
            A class to handle a test inside an HDF5 file.

            Attributes
            ----------
            _h5ref : h5py.Group
                Reference to the HDF5 group.
            _h5file : h5py.File
                Reference to the HDF5 file.
            h5unit : object
                Parent object containing the HDF5 unit.
            date : str
                Date extracted from the HDF5 group name.
            model : str
                Model name from the parent object.
            unit : str
                Unit name from the parent object.
            name : str
                Name of the test group, set to the date.

            Methods
            -------
            is_runIn():
                Checks if the test group is a run-in test.
            to_dict(vars=None, processesdVars=None, tStart=None, tEnd=None, indexes=None):
                Converts the test data to a dictionary.
            to_dataframe(vars=None, processesdVars=None, tStart=None, tEnd=None, indexes=None):
                Converts the test data to a pandas DataFrame.
            get_varNames():
                Returns the names of all measurement variables for the test group.
            """

            # Class for a test group inside a hdf5 file
            def __init__(self, parent, testGroupId:h5py.Group):
                """
                Initializes the instance with the provided parent and testGroupId.
                Args:
                    parent: The parent object containing the HDF5 file reference and model information.
                    testGroupId (h5py.Group): The HDF5 group representing the test group ID.
                Attributes:
                    _h5ref (h5py.Group): Reference to the HDF5 group representing the test group ID.
                    _h5file: Reference to the HDF5 file from the parent object.
                    h5unit: Reference to the parent object.
                    date (str): The date extracted from the test group ID's name.
                    model: The model information from the parent object.
                    unit: The compressor unit from the parent object.
                    name (str): The date extracted from the test group ID's name.
                """

                self._h5ref = testGroupId
                self._h5file = parent._h5file
                self.h5unit = parent
                self.date = testGroupId.name.split("/")[3]
                self.model = parent.model
                self.unit = parent.name
                self.name = self.date

            def is_runIn(self):
                """
                Check if the current test is a running-in test.
                Returns:
                    bool: True if the instance is marked as 'runIn', False otherwise.
                """

                return self._h5ref.attrs["runIn"]

            def __repr__(self):
                return str(self)
    
            def __str__(self):
                return self.name
            
            def _applyProcess(self, data: h5py.Dataset, processes: list[str], index: list[int], dbIndex: list[int]) -> dict[np.ndarray]:
                """
                Applies specified statistical processes to the rows of the dataset at given indices.

                Parameters:
                -----------
                data : h5py.Dataset
                    The dataset containing the data to be processed. It is assumed that the dataset has an attribute "index" which is a list of indices.
                processes : list of str
                    A list of processes to apply. Current valid processes are "RMS", "Kurtosis", "Variance", "Skewness", "Peak", and "Crest".
                index : list of int
                    A list of indices specifying which rows of the dataset to process.

                Returns:
                --------
                results : dict
                    A dictionary where keys are the process names and values are numpy arrays containing the results of the processes for each specified index.
                    If an index is not found in the dataset or the row contains only NaN values, the result for that index will be NaN.

                Raises:
                -------
                Exception
                    If an invalid process name is provided.
                """

                results = {process: np.empty((len(index))) for process in processes}

                for kNew,ind in enumerate(index):
                    # Check if the index is in the dataset
                    if ind not in dbIndex:
                        for process in processes:
                            results[process][kNew] = np.nan
                        continue
                    else:
                        row = data[dbIndex.index(ind),:]

                    # Check if the row is all NaN
                    if np.isnan(row).all():
                        results[kNew] = np.nan
                        continue
                    elif np.isnan(row).any():
                        # If the row has NaN values, remove them
                        row = row[~np.isnan(row)]
                    
                    for process in processes:
                        
                        if process == "RMS":
                            results[process][kNew] = np.sqrt(np.mean(np.square(row)))
                        elif process == "Kurtosis":
                            results[process][kNew] = scipy.stats.kurtosis(row)
                        elif process == "Variance":
                            results[process][kNew] = np.var(row)
                        elif process == "Skewness":
                            results[process][kNew] = scipy.stats.skew(row)
                        elif process == "Peak":
                            results[process][kNew] = np.max(row)
                        elif process == "Crest":
                            results[process][kNew] = np.max(row)/np.sqrt(np.mean(np.square(row)))
                        else:
                            raise Exception("Invalid process. Choose from 'RMS', 'Kurtosis', 'Variance', 'Skewness', 'Peak' or 'Crest'.")
                    
                return results
            
            def _procVars(self, processesdVars: list[dict], index: list[int]) -> dict[np.ndarray]:
                """
                Processes variables based on the provided processing instructions and indices.
                Args:
                    processesdVars (list[dict]): A list of dictionaries where each dictionary contains:
                        - "var" (str): The name of the variable to be processed.
                        - "process" (list): A list of processing methods to be applied to the variable.
                    index (list[int]): A list of indices to be used for processing.
                Returns:
                    dict[np.ndarray]: A dictionary where keys are the processed variable names and values are the processed data arrays.
                """

                data_processed = {}
                for select in processesdVars:
                    varName = select["var"]
                    processes = select["process"]
                
                    if varName in list(self._h5ref.keys()): # Check if the variable is in the database
                        if varName in ["vibrationLongitudinal", "vibrationRig", "vibrationLateral"]:
                            dbIndex = self._h5ref["index_vibration"][()].tolist()
                        else:
                            dbIndex = self._h5ref["index_"+varName][()].tolist()
                        proc = self._applyProcess(self._h5ref[varName], processes, index, dbIndex)
                        data_processed.update({varName+key:proc[key] for key in proc.keys()})
                    else: # If the variable is not in the database, fill with NaN
                        data_processed.update({varName+key:np.nan*np.ones(len(index)) for key in processes})
                    
                    # If the variable is "current" and the RMS is NaN, fill with the RMS from the "currentRMS" variable if available (special case when HF data was not recorded)
                    if (varName == "current") and ("RMS" in processes) and np.all(np.isnan(data_processed[varName + "RMS"])) and ("currentRMS" in self.get_varNames()):
                        measurementHeader = list(self._h5ref["measurements"].attrs["columnNames"])
                        data_processed.update({"currentRMS": self._h5ref["measurements"][index,measurementHeader.index("currentRMS")]})
                
                return data_processed

            def to_dict(self, vars: list[str] = None, processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> dict[np.ndarray]:
                """
                Converts the test group data to a dictionary.

                Parameters
                ----------
                vars : list[str], optional
                    List of variable names to include in the dictionary. If None, all variables are included.
                processesdVars : list[dict], optional
                    List of dictionaries specifying variables and processes to apply.
                    This only applies to high frequency data, which may be one of the following:
                        - current: Current of the electric motor of the compressor
                        - voltage: Voltage of the electric motor of the compressor
                        - acousticEmission: Acoustic emissions
                        - vibrationLongitudinal: Longitudinal vibration in the upper half of the compressor.
                        - vibrationRig: Rig vibration.
                        - vibrationLateral: Lateral vibration in the lower half of the compressor.
                    The processing methods currently implemented are:
                        - RMS: Root mean square 
                        - Kurtosis: Kurtosis
                        - Variance: Variance
                        - Skewness: Skewness
                        - Peak: Peak value
                        - Crest: Crest factor
                tStart : float, optional
                    Start time for data extraction, in seconds. If None, starts from 0 seconds, which is the instant at which the compressor was turned on.
                tEnd : float, optional
                    End time for data extraction, in seconds. If None, ends at the last time point. Time is in seconds.
                indexes : list[int], optional
                    List of indexes to include in the dictionary. If provided, tStart and tEnd are ignored.

                Returns
                -------
                dict[np.ndarray]
                    Dictionary containing the test group data.

                Example:
                --------
                # Get the time; voltage; Kurtosis and RMS of the current; and variance of the lateral vibration from the first hour of testing:
                data = test.to_dict(vars=["time","voltage"],processesdVars=[
                                            {"var":"current","process":["Kurtosis","RMS"]},
                                            {"var":"vibrationLateral","process":["Variance"]}],
                                            tEnd=3600)
                """
                
                if (indexes is not None) and ((tEnd is not None) or (tStart is not None)):
                    raise Exception("Both index and time range provided. Only one allowed.")
                
                allVars = self.get_varNames()
                if ("current" in allVars) and ("voltage" in allVars):
                    # If current and voltage are available, power can be calculated
                    allVars.append("power")

                if vars is None:
                    vars = allVars

                measurementHeader = list(self._h5ref["measurements"].attrs["columnNames"])

                # Check vars
                for var in vars:
                    if var not in allVars:
                        warnings.warn("One or more variables are not available for the selected test. Run get_varNames() to list all available variables.")

                data = {}

                if indexes is None:
                    if tStart is None:
                        tStart = 0
                    if tEnd is None:
                        tEnd = float("inf")

                    time = self._h5ref["measurements"][:,measurementHeader.index("time")]

                    indexes = np.where((time >= tStart) & (time <= tEnd))[0]
                    
                # Create a dictionary with the processed variables
                row = self._procVars(processesdVars, indexes)

                for var in vars:                            
                        
                    if var in ["voltage","acousticEmission", "current",
                            "vibrationLongitudinal", "vibrationRig", "vibrationLateral","power"]:
                        # Get values from high frequency dataset
                        if var in list(self._h5ref.keys()):
                            
                            if var in ["vibrationLongitudinal", "vibrationRig", "vibrationLateral"]:
                                dbIndex = self._h5ref["index_vibration"][()].tolist()
                            else:
                                dbIndex = self._h5ref["index_"+var][()].tolist()
                            indInDb = [dbIndex.index(ind) for ind in indexes if ind in dbIndex]
                            indNotInDb = [k for k,ind in enumerate(indexes) if ind not in dbIndex]

                            row[var] = self._h5ref[var][indInDb,:] # Get values from database

                            for ind in indNotInDb: # Insert nan values where indexes is not in dbIndex
                                row[var] = np.insert(row[var],ind,np.nan,axis=0)

                        elif (var == "power") and ("current" in list(self._h5ref.keys())) and ("voltage" in list(self._h5ref.keys())):
                            # Calculate power if current and voltage are available
                            dbIndex = self._h5ref["index_current"][()].tolist()
                            indInDb = [dbIndex.index(ind) for ind in indexes if ind in dbIndex]
                            indNotInDb = [k for k,ind in enumerate(indexes) if ind not in dbIndex]

                            row[var] = self._h5ref["current"][indInDb,:] * self._h5ref["voltage"][indInDb,:] # Get values from database

                            for ind in indNotInDb: # Insert nan values where indexes is not in dbIndex
                                row[var] = np.insert(row[var],ind,np.nan,axis=0)
                            

                        else:
                            row[var] = np.nan*np.ones((len(indexes),1))
                        
                    elif var in measurementHeader: # Get values from measurements dataset (low frequency)
                        row[var] = self._h5ref["measurements"][indexes,measurementHeader.index(var)]
                    else:
                        row[var] = np.nan*np.ones(len(indexes))

                for key in row.keys():
                    if key in data.keys():
                        data[key].append(row[key]) # Append to existing variable
                    else:
                        data[key] = [row[key]] # Create new variable
                    
                return data

            def to_dataframe(self, vars: list[str] = None, processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> pd.DataFrame:
                """
                Convert the test group data to a pandas DataFrame.

                Parameters
                ----------
                vars : list[str], optional
                    List of variable names to include in the DataFrame. If None, all variables are included.
                processesdVars : list[dict], optional
                    List of dictionaries specifying variables and processes to apply.
                    This only applies to high frequency data, which may be one of the following:
                        - current: Current of the electric motor of the compressor
                        - voltage: Voltage of the electric motor of the compressor
                        - acousticEmission: Acoustic emissions
                        - vibrationLongitudinal: Longitudinal vibration in the upper half of the compressor.
                        - vibrationRig: Rig vibration.
                        - vibrationLateral: Lateral vibration in the lower half of the compressor.
                    The processing methods currently implemented are:
                        - RMS: Root mean square 
                        - Kurtosis: Kurtosis
                        - Variance: Variance
                        - Skewness: Skewness
                        - Peak: Peak value
                        - Crest: Crest factor
                tStart : float, optional
                    Start time for data extraction, in seconds. If None, starts from 0 seconds, which is the instant at which the compressor was turned on.
                tEnd : float, optional
                    End time for data extraction, in seconds. If None, ends at the last time point. Time is in seconds.
                indexes : list[int], optional
                    List of indexes to include in the DataFrame. If provided, tStart and tEnd are ignored.

                Returns
                -------
                pd.DataFrame
                    DataFrame containing the test group data.

                Example
                -------
                # Get the time; voltage; Kurtosis and RMS of the current; and variance of the lateral vibration from the first hour of testing:
                data = test.to_dataframe(vars=["time","voltage"],processesdVars=[
                                            {"var":"current","process":["Kurtosis","RMS"]},
                                            {"var":"vibrationLateral","process":["Variance"]}],
                                            tEnd=3600)
                """
                data_dict = self.to_dict(vars, processesdVars, tStart, tEnd, indexes)
                
                return pd.DataFrame({k: v[0] if v[0].ndim == 1 else v[0].tolist() for k, v in data_dict.items()})

            def get_varNames(self) -> list[str]:
                '''
                Returns the name of all measurement variables in the test
                ''' 

                return list(self._h5ref["measurements"].attrs["columnNames"])+list(self._h5ref.keys())