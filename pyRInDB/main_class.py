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
        # Class for a unit group inside a hdf5 file
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
            return [test.name for test in self.tests]
            
        def to_dict(self, testName: list[str] = None, vars: list[str] = None, 
                    processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> pd.DataFrame:
            # Returns a dataframe containing the measurements of the desired indexes or time range

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
            # Class for a test group inside a hdf5 file
            def __init__(self, parent, testGroupId:h5py.Group):
                self._h5ref = testGroupId
                self._h5file = parent._h5file
                self.h5unit = parent
                self.date = testGroupId.name.split("/")[3]
                self.model = parent.model
                self.unit = parent.name
                self.name = self.date

            def isRunIn(self):
                return self._h5ref.attrs["runIn"]

            def __repr__(self):
                return str(self)
    
            def __str__(self):
                return self.name
            
            def _applyProcess(self, data: h5py.Dataset, processes: list[str], index: list[int]):
                dbIndex = data.attrs["index"].tolist()
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

                data_processed = {}
                for select in processesdVars:
                    varName = select["var"]

                    processes = select["process"]
                    
                
                    if varName in list(self._h5ref.keys()):
                        proc = self._applyProcess(self._h5ref[varName], processes, index)
                        data_processed.update({varName+key:proc[key] for key in proc.keys()})
                    else:
                        data_processed.update({varName+key:np.nan*np.ones(len(index)) for key in processes})
                
                return data_processed

            def to_dict(self, vars: list[str] = None, processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> dict[np.ndarray]:
                
                if (indexes is not None) and ((tEnd is not None) or (tStart is not None)):
                    raise Exception("Both index and time range provided. Only one allowed.")
                
                allVars = self.getVarNames()

                if vars is None:
                    vars = allVars

                measurementHeader = list(self._h5ref["measurements"].attrs["columnNames"])

                # Check vars
                for var in vars:
                    if var not in allVars:
                        warnings.warn("One or more variables are not available for the selected test. Run getVarNames() to list all available variables.")

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
                            "vibrationLongitudinal", "vibrationRig", "vibrationLateral"]:
                        # Get values from high frequency dataset
                        if var in list(self._h5ref.keys()):
                            
                            dbIndex = self._h5ref[var].attrs["index"].tolist()
                            indInDb = [dbIndex.index(ind) for ind in indexes if ind in dbIndex]
                            indNotInDb = [k for k,ind in enumerate(indexes) if ind not in dbIndex]

                            row[var] = self._h5ref[var][indInDb,:] # Get values from database

                            for ind in indNotInDb: # Insert nan values where indexes is not in dbIndex
                                row[var] = np.insert(row[var],ind,np.nan,axis=0)

                        else:
                            row[var] = np.nan*np.ones(len(indexes),1)
                        
                    elif var in measurementHeader:
                        row[var] = self._h5ref["measurements"][indexes,measurementHeader.index(var)]
                    else:
                        row[var] = np.nan*np.ones(len(indexes))

                for key in row.keys():
                    if key in data.keys():
                        data[key].append(row[key])
                    else:
                        data[key] = [row[key]]
                    
                return data

            def to_dataframe(self, vars: list[str] = None, processesdVars: list[dict] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None) -> pd.DataFrame:
                data_dict = self.to_dict(vars, processesdVars, tStart, tEnd, indexes)
                
                return pd.DataFrame({k: v[0] if v[0].ndim == 1 else v[0].tolist() for k, v in data_dict.items()})

            def getVarNames(self) -> list[str]:
                # Return the name of all measurement variables of a given test

                # List of all available variables based on columnNames attribute and dataset names
                return list(self._h5ref["measurements"].attrs["columnNames"])+list(self._h5ref.keys())