import h5py
import pandas as pd

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
    
    def getTestDict(self):
        return {unit.name:unit.getTestNames() for unit in self.units}
    
    def getUnits(self):
        return {unit.name:unit for unit in self.units}
    
    def getMeasurements(self, testDict: dict = None, varName: list[str] = None, 
                                tStart:float = None, tEnd:float = None, indexes: list[int] = None):
        # Returns a dataframe containing the measurements of the desired indexes or time range

        if testDict is None:
            testDict = self.getTestDict()

        data = []

        for (unit,tests) in testDict.items():
            # Add unit name to each dict list

            rows = [dict(item, unit=unit) for item in self[unit].getMeasurements(testName = tests, varName=varName, 
                                                                                 tStart = tStart, tEnd = tEnd, 
                                                                                 indexes = indexes)]
            data.extend(rows)

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
            
        def getMeasurements(self, testName: list[str] = None, varName: list[str] = None, 
                                     tStart:float = None, tEnd:float = None, indexes: list[int] = None):
            # Returns a dataframe containing the measurements of the desired indexes or time range

            if testName is None:
                testName = self.getTestNames()

            selTests = [test for test in self.tests if test.name in testName]

            data = []

            for test in selTests:
                # Add test name to each dict of list
                rows = [dict(item, test=test.name) for item in test.getMeasurements(varName, tStart, tEnd, indexes)]
                data.extend(rows)

            return data


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

            def __repr__(self):
                return str(self)
    
            def __str__(self):
                return self.name

            def getMeasurements(self, varName: list[str] = None, tStart:float = None, tEnd:float = None, indexes: list[int] = None):
                # Returns a dataframe containing the measurements of the desired indexes or time range

                if (indexes is not None) and ((tEnd is not None) or (tStart is not None)):
                    raise Exception("Both index and time range provided. Only one allowed.")
                else:
                    pass
                
                allVars = self.getVarNames()

                if varName is None:
                    varName = allVars

                measurementHeader = list(self._h5ref["measurements"].attrs["columnNames"])

                # Check vars
                for var in varName:
                    if var not in allVars:
                        raise Exception("One or more variables are not available for the selected test. Run getVarNames() to list all available variables.")  

                data = []

                if indexes is not None: # Get by index

                    for ind in indexes:
                        row = {}
                        for var in varName:
                            if var in ["voltageRAW","acousticEmissionRAW", "currentRAW",
                                    "vibrationRAWLongitudinal", "vibrationRAWRig", "vibrationRAWLateral"]:
                                # Get values from high frequency dataset
                                if var in list(self._h5ref[str(ind)].keys()):
                                    row[var] = self._h5ref[str(ind)][var][()]
                            else:
                                row[var] = self._h5ref["measurements"][ind,measurementHeader.index(var)]
                        data.append(row)
                    
                
                else: # Get by time
                    if tStart is None:
                        tStart = 0
                    if tEnd is None:
                        tEnd = float("inf")

                    if tStart > tEnd:
                        raise Exception("Starting time should be smaller than end time.")

                    allTests = list(self._h5ref.keys())
                    allTests.remove("measurements")

                    count = 0

                    t_act = self._h5ref["measurements"][count,measurementHeader.index("time")]

                    while t_act <= tEnd:

                        if t_act > tStart:
                            row = {}
                            for var in varName:
                                if var in ["voltageRAW","acousticEmissionRAW", "currentRAW",
                                        "vibrationRAWLongitudinal", "vibrationRAWRig", "vibrationRAWLateral"]:
                                    if var in list(self._h5ref[str(count)].keys()):
                                        # Get values from high frequency dataset
                                        row[var] = self._h5ref[str(count)][var][()]
                                else:
                                    row[var] = self._h5ref["measurements"][count,measurementHeader.index(var)]
                            data.append(row)
                        count = count + 1

                        if str(count) not in allTests: # All measurements have been checked
                            break
                        else:
                            t_act = self._h5ref["measurements"][count,measurementHeader.index("time")]
                    
                return data


            def getVarNames(self):
                # Return the name of all measurement variables of a given test

                # List of all available variables based on columnNames attribute and dataset names
                return list(self._h5ref["measurements"].attrs["columnNames"])+list(self._h5ref["0"].keys())
        
#if __name__ == "__main__":
#