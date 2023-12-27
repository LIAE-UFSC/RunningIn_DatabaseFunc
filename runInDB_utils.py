import h5py

class RunIn_File(h5py.File):
    # Class for running-in database in an hdf5 file
    def __init__(self, filePath, mode):
        self.path = filePath
        super().__init__(filePath, mode)
        self._modelh5ref = [self[key] for key in self.keys()][0]
        self.model = self._modelh5ref.name[1:]

    class RunIn_Unit_Reference:
        # Class for a unit group inside a hdf5 file
        def __init__(self, parent, unitGroupId:h5py.Group):
            self.h5ref = unitGroupId
            self.h5file = parent
            self.name = unitGroupId.name[1:]
            self.model = parent.model

        class RunIn_Test_Reference:
            # Class for a test group inside a hdf5 file
            def __init__(self, parent, testGroupId:h5py.Group):
                self.h5ref = testGroupId
                self.h5file = parent.h5file
                self.h5unit = parent
                self.date = testGroupId.name[1:]
                self.model = parent.model

            def getMeasurements(self, varName, tStart = 0, tEnd = None, index = None):
                pass

            def getVarNames(self):
                pass

        def tests(self):
            return [RunIn_Test_Reference(self,self[group]) for group in self.keys()]

    def units(self):
        return [RunIn_Unit_Reference(self,self[group]) for group in self.keys()]
