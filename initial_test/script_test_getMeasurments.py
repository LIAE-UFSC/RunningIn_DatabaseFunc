from source.runInDB_utils import *
import pandas as pd

pathIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"
# pathIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"

file =  RunIn_File(pathIn)

dictt = file.getMeasurements(indexes=range(5))

print(pd.DataFrame(dictt))