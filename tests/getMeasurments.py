import pyRInDB as ridb

pathIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"

file =  ridb.RunIn_File(pathIn)

df = file.to_dataframe(vars = ["time","massFlow","current"],
                       processesdVars = [
                            {"var":"current","process":["Kurtosis","RMS"]},
                            {"var":"vibrationLateral","process":["Variance"]}],
                        tStart = 0,
                        tEnd = 3600)

print(df)