import h5py
import tqdm

def addMinMaxTOn_hdf(path:str):
    # Adds dictionary with min and max of each attribute to a database, as well as the time that each compressor is turned on

    with h5py.File(path, 'r+') as f:
        # Get units from model
        model = f[list(f.keys())[0]]

        # Initialize min and max dictionaries
        minValuesModel = {}
        maxValuesModel = {}

        for unit in tqdm.tqdm([model[un] for un in model.keys()], desc = "Unidade",  position=0):
            minValuesUnit = {}
            maxValuesUnit = {}
            for test in tqdm.tqdm([unit[tst] for tst in unit.keys()], desc = " Ensaio", leave=False,  position=1):
                tOn = float('inf')
                minValuesTest = {}
                maxValuesTest = {}
                for measurement in tqdm.tqdm([test[msm] for msm in test.keys()], desc = "    Medicao", leave=False,  position=2):

                    # Get first "compressor on" time
                    tOn = tOn if not measurement.attrs["compressorOn"] else min(tOn, measurement.attrs["time"])

                    for attr, value in measurement.attrs.items():
                        # Check min and max for every attribute
                        if attr in minValuesTest:
                            minValuesTest[attr] = min(minValuesTest[attr],value)
                            maxValuesTest[attr] = max(maxValuesTest[attr],value)
                        else:
                            minValuesTest[attr] = value
                            maxValuesTest[attr] = value

                        if attr in minValuesUnit:
                            minValuesUnit[attr] = min(minValuesUnit[attr],value)
                            maxValuesUnit[attr] = max(maxValuesUnit[attr],value)
                        else:
                            minValuesUnit[attr] = value
                            maxValuesUnit[attr] = value

                        if attr in minValuesModel:
                            minValuesModel[attr] = min(minValuesModel[attr],value)
                            maxValuesModel[attr] = max(maxValuesModel[attr],value)
                        else:
                            minValuesModel[attr] = value
                            maxValuesModel[attr] = value
                    
                    
                    for dataBase in [measurement[db] for db in measurement.keys()]:
                        # Check min and max for every dataBase
                        if str(dataBase) in minValuesTest:
                            minValuesTest[str(dataBase)] = min(minValuesTest[attr],min(dataBase[()]))
                            maxValuesTest[str(dataBase)] = max(maxValuesTest[attr],max(dataBase[()]))
                        else:
                            minValuesTest[str(dataBase)] = min(dataBase[()])
                            maxValuesTest[str(dataBase)] = max(dataBase[()])

                        if str(dataBase) in minValuesUnit:
                            minValuesUnit[str(dataBase)] = min(minValuesUnit[attr],min(dataBase[()]))
                            maxValuesUnit[str(dataBase)] = max(maxValuesUnit[attr],max(dataBase[()]))
                        else:
                            minValuesUnit[str(dataBase)] = min(dataBase[()])
                            maxValuesUnit[str(dataBase)] = max(dataBase[()])

                        if str(dataBase) in minValuesModel:
                            minValuesModel[str(dataBase)] = min(minValuesModel[attr],min(dataBase[()]))
                            maxValuesModel[str(dataBase)] = max(maxValuesModel[attr],max(dataBase[()]))
                        else:
                            minValuesModel[str(dataBase)] = min(dataBase[()])
                            maxValuesModel[str(dataBase)] = max(dataBase[()])

                # Add attributes to group
                test.attrs["tOn"] = tOn
                test.attrs["minValues"] = minValuesTest
                test.attrs["maxValues"] = maxValuesTest
            
            # Add attributes to unit
            unit.attrs["minValues"] = minValuesUnit
            unit.attrs["maxValues"] = maxValuesUnit
        
        # Add attributes to model
        model.attrs["minValues"] = minValuesModel
        model.attrs["maxValues"] = maxValuesModel