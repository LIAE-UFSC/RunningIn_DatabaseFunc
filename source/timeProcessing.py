from runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import warnings

# Variables that just need to be loaded [databaseName,outName]
_LOAD_VARS = ["time",
            "temperatureSuction",
            "temperatureCompressorBody",
            "temperatureBuffer",
            "temperatureDischarge",
            "presSuction_Analog",
            "presDischarge",
            "presBuffer",
            "valveDischargePositionSetpoint",
            "valveDischargeOpening",
            "valveDischargeLimit",
            "valveSuctionVoltage",
            "compressorOn",
            "massFlow",
            "resistorDischargeDC",
            "presSuction_GEPreValve",
            "presSuction_GEPostValve",
            "resistorSuctionDC",
            "temperatureCoil1",
            "temperatureCoil2",
            "currentRMS_WT130",
            "voltageRMS_WT130",
            "PowerRMS_WT130",
            "currentRMS_TMCS",
            "voltageRMS_NI_DAQ",
            "temperatureRoom",
            "runningIn",
            ]

# Variables that need to be processed [databaseName,outName]
_PROCESSED_VARS = [
            # Extract features from high frequency measurements
            "currentRMS",
            "currentKur",
            "currentSkew",
            "currentShape",
            "currentCrest",
            "currentPeak",
            "currentVar",
            "currentStd",
            "vibRigRMS",
            "vibLongitudinalRMS",
            "vibLongitudinalKur",
            "vibLongitudinalSkew",
            "vibLongitudinalShape",
            "vibLongitudinalCrest",
            "vibLongitudinalPeak",
            "vibLongitudinalVar",
            "vibLongitudinalStd",
            "vibLateralRMS",
            "vibLateralKur",
            "vibLateralSkew",
            "vibLateralShape",
            "vibLateralCrest",
            "vibLateralPeak",
            "vibLateralVar",
            "vibLateralStd",
            "acousticEmissionsRMS",
            "acousticEmissionsKur",
            "acousticEmissionsSkew",
            "acousticEmissionsShape",
            "acousticEmissionsCrest",
            "acousticEmissionsPeak",
            "acousticEmissionsVar",
            "acousticEmissionsStd",

            "power", # Multiply voltage and current

            "presSuction", # Select based on available readings
            ]

def processVar(dF:pd.DataFrame,varName):
    data = []
    if varName[-3:] == "RMS":
        # RMS value
        for _, row in dF.iterrows():
            data.append(np.sqrt(np.mean(row[0]**2)))

    elif varName[-3:] == "Kur":
        # Kurtosis value
        for _, row in dF.iterrows():
            data.append(kurtosis(row[0],fisher = False))
                        
    elif varName[-3:] == "THD":
        pass # Not implemented

    elif varName[-3:] == "Var":
        # Signal variance
        for _, row in dF.iterrows():
            data.append(np.var(row[0]))

    elif varName[-3:] == "Std":
        # Signal standard deviation
        for _, row in dF.iterrows():
            data.append(np.sqrt(np.var(row[0])))

    elif varName[-4:] == "Skew":
        # Skewness — Asymmetry of a signal distribution
        for _, row in dF.iterrows():
            data.append(skew(row[0]))

    elif varName[-4:] == "Peak":
        # Peak value
        for _, row in dF.iterrows():
            data.append(np.max(row[0]))
            
    elif varName[-5:] == "Shape":
        # Shape factor — RMS divided by the mean of the absolute value
        for _, row in dF.iterrows():
            rms = np.sqrt(np.mean(row[0]**2))
            data.append(rms/np.mean(abs(row[0])))

    elif varName[-5:] == "Crest":
        # Crest Factor — Peak value divided by the RMS
        for _, row in dF.iterrows():
            rms = np.sqrt(np.mean(row[0]**2))
            data.append(np.max(row[0])/np.mean(abs(row[0])))

    else:
        raise Exception("Unknown processing mode")

def hdf2csv(pathHDF, target:dict = None, tStart = 0, tEnd = float("inf"), vars = None, discardEnd = True, onlyAvailableVars = False):
    
    # Check time conditions:
    if tStart > tEnd:
        raise Exception("Starting time should be smaller than end time.")
    
    if vars is None:
        # Selects all possible variables if None is given
        vars = [varName[0] for varName in (_LOAD_VARS+_PROCESSED_VARS)]

    with RunIn_File(pathHDF) as file:
        if target is None:
            target = dict()
            for unit in file:
                target[str(unit)] = str(unit.tests)

        for unit,tests in target.items():
            for test in tests:

                testRef = file[unit][test]

                if onlyAvailableVars:
                    # Selects only vars available in test
                    testVars = [varName for varName in vars if (varName in testRef.getVarNames())]
                else:
                    # selects all vars
                    testVars = vars
                
                loadVars = [varName for varName in vars if (varName in _LOAD_VARS)]
                processVars = [varName for varName in vars if (varName not in _LOAD_VARS)]
                varsDF = testRef.getMeasurements(varName=loadVars, tEnd=tEnd, tStart=tStart, unknownIsNan=True)
                
                for var in processVars:
                    


                

                

if __name__ == "__main__":
    path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"
    hdf2csv(path)
    
