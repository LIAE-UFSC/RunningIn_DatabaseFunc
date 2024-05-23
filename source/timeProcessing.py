"""
This module contains functions for processing time-based variables from a database file in HDF5 format.
"""

from runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import warnings

_LOAD_VARS = [
    # Variables that just need to be loaded
    "time",
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

_PROCESSED_VARS = [
    # Variables that need to be processed
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
    "power",  # Multiply voltage and current
]


def processVar(runInTest: RunIn_File.RunIn_Unit_Reference.RunIn_Test_Reference, varName, tStart, tEnd):
    """
    Process a specific variable based on the given RunIn_Test_Reference object.

    Args:
        runInTest (RunIn_File.RunIn_Unit_Reference.RunIn_Test_Reference): The RunIn_Test_Reference object.
        varName (str): The name of the variable to be processed.
        tStart (float): The starting time for processing.
        tEnd (float): The ending time for processing.

    Returns:
        list: The processed data for the variable.
    """
    data = []
    if varName[-3:] == "RMS":
        # RMS value
        varRaw = varName[:-3] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            data.append(np.sqrt(np.mean(row[0] ** 2)))

    elif varName[-3:] == "Kur":
        # Kurtosis value
        varRaw = varName[:-3] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            data.append(kurtosis(row[0], fisher=False))

    elif varName[-3:] == "THD":
        varRaw = varName[:-3] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)
        pass  # Not implemented

    elif varName[-3:] == "Var":
        # Signal variance
        varRaw = varName[:-3] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            data.append(np.var(row[0]))

    elif varName[-3:] == "Std":
        # Signal standard deviation
        varRaw = varName[:-3] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            data.append(np.sqrt(np.var(row[0])))

    elif varName[-4:] == "Skew":
        # Skewness — Asymmetry of a signal distribution
        varRaw = varName[:-4] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            data.append(skew(row[0]))

    elif varName[-4:] == "Peak":
        # Peak value
        varRaw = varName[:-4] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            data.append(np.max(row[0]))

    elif varName[-5:] == "Shape":
        # Shape factor — RMS divided by the mean of the absolute value
        varRaw = varName[:-5] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            rms = np.sqrt(np.mean(row[0] ** 2))
            data.append(rms / np.mean(abs(row[0])))

    elif varName[-5:] == "Crest":
        # Crest Factor — Peak value divided by the RMS
        varRaw = varName[:-5] + "RAW"
        dF = runInTest.getMeasurements(varName=varRaw, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            rms = np.sqrt(np.mean(row[0] ** 2))
            data.append(np.max(row[0]) / np.mean(abs(row[0])))

    elif varName == "Power":
        dF = runInTest.getMeasurements(varName=["currentRAW", "voltageRAW"], tEnd=tEnd, tStart=tStart, unknownIsNan=True)

        for _, row in dF.iterrows():
            current = row["currentRAW"]
            voltage = row["voltageRAW"]
            power = np.multiply(current, voltage)

            data.append(np.sqrt(np.mean(power ** 2)))

    else:
        raise Exception("Unknown processing mode")

    return data


def hdf2csv(pathHDF, target: dict = None, tStart=0, tEnd=float("inf"), vars=None, discardEnd=True, onlyAvailableVars=False):
    """
    Convert HDF5 file to CSV format.

    Args:
        pathHDF (str): The path to the HDF5 file.
        target (dict, optional): The target dictionary. Defaults to None.
        tStart (float, optional): The starting time for processing. Defaults to 0.
        tEnd (float, optional): The ending time for processing. Defaults to float("inf").
        vars (list, optional): The list of variables to be processed. Defaults to None.
        discardEnd (bool, optional): Whether to discard the end of the data. Defaults to True.
        onlyAvailableVars (bool, optional): Whether to select only available variables in the test. Defaults to False.
    """
    # Check time conditions:
    if tStart > tEnd:
        raise Exception("Starting time should be smaller than end time.")

    if vars is None:
        # Selects all possible variables if None is given
        vars = [varName[0] for varName in (_LOAD_VARS + _PROCESSED_VARS)]

    with RunIn_File(pathHDF) as file:
        if target is None:
            target = dict()
            for unit in file:
                target[str(unit)] = str(unit.tests)

        for unit, tests in target.items():
            for test in tests:

                testRef = file[unit][test]

                if onlyAvailableVars:
                    # Selects only vars available in test
                    testVars = [varName for varName in vars if (varName in testRef.getVarNames())]
                else:
                    # selects all vars
                    testVars = vars

                loadVars = [varName for varName in vars if (varName in _LOAD_VARS)]
                processVars = [varName for varName in vars if (varName in _PROCESSED_VARS)]
                varsDF = testRef.getMeasurements(varName=loadVars, tEnd=tEnd, tStart=tStart, unknownIsNan=True)

                for var in processVars:
                    processVar(test, var, tStart, tEnd)


if __name__ == "__main__":
    path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"
    hdf2csv(path)