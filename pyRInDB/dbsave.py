import pandas as pd
import numpy as np
from main_class import RunIn_File
from multiprocessing import Pool
import functools
from tqdm import tqdm
import scipy
import os

path = r"X:\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"

def getStats_2d(data):
    """
    Calculate statistical measures for each 1D array in a 2D array.
    Parameters:
    - data (numpy.ndarray): The 2D array containing the data.
    Returns:
    - tuple: A tuple containing the following statistical measures for each 1D array in the input data:
        - rms (numpy.ndarray): Root mean square of each 1D array.
        - kur (numpy.ndarray): Kurtosis of each 1D array.
        - var (numpy.ndarray): Variance of each 1D array.
        - skew (numpy.ndarray): Skewness of each 1D array.
        - peak (numpy.ndarray): Maximum value of each 1D array.
        - crest (numpy.ndarray): Peak-to-RMS ratio of each 1D array.
    """

    # Check for NaN values
    if [np.isnan(row).any() for row in data].count(True) > 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    rms = []
    kur = []
    var = []
    skew = []
    peak = []
    crest = []
    for i in data:
        if len(i) == 0:
            rms.append(np.nan)
            kur.append(np.nan)
            var.append(np.nan)
            skew.append(np.nan)
            peak.append(np.nan)
            crest.append(np.nan)
            continue
        rms.append(np.sqrt(np.mean(i**2)))
        kur.append(scipy.stats.kurtosis(i))
        var.append(np.var(i)) 
        skew.append(scipy.stats.skew(i))
        peak.append(np.max(i))
        crest.append(np.max(i)/np.sqrt(np.mean(i**2)))
    return (rms, kur, var, skew, peak, crest)

def procTest(test: RunIn_File.RunIn_Unit_Reference.RunIn_Test_Reference):
    """
    Process the test data and return a dictionary of calculated statistics.
    Parameters:
    - test: RunIn_File.RunIn_Unit_Reference.RunIn_Test_Reference
        The test object containing the measurements data.
    Returns:
    - dict:
        A dictionary containing the following calculated statistics:
        - 'time': numpy.ndarray
            The time values rounded to the nearest integer.
        - 'runIn': numpy.ndarray
            An array indicating whether each time point is within the run-in period (0 = no, 1 = yes, -1 = unknown).
        - 'massFlow': numpy.ndarray
            The mass flow values.
        - 'vibrationRMSLateral': float
            The root mean square (RMS) value of the lateral vibration.
        - 'vibrationRMSLongitudinal': float
            The RMS value of the longitudinal vibration.
        - 'currentRMS': float
            The RMS value of the current.
        - 'vibrationKurtosisLateral': float
            The kurtosis value of the lateral vibration.
        - 'vibrationKurtosisLongitudinal': float
            The kurtosis value of the longitudinal vibration.
        - 'currentKurtosis': float
            The kurtosis value of the current.
        - 'vibrationVarianceLateral': float
            The variance value of the lateral vibration.
        - 'vibrationVarianceLongitudinal': float
            The variance value of the longitudinal vibration.
        - 'currentVariance': float
            The variance value of the current.
        - 'vibrationSkewnessLateral': float
            The skewness value of the lateral vibration.
        - 'vibrationSkewnessLongitudinal': float
            The skewness value of the longitudinal vibration.
        - 'currentSkewness': float
            The skewness value of the current.
        - 'vibrationPeakLateral': float
            The peak value of the lateral vibration.
        - 'vibrationPeakLongitudinal': float
            The peak value of the longitudinal vibration.
        - 'currentPeak': float
            The peak value of the current.
        - 'vibrationCrestLateral': float
            The crest factor of the lateral vibration.
        - 'vibrationCrestLongitudinal': float
            The crest factor of the longitudinal vibration.
        - 'currentCrest': float
            The crest factor of the current.
    """

    data = pd.DataFrame.from_dict(test.getMeasurements(varName=["time","vibrationRAWLateral","vibrationRAWLongitudinal","currentRAW","massFlow"], tStart=0,tEnd=float('inf')))
    vibLateralStats = getStats_2d(data["vibrationRAWLateral"].to_numpy())
    vibLongitudinalStats = getStats_2d(data["vibrationRAWLongitudinal"].to_numpy())
    currentStats = getStats_2d(data["currentRAW"].to_numpy())
    time = data["time"].to_numpy().round()

    if test.isRunIn:
        runin = np.ones(len(time)) * -1
        runin[time<=5*60] = 1
    else:
        runin = np.zeros(len(time))
    
    return {'time': time, 'runIn': runin, 'massFlow': data["massFlow"].to_numpy(),
            'vibrationRMSLateral': vibLateralStats[0], 'vibrationRMSLongitudinal': vibLongitudinalStats[0], 'currentRMS': currentStats[0],
            'vibrationKurtosisLateral': vibLateralStats[1], 'vibrationKurtosisLongitudinal': vibLongitudinalStats[1], 'currentKurtosis': currentStats[1],
            'vibrationVarianceLateral': vibLateralStats[2], 'vibrationVarianceLongitudinal': vibLongitudinalStats[2], 'currentVariance': currentStats[2],
            'vibrationSkewnessLateral': vibLateralStats[3], 'vibrationSkewnessLongitudinal': vibLongitudinalStats[3], 'currentSkewness': currentStats[3],
            'vibrationPeakLateral': vibLateralStats[4], 'vibrationPeakLongitudinal': vibLongitudinalStats[4], 'currentPeak': currentStats[4],
            'vibrationCrestLateral': vibLateralStats[5], 'vibrationCrestLongitudinal': vibLongitudinalStats[5], 'currentCrest': currentStats[5]}

if __name__ == '__main__':
    with RunIn_File(path) as file:
        df = pd.DataFrame()
        for unit in tqdm(file, total=len(file.units), desc = "Units", leave = False):
            for test in tqdm(unit,total = len(unit.tests), desc = "Tests", leave = False):
                
                test_res = procTest(test)
                test_res['test'] = [test.name]*len(test_res['time'])
                test_res['unit'] = [unit.name]*len(test_res['time'])
                df = pd.concat([df, pd.DataFrame.from_dict(test_res)], ignore_index=True)

    df.to_csv("allDataA.csv", index=False)
    df.to_feather("allDataA.feather")

            
            

            