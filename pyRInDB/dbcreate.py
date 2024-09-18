import os
import re
import csv
import h5py
import warnings
import tqdm
import numpy as np
import pandas as pd
from pyRInDB.utils import Waveform

def addMinOrMax(dictMin, dictMax, name, value):
    # Compare and add to min max dict

    if isinstance(value, np.ndarray):
        if name in dictMin:
            dictMin[name] = min(dictMin[name],value.min())
            dictMax[name] = max(dictMax[name],value.max())
        else:
            dictMin[name] = value.min()
            dictMax[name] = value.max()
    else:
        if name in dictMin:
            dictMin[name] = min(dictMin[name],value)
            dictMax[name] = max(dictMax[name],value)
        else:
            dictMin[name] = value
            dictMax[name] = value

def joinMinMaxDict(dictMain: dict, dictAdd:dict, mode = 'min'):
    # Merge 2 min or max dict into one (dictMain)
    for key in dictAdd:
        if key in dictMain:
            if mode == 'max':
                dictMain[key] = max(dictMain[key],dictAdd[key])
            if mode == 'min':
                dictMain[key] = min(dictMain[key],dictAdd[key])
        else:
            dictMain[key] = dictAdd[key]

def textfile2dict(path: str):
    d = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split(sep = ":")
            d[key] = val.strip()
        return d

def nameVar(headerName:str) -> str:
    # Convert column name from "medicoesGerais.dat" file to hdf5 attribute name

    varNames = [["time","Tempo [s]"],
            ["temperatureSuction","Temperatura Sucção [ºC]"],
            ["temperatureCompressorBody","Temperatura Corpo [ºC]"],
            ["temperatureBuffer","Temperatura Reservatório [ºC]"],
            ["temperatureDischarge","Temperatura Descarga [ºC]"],
            ["presSuction_Analog","Pressão Sucção [bar]"],
            ["presDischarge","Pressão Descarga [bar]"],
            ["presBuffer","Pressão Intermediária [bar]"],
            ["valveDischargePositionSetpoint","Setpoint da Abertura da Válvula [Passos]"],
            ["valveDischargeOpening","Abertura da válvula [Passos]"],
            ["valveDischargeLimit","Limite da válvula [Passos]"],
            ["valveSuctionVoltage","Tensão da válvula de sucção [V]"],
            ["currentRMS","Corrente RMS [A]"],
            ["compressorOn","Compressor Ativado"],
            ["vibLateralRMS","Aceleração RMS Inferior Corpo [g]"],
            ["vibRigRMS","Aceleração RMS Bancada [g]"],
            ["vibLongitudinalRMS","Aceleração RMS Superior Corpo [g]"],
            ["massFlow","Vazão Mássica [kg/h]"],
            ["resistorDischargeDC","Duty Cycle Descarga [%]"],
            ["presSuction_GEPreValve","Pressão Sucção Válvula (DPS) [bar]"],
            ["presSuction_GEPostValve","Pressão Sucção Compressor (DPS) [bar]"],
            ["resistorDischargeDC","Duty Cycle [%]"],
            ["temperatureCoil1","Temperatura Bobina 1 [ºC]"],
            ["temperatureCoil2","Temperatura Bobina 2 [ºC]"],
            ["currentRMS_WT130","Corrente Yokogawa [A]"],
            ["voltageRMS_WT130","Tensão Yokogawa [V]"],
            ["PowerRMS_WT130","Potência Yokogawa [W]"],
            ["currentRMS_TMCS","Corrente TMCS [A]"],
            ["voltageRMS_NI_DAQ","Tensão DAQ [V]"],
            ["temperatureRoom","Temperatura Ambiente [ºC]"]]
    
    index = [x[1] for x in varNames].index(headerName)
    return varNames[index][0]

def create_HF_dataset(testGrp:h5py.Group, folderIn: str, var:str):
    
    if var == "vibration":
        folder = f"{folderIn}\\vibracao"
        pattern = r"vib\d+\.dat"
    elif var == "acousticEmission":
        folder = f"{folderIn}\\acusticas"
        pattern = r"acu\d+\.dat"
    elif var == "voltage":
        folder = f"{folderIn}\\tensao"
        pattern = r"ten\d+\.dat"
    elif var == "current":
        folder = f"{folderIn}\\corrente"
        pattern = r"corr\d+\.dat"
    else:
        raise ValueError("Invalid variable name")
        
    files = os.listdir(folder)
    matching_files = [file for file in files if re.match(pattern, file)]

    indexes = [int(re.search(r'\d+', file).group()) for file in matching_files]
    matching_files = [file for _, file in sorted(zip(indexes, matching_files))]

    indexes.sort()
    
    if len(indexes) == 0:
        return

    if var == "vibration":
        dSetLong = testGrp.create_dataset("vibrationLongitudinal", (len(indexes),25600),compression="gzip", shuffle=True)
        dSetLat = testGrp.create_dataset("vibrationLateral", (len(indexes),25600), compression="gzip", shuffle=True)
        dSetRig = testGrp.create_dataset("vibrationRig", (len(indexes),25600), compression="gzip", shuffle=True)

        dSetLong.attrs["index"] = indexes
        dSetLat.attrs["index"] = indexes
        dSetRig.attrs["index"] = indexes

        for indexMeas, file in enumerate(tqdm.tqdm(matching_files, desc = f"    {var}", position = 3, leave = False)):
            filePath = f"{folder}\\{file}"

            wvf = Waveform.read_array_labview_waveform(filePath)

            if len(wvf[0].data) == 0:
                continue
            elif len(wvf[0].data) > 25600:
                wvf[0].data = wvf[0].data[:25600]
                wvf[1].data = wvf[1].data[:25600]
                wvf[2].data = wvf[2].data[:25600]
            elif len(wvf[0].data) < 25600:
                wvf[0].data = np.pad(wvf[0].data, (0,25600-len(wvf[0].data)), constant_values = np.nan)
                wvf[1].data = np.pad(wvf[1].data, (0,25600-len(wvf[1].data)), constant_values = np.nan)
                wvf[2].data = np.pad(wvf[2].data, (0,25600-len(wvf[2].data)), constant_values = np.nan)

            dSetLat.attrs["dt"] = wvf[0].dt
            dSetLat[indexMeas,:] = wvf[0].data

            dSetRig.attrs["dt"] = wvf[1].dt
            dSetRig[indexMeas,:] = wvf[1].data

            dSetLong.attrs["dt"] = wvf[2].dt
            dSetLong[indexMeas,:] = wvf[2].data

            if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                testGrp.attrs['startTime'] = os.path.getmtime(filePath)

        return
    
    else:
        if var == "acousticEmission":
            dSet = testGrp.create_dataset(var, (len(indexes),300000),compression="gzip", shuffle=True)
        else:
            dSet = testGrp.create_dataset(var, (len(indexes),25600),compression="gzip", shuffle=True)

        dSet.attrs["index"] = indexes

        for indexMeas, file in enumerate(tqdm.tqdm(matching_files, desc = f"    {var}", position = 3, leave = False)):
            filePath = f"{folder}\\{file}"

            wvf = Waveform.read_labview_waveform(filePath,0)

            if len(wvf.data) == 0:
                continue
            
            if var == "acousticEmission":
                if len(wvf.data) > 300000:
                    wvf.data = wvf.data[:300000]
                elif len(wvf.data) < 300000:
                    wvf.data = np.pad(wvf.data, (0,300000-len(wvf.data)), constant_values = np.nan)
            else:
                if len(wvf.data) > 25600:
                    wvf.data = wvf.data[:25600]
                elif len(wvf.data) < 25600:
                    wvf.data = np.pad(wvf.data, (0,25600-len(wvf.data)), constant_values = np.nan)

            dSet.attrs["dt"] = wvf.dt
            dSet[indexMeas,:] = wvf.data

            if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                testGrp.attrs['startTime'] = os.path.getmtime(filePath)

        return

def convertModel(UnitFoldersIn:list[str], fileOut:str, modelName:str, supressWarnings = False):

    if supressWarnings:
        warnings.filterwarnings('ignore')

    unitFolders = UnitFoldersIn

    # Dict for max and min values of a given unit
    minValuesModel = {}
    maxValuesModel = {}

    with h5py.File(fileOut, "a") as fModel:

        if not f"Model{modelName}" in fModel:
            modelGrp = fModel.create_group(f"Model{modelName}") # Create new group for each compressor model
        else:
            modelGrp = fModel[f"Model{modelName}"]

        for unitFolder in tqdm.tqdm(unitFolders, desc = "  Unidade", leave=False,  position=1):
            # print("Unidade atual: "+str(unitName))
            unitAttributes = textfile2dict(f"{unitFolder}\\modelInfo.txt")
            unit = unitAttributes["unit"]

            if not unit in modelGrp:
                unitGrp = modelGrp.create_group(unit) # Create new group for each compressor unit
            else:
                unitGrp = modelGrp[unit]
            
            for key in unitAttributes:
                if key in ["model", "fluid"]: # Add attribute to model group
                    modelGrp.attrs[key] = unitAttributes[key]
                elif key != "unit": # Add attribute to unit group
                    unitGrp.attrs[key] = unitAttributes[key]

            # Dict for max and min values of a given unit
            minValuesUnit = {}
            maxValuesUnit = {}

            # Get all tests from a given unit
            fullTestFolder = os.listdir(f"{unitFolder}")
            for k,testFolderName in enumerate(tqdm.tqdm(fullTestFolder, desc = "   Teste", leave = False, position = 2)):
                
                testFolder = f"{unitFolder}\\\{testFolderName}"
                
                if not os.path.isdir(testFolder):
                    continue

                dirList = os.listdir(testFolder)
                if not testFolderName[0].isnumeric(): # Remove N or A from test name
                    testName = testFolderName[2:]
                else:
                    testName = testFolderName

                if testName in unitGrp: # Test already in unit
                    continue
                    
                # Set dataset attributes
                testGrp = unitGrp.create_group(testName)
                testGrp.attrs['startTime'] = os.path.getmtime(f'{testFolder}\\medicoesGerais.dat')
                testGrp.attrs['runningIn'] = True if testFolderName[0] == 'N' else False

                # Check available high frequency readings 
                corrRead = True if "corrente" in dirList else False
                vibRead = True if "vibracao" in dirList else False   
                voltRead = True if "tensao" in dirList else False
                acuRead = True if "acusticas" in dirList else False

                # Read csv data and drop nan columns
                testData = pd.read_table(f'{testFolder}\\medicoesGerais.dat', delimiter = '\t', decimal = ',', encoding='ANSI')
                testData = testData.apply(pd.to_numeric, errors='coerce')
                testData = testData.dropna(axis=1, how='all')
                headers = [nameVar(variable) for variable in testData.columns.values]

                # Get index of test data with compressor turned on
                compressorOn = testData.iloc[:,headers.index('compressorOn')].values
                tStart = testData.iloc[np.nonzero(compressorOn)[0][0] , headers.index("time")]

                # Subtract starting time so that time 0 is the first measurement with compressor turned on
                testData.iloc[:,headers.index('time')] = testData.iloc[:,headers.index('time')] - tStart

                # Store data and headers from csv
                testData = np.array(testData)
                dMeas = testGrp.create_dataset("measurements", data = testData, compression="gzip", shuffle=True)
                dMeas.attrs['columnNames'] = headers

                # Removes time and compressor state from data and headers
                testData = np.delete(testData, [headers.index("time"),headers.index("compressorOn")] , axis = 1)
                headers.remove("compressorOn")
                headers.remove("time")

                # Initializes min and max arrays
                minValuesTest = dict(zip(headers,testData[compressorOn].min(axis = 0)))
                maxValuesTest = dict(zip(headers,testData[compressorOn].max(axis = 0)))

                # Add high-frequency datasets

                if corrRead:
                    create_HF_dataset(testGrp, testFolder, "current")
                    addMinOrMax(minValuesTest, maxValuesTest, "current", testGrp["current"])

                if vibRead:
                    create_HF_dataset(testGrp, testFolder, "vibration")
                    addMinOrMax(minValuesTest, maxValuesTest, "vibrationLateral", testGrp["vibrationLateral"])
                    addMinOrMax(minValuesTest, maxValuesTest, "vibrationRig", testGrp["vibrationRig"])
                    addMinOrMax(minValuesTest, maxValuesTest, "vibrationLongitudinal", testGrp["vibrationLongitudinal"])

                if acuRead:
                    create_HF_dataset(testGrp, testFolder, "acousticEmission")
                    addMinOrMax(minValuesTest, maxValuesTest, "acousticEmission", testGrp["acousticEmission"])

                if voltRead:
                    create_HF_dataset(testGrp, testFolder, "voltage")
                    addMinOrMax(minValuesTest, maxValuesTest, "voltage", testGrp["voltage"])

                # Adds datasets for max and min values to test
                minDset = testGrp.create_dataset("minValues", data = list(minValuesTest.values()), compression="gzip", shuffle=True)
                minDset.attrs["columnNames"] = list(minValuesTest.keys())

                maxDset = testGrp.create_dataset("maxValues", data = list(maxValuesTest.values()), compression="gzip", shuffle=True)
                maxDset.attrs["columnNames"] = list(maxValuesTest.keys())

                # Merge test and unit dicts
                joinMinMaxDict(minValuesUnit, minValuesTest, 'min')
                joinMinMaxDict(maxValuesUnit, maxValuesTest, 'max')

            # Adds datasets for max and min values to unit
            minDset = unitGrp.create_dataset("minValues", data = list(minValuesUnit.values()), compression="gzip", shuffle=True)
            minDset.attrs["columnNames"] = list(minValuesUnit.keys())

            maxDset = unitGrp.create_dataset("maxValues", data = list(maxValuesUnit.values()), compression="gzip", shuffle=True)
            maxDset.attrs["columnNames"] = list(maxValuesUnit.keys())

            # Merge unit and model dicts
            joinMinMaxDict(minValuesModel, minValuesUnit, 'min')
            joinMinMaxDict(maxValuesModel, maxValuesUnit, 'max')

        # Adds datasets for max and min values to model
        minDset = modelGrp.create_dataset("minValues", data = list(minValuesModel.values()), compression="gzip", shuffle=True)
        minDset.attrs["columnNames"] = list(minValuesModel.keys())

        maxDset = modelGrp.create_dataset("maxValues", data = list(maxValuesModel.values()), compression="gzip", shuffle=True)
        maxDset.attrs["columnNames"] = list(maxValuesModel.keys())

    if supressWarnings:
        warnings.resetwarnings()

def convertFolders(folderIn: list[str], folderOut: str, filePrefix = "Model", supressWarnings = False) -> list[str]:
    allUnitFolders = os.listdir(folderIn)

    allModels = [re.findall("Unidade .", unit) for unit in allUnitFolders] # Get all folder names with "Unidade "
    allModels = set([name[0][-1] for name in allModels if len(name)>0]) # Filter for unique models

    listOut = []
    for model in tqdm.tqdm(allModels,desc = " Modelo", position=0):
        r = re.compile(f"Unidade {model}.*")
        unitFolders = list(filter(r.match,allUnitFolders))
        convertModel(unitFolders, f"{folderOut}\\Model{model}.hdf5", model, supressWarnings=supressWarnings)
        listOut.append(f"{folderOut}\\{filePrefix}{model}.hdf5")

    return listOut

if __name__ == "__main__":
    mainFolder = input("Digite a pasta de origem:\n")
    saveFolder = input("Digite a pasta de destino:\n")
    # mainFolder = "D:\Dados - Thaler\Documentos\Amaciamento\Ensaios Brutos"
    # saveFolder = "D:\Dados - Thaler\Documentos\Amaciamento"

    fullUnitFolder = os.listdir(mainFolder) # Extract folders

    convertFolders(fullUnitFolder, saveFolder)