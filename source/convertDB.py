import os
import re
import csv
import h5py
import warnings
import tqdm
import numpy as np
import pandas as pd
from source.waveformConversion import Waveform

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
            ["vibRMSLateral","Aceleração RMS Inferior Corpo [g]"],
            ["vibRMSRig","Aceleração RMS Bancada [g]"],
            ["vibRMSLongitudinal","Aceleração RMS Superior Corpo [g]"],
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

def convertFolder(UnitFolderIn, UnitFolderOut, supressWarnings = False):

    if supressWarnings:
        warnings.filterwarnings('ignore')

    allUnitFolders = os.listdir(UnitFolderIn)

    allModels = [re.findall("Unidade .", unit) for unit in allUnitFolders] # Get all folder names with "Unidade "
    allModels = set([name[0][-1] for name in allModels if len(name)>0]) # Filter for unique models

    for model in tqdm.tqdm(allModels,desc = " Modelo", position=0):
        r = re.compile(f"Unidade {model}.*")
        unitFolders = list(filter(r.match,allUnitFolders))

        # Dict for max and min values of a given unit
        minValuesModel = {}
        maxValuesModel = {}

        with h5py.File(f"{UnitFolderOut}/datasetModel{model}.hdf5", "w") as fModel:
            modelGrp = fModel.create_group(f"Model{model}") # Create new group for each compressor model
            
            for unitName in tqdm.tqdm(unitFolders, desc = "  Unidade", leave=False,  position=1):
                # print("Unidade atual: "+str(unitName))
                unit = unitName.replace(f"Unidade ",'') # Get unit names
                unitGrp = modelGrp.create_group(unit) # Create new group for each compressor unit

                # Dict for max and min values of a given unit
                minValuesUnit = {}
                maxValuesUnit = {}

                fullTestFolder = os.listdir(f"{UnitFolderIn}/{unitName}") # Get all tests from a given unit

                for k,testName in enumerate(tqdm.tqdm(fullTestFolder, desc = "   Teste", leave = False, position = 2)):
                    testFolder = f"{UnitFolderIn}/{unitName}/{testName}"

                    dirList = os.listdir(testFolder)

                    # Set dataset attributes
                    testGrp = unitGrp.create_group(testName[2:])
                    testGrp.attrs['startTime'] = os.path.getmtime(f'{testFolder}/medicoesGerais.dat')
                    testGrp.attrs['runningIn'] = True if testName[0] == 'N' else False

                    # Check available high frequency readings 
                    corrRead = True if "corrente" in dirList else False
                    vibRead = True if "vibracao" in dirList else False   
                    voltRead = True if "tensao" in dirList else False
                    acuRead = True if "acusticas" in dirList else False

                    # Read csv data
                    testData = pd.read_table('D:\\Dados - Thaler\\Documentos\\Amaciamento\\Ensaios Brutos\\Unidade C1\\A_2022_10_10\\medicoesGerais.dat', delimiter = '\t', decimal = ',', encoding='ANSI')
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

                    for indexMeas, row in enumerate(tqdm.tqdm(testData,desc="    Arquivo", position=3, leave = False)):
                        # Create new group for each measurement
                        measurementGrp = testGrp.create_group(str(indexMeas))

                        # Add high-frequency datasets

                        if corrRead:
                            filePath = f"{testFolder}/corrente/corr{indexMeas}.dat"
                            if os.path.isfile(filePath):
                                try:
                                    wvf = Waveform.read_labview_waveform(filePath,0)
                                    
                                    dSet = measurementGrp.create_dataset("currentRAW", data = wvf.data, compression="gzip", shuffle=True)
                                    dSet.attrs["dt"] = wvf.dt

                                    if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                                        testGrp.attrs['startTime'] = os.path.getmtime(filePath)
                                    
                                    attrName = "currentRAW"

                                    # Compare and add to min max dict
                                    addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)
                                
                                except:
                                    warnings.warn("File empty:" + filePath)

                        if vibRead:
                            filePath = f"{testFolder}/vibracao/vib{indexMeas}.dat"
                            if os.path.isfile(filePath):
                                try:
                                    wvf = Waveform.read_labview_waveform(filePath,0)
                                    
                                    dSet = measurementGrp.create_dataset("vibrationRAWLateral", data = wvf.data, compression="gzip", shuffle=True)
                                    dSet.attrs["dt"] = wvf.dt

                                    attrName = "vibrationRAWLateral"

                                    # Compare and add to min max dict
                                    addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                                    wvf = Waveform.read_labview_waveform(filePath,1)
                                    dSet = measurementGrp.create_dataset("vibrationRAWRig", data = wvf.data, compression="gzip", shuffle=True)
                                    dSet.attrs["dt"] = wvf.dt

                                    attrName = "vibrationRAWRig"

                                    # Compare and add to min max dict
                                    addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                                    wvf = Waveform.read_labview_waveform(filePath,2)
                                    dSet = measurementGrp.create_dataset("vibrationRAWLongitudinal", data = wvf.data, compression="gzip", shuffle=True)
                                    dSet.attrs["dt"] = wvf.dt

                                    attrName = "vibrationRAWLongitudinal"

                                    # Compare and add to min max dict
                                    addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                                    if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                                        testGrp.attrs['startTime'] = os.path.getmtime(filePath)
                                except:
                                    warnings.warn("File empty:" + filePath)

                                
                        if acuRead :
                            filePath = f"{testFolder}/acusticas/acu{indexMeas}.dat"
                            if os.path.isfile(filePath):
                                try:
                                    wvf = Waveform.read_labview_waveform(filePath,0)
                                    
                                    dSet = measurementGrp.create_dataset("acousticEmissionRAW", data = wvf.data, compression="gzip", shuffle=True)
                                    dSet.attrs["dt"] = wvf.dt

                                    if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                                        testGrp.attrs['startTime'] = os.path.getmtime(filePath)

                                    attrName = "acousticEmissionRAW"

                                    # Compare and add to min max dict
                                    addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                                except:
                                    warnings.warn("File empty:" + filePath)

                            
                        if voltRead:
                            filePath = f"{testFolder}/tensao/ten{indexMeas}.dat"
                            if os.path.isfile(filePath):
                                try:
                                    wvf = Waveform.read_labview_waveform(filePath,0)
                                    
                                    dSet = measurementGrp.create_dataset("voltageRAW", data = wvf.data, compression="gzip", shuffle=True)
                                    dSet.attrs["dt"] = wvf.dt

                                    if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                                        testGrp.attrs['startTime'] = os.path.getmtime(filePath)

                                    attrName = "voltageRAW"

                                    # Compare and add to min max dict
                                    addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                                except:
                                    warnings.warn("File empty:" + filePath)

                    # Adds datasets for max and min values to test
                    minDset = testGrp.create_dataset("minValues", data = list(minValuesTest.values()), compression="gzip", shuffle=True)
                    minDset.attrs["columnNames"] = list(minValuesTest.keys())

                    maxDset = testGrp.create_dataset("maxValues", data = list(maxValuesTest.values()), compression="gzip", shuffle=True)
                    maxDset.attrs["columnNames"] = list(maxValuesTest.keys())

                # Merge test and unit dicts
                joinMinMaxDict(minValuesUnit, minValuesTest, 'min')
                joinMinMaxDict(maxValuesUnit, maxValuesTest, 'max')

                # Adds datasets for max and min values to test
                minDset = unitGrp.create_dataset("minValues", data = list(minValuesUnit.values()), compression="gzip", shuffle=True)
                minDset.attrs["columnNames"] = list(minValuesUnit.keys())

                maxDset = unitGrp.create_dataset("maxValues", data = list(maxValuesUnit.values()), compression="gzip", shuffle=True)
                maxDset.attrs["columnNames"] = list(maxValuesUnit.keys())

            # Merge unit and model dicts
            joinMinMaxDict(minValuesModel, minValuesUnit, 'min')
            joinMinMaxDict(maxValuesModel, maxValuesUnit, 'max')

            # Adds datasets for max and min values to test
            minDset = modelGrp.create_dataset("minValues", data = list(minValuesModel.values()), compression="gzip", shuffle=True)
            minDset.attrs["columnNames"] = list(minValuesModel.keys())

            maxDset = modelGrp.create_dataset("maxValues", data = list(maxValuesModel.values()), compression="gzip", shuffle=True)
            maxDset.attrs["columnNames"] = list(maxValuesModel.keys())

    if supressWarnings:
        warnings.resetwarnings()

def addTest(hdf5File: str, testFolder: str, unitName: str):
    # Add test to hdf file
    with h5py.File(hdf5File, "a") as fModel:
        model = unitName[0].upper()
        unit = unitName[1:]

        if model in fModel:
            modelGrp = fModel[f"Model{model}"]
        else:
            fModel.create_group(f"Model{model}")

        if unit in modelGrp:
            unitGrp = modelGrp[unit]
        else:
            unitGrp = modelGrp.create_group(unit)
        


        # Get test name from folder
        testName = re.compile("\w*$").findall(testFolder)[0]

        # Variable for the first time that the compressor is turned on
        tOn = float('inf')

        # Dict for max and min values of a given test
        minValuesTest = {}
        maxValuesTest = {}

        dirList = os.listdir(testFolder)

        # Set dataset attributes
        testGrp = unitGrp.create_group(testName[2:])
        testGrp.attrs['startTime'] = os.path.getmtime(f'{testFolder}/medicoesGerais.dat')
        testGrp.attrs['runningIn'] = True if testName[0] == 'N' else False

        # Check available high frequency readings 
        corrRead = True if "corrente" in dirList else False
        vibRead = True if "vibracao" in dirList else False   
        voltRead = True if "tensao" in dirList else False
        acuRead = True if "acusticas" in dirList else False

        with open(f'{testFolder}/medicoesGerais.dat', encoding='ANSI') as csv_file:
            nLines = len(csv_file.readlines())

        with open(f'{testFolder}/medicoesGerais.dat', encoding='ANSI') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter='\t')
            headers = next(csv_reader) # Extract headers
            
            for indexMeas, row in enumerate(tqdm.tqdm(csv_reader,desc="    Arquivo", position=3, leave = False, total = nLines-1)):
                # Create new group for each measurement
                measurementGrp = testGrp.create_group(str(indexMeas))

                for columnNum, element in enumerate(row): # Add attributes
                    attrName = nameVar(headers[columnNum])
                    value = float(element.replace(",","."))
                    measurementGrp.attrs[attrName] = value

                    # Compare and add to min max dict
                    addMinOrMax(minValuesTest, maxValuesTest, attrName, value)

                # Get first "compressor on" time
                tOn = tOn if not measurementGrp.attrs["compressorOn"] else min(tOn, measurementGrp.attrs["time"])

                # Add high-frequency datasets

                if corrRead:
                    filePath = f"{testFolder}/corrente/corr{indexMeas}.dat"
                    try:
                        wvf = Waveform.read_labview_waveform(filePath,0)
                        dSet = measurementGrp.create_dataset("current", data = wvf.data, compression="gzip", shuffle=True)
                        dSet.attrs["dt"] = wvf.dt

                        if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                            testGrp.attrs['startTime'] = os.path.getmtime(filePath)
                        
                        attrName = "currentRAW"

                        # Compare and add to min max dict
                        addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)
                        
                    except:
                        warnings.warn("File not found or empty:" + filePath)

                if vibRead:
                    filePath = f"{testFolder}/vibracao/vib{indexMeas}.dat"
                    try:
                        wvf = Waveform.read_labview_waveform(filePath,0)
                        dSet = measurementGrp.create_dataset("vibrationLateral", data = wvf.data, compression="gzip", shuffle=True)
                        dSet.attrs["dt"] = wvf.dt

                        attrName = "vibrationRAWLateral"

                        # Compare and add to min max dict
                        addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                        wvf = Waveform.read_labview_waveform(filePath,1)
                        dSet = measurementGrp.create_dataset("vibrationRigDummy", data = wvf.data, compression="gzip", shuffle=True)
                        dSet.attrs["dt"] = wvf.dt

                        attrName = "vibrationRAWRig"

                        # Compare and add to min max dict
                        addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                        wvf = Waveform.read_labview_waveform(filePath,2)
                        dSet = measurementGrp.create_dataset("vibrationLongitudinal", data = wvf.data, compression="gzip", shuffle=True)
                        dSet.attrs["dt"] = wvf.dt

                        attrName = "vibrationRAWLongitudinal"

                        # Compare and add to min max dict
                        addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                        if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                            testGrp.attrs['startTime'] = os.path.getmtime(filePath)
                    except:
                        warnings.warn("File not found or empty:" + filePath)

                        
                if acuRead:
                    filePath = f"{testFolder}/acusticas/acu{indexMeas}.dat"
                    try:
                        wvf = Waveform.read_labview_waveform(filePath,0)
                        dSet = measurementGrp.create_dataset("acousticEmission", data = wvf.data, compression="gzip", shuffle=True)
                        dSet.attrs["dt"] = wvf.dt

                        if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                            testGrp.attrs['startTime'] = os.path.getmtime(filePath)

                        attrName = "acousticEmissionRAW"

                        # Compare and add to min max dict
                        addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                    except:
                        warnings.warn("File not found or empty:" + filePath)

                    
                if voltRead:
                    filePath = f"{testFolder}/tensao/ten{indexMeas}.dat"
                    try:
                        wvf = Waveform.read_labview_waveform(filePath,0)
                        dSet = measurementGrp.create_dataset("voltage", data = wvf.data, compression="gzip", shuffle=True)
                        dSet.attrs["dt"] = wvf.dt

                        if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                            testGrp.attrs['startTime'] = os.path.getmtime(filePath)

                        attrName = "voltageRAW"

                        # Compare and add to min max dict
                        addMinOrMax(minValuesTest, maxValuesTest, attrName, wvf.data)

                    except:
                        warnings.warn("File not found or empty:" + filePath)
        
        for key in minValuesTest:
            testGrp.attrs[f"min_{key}"] = minValuesTest[key]
            testGrp.attrs[f"max_{key}"] = maxValuesTest[key]
            if f"min_{key}" in unitGrp.attrs:
                unitGrp.attrs[f"min_{key}"] = min(unitGrp.attrs[f"min_{key}"],minValuesTest[key])
                unitGrp.attrs[f"max_{key}"] = max(unitGrp.attrs[f"max_{key}"],maxValuesTest[key])
            if f"min_{key}" in modelGrp.attrs:
                modelGrp.attrs[f"min_{key}"] = min(modelGrp.attrs[f"min_{key}"],minValuesTest[key])
                modelGrp.attrs[f"max_{key}"] = max(modelGrp.attrs[f"max_{key}"],maxValuesTest[key])


if __name__ == "__main__":
    mainFolder = input("Digite a pasta de origem:\n")
    saveFolder = input("Digite a pasta de destino:\n")
    # mainFolder = "D:/Dados - Thaler/Documentos/Amaciamento/Ensaios Brutos"
    # saveFolder = "D:/Dados - Thaler/Documentos/Amaciamento"

    fullUnitFolder = os.listdir(mainFolder) # Extract folders

    convertFolder(fullUnitFolder, saveFolder)