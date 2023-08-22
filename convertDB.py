import os
import re
import csv
import h5py
import warnings
import tqdm
from waveformConversion import Waveform


def nameVar(headerName:str) -> str:
    # Convert column name from "medicoesGerais.dat" file to hdf5 attribute name

    varNames = [["time","Tempo [s]"],
            ["temperatureSuction","Temperatura Sucção [ºC]"],
            ["temperatureCompressorBody","Temperatura Corpo [ºC]"],
            ["temperatureBuffer","Temperatura Reservatório [ºC]"],
            ["temperatureDischarge","Temperatura Descarga [ºC]"],
            ["pressureSuction_Analog","Pressão Sucção [bar]"],
            ["pressureDischarge","Pressão Descarga [bar]"],
            ["pressureBuffer","Pressão Intermediária [bar]"],
            ["valveDischargePositionSetpoint","Setpoint da Abertura da Válvula [Passos]"],
            ["valveDischargeOpening","Abertura da válvula [Passos]"],
            ["valveDischargeLimit","Limite da válvula [Passos]"],
            ["valveSuctionVoltage","Tensão da válvula de sucção [V]"],
            ["currentRMS","Corrente RMS [A]"],
            ["compressorOn","Compressor Ativado"],
            ["vibrationRMSLateral","Aceleração RMS Inferior Corpo [g]"],
            ["vibrationRMSRig","Aceleração RMS Bancada [g]"],
            ["vibrationRMSLongitudinal","Aceleração RMS Superior Corpo [g]"],
            ["massFlow","Vazão Mássica [kg/h]"],
            ["resistorDischargeDC","Duty Cycle Descarga [%]"],
            ["pressureSuction_GEPreValve","Pressão Sucção Válvula (DPS) [bar]"],
            ["pressureSuction_GEPostValve","Pressão Sucção Compressor (DPS) [bar]"],
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

mainFolder = "D:/Dados - Thaler/Documentos/Amaciamento/Ensaios Brutos"
saveFolder = "D:/Dados - Thaler/Documentos/Amaciamento"

fullUnitFolder = os.listdir(mainFolder) # Extract folders

allModels = set([re.findall("Unidade .", unit)[0][-1] for unit in fullUnitFolder]) # Get all compressor models from folder names

for model in tqdm.tqdm(allModels,desc = " Modelo", position=0):
    r = re.compile(f"Unidade {model}.*")
    unitFolders = list(filter(r.match,fullUnitFolder))

    with h5py.File(f"{saveFolder}/datasetModel{model}.hdf5", "w") as f:
        modelGrp = f.create_group(f"Model{model}") # Create new group for each compressor model
        
        for unitName in tqdm.tqdm(unitFolders, desc = "  Unidade", leave=False,  position=1):
            # print("Unidade atual: "+str(unitName))
            unit = unitName.replace(f"Unidade ",'') # Get unit names
            unitGrp = modelGrp.create_group(unit) # Create new group for each compressor unit

            fullTestFolder = os.listdir(f"{mainFolder}/{unitName}") # Get all tests from a given unit

            for k,testName in enumerate(tqdm.tqdm(fullTestFolder, desc = "   Teste", leave = False, position = 2)):
                testFolder = f"{mainFolder}/{unitName}/{testName}"
                # print(f"Ensaio {k+1}/{len(fullTestFolder)}")

                dirList = os.listdir(testFolder)

                # Set dataset attributes
                testGrp = unitGrp.create_group(testName[3:])
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
                            measurementGrp.attrs[nameVar(headers[columnNum])] = float(element.replace(",","."))
                            
                        # Add high-frequency datasets

                        if corrRead:
                            filePath = f"{testFolder}/corrente/corr{indexMeas}.dat"
                            try:
                                wvf = Waveform.read_labview_waveform(filePath,0)
                                dSet = measurementGrp.create_dataset("current", data = wvf.data, compression="gzip", shuffle=True)
                                dSet.attrs["dt"] = wvf.dt

                                if testGrp.attrs['startTime']> os.path.getmtime(filePath): # Current file is older than MedicoesGerais
                                    testGrp.attrs['startTime'] = os.path.getmtime(filePath)
                            except:
                                warnings.warn("File not found or empty:" + filePath)

                        if vibRead:
                            filePath = f"{testFolder}/vibracao/vib{indexMeas}.dat"
                            try:
                                wvf = Waveform.read_labview_waveform(filePath,0)
                                dSet = measurementGrp.create_dataset("vibrationLateral", data = wvf.data, compression="gzip", shuffle=True)
                                dSet.attrs["dt"] = wvf.dt

                                wvf = Waveform.read_labview_waveform(filePath,1)
                                dSet = measurementGrp.create_dataset("vibrationRigDummy", data = wvf.data, compression="gzip", shuffle=True)
                                dSet.attrs["dt"] = wvf.dt

                                wvf = Waveform.read_labview_waveform(filePath,2)
                                dSet = measurementGrp.create_dataset("vibrationLongitudinal", data = wvf.data, compression="gzip", shuffle=True)
                                dSet.attrs["dt"] = wvf.dt

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
                            except:
                                warnings.warn("File not found or empty:" + filePath)