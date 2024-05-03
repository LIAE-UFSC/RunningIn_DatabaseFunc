from runInDB_utils import RunIn_File
import warnings

# Variables that just need to be loaded [databaseName,outName]
_LOAD_VARS = [["time","Tempo"],
            ["temperatureSuction","TempSuc"],
            ["temperatureCompressorBody","TempCorpo"],
            ["temperatureBuffer","TempReservatorio"],
            ["temperatureDischarge","TemperDesc"],
            ["presSuction_Analog","PressaoSuccaoAnalog"],
            ["presDischarge","PressaoDescarga"],
            ["presBuffer","PressIntermediaria"],
            ["valveDischargePositionSetpoint","DescargaAberturaSP"],
            ["valveDischargeOpening","DescargaAbertura"],
            ["valveDischargeLimit","DescargaLimite"],
            ["valveSuctionVoltage","SuccaoAbertura"],
            ["compressorOn","CompressorLigado"],
            ["massFlow","Vazao"],
            ["resistorDischargeDC","ResistorDescargaDC"],
            ["presSuction_GEPreValve","PressaoSuccaoPreValv"],
            ["presSuction_GEPostValve","PressaoSuccaoPosValv"],
            ["resistorSuctionDC","ResistorSuccaoDC"],
            ["temperatureCoil1","TempBobina1"],
            ["temperatureCoil2","TempBobina2"],
            ["currentRMS_WT130","CorrenteRMSYokogawa"],
            ["voltageRMS_WT130","TensaoRMSYokogawa"],
            ["PowerRMS_WT130","PotenciaRMSYokogawa"],
            ["currentRMS_TMCS","CorrenteRMSTMCS"],
            ["voltageRMS_NI_DAQ","TensaoRMSDAQ"],
            ["temperatureRoom","TempAmbiente"],
            ["runningIn","Amaciado"],
            ]

# Variables that need to be processed [databaseName,outName]
_PROCESSED_VARS = [
            # Extract features from high frequency measurements
            ["currentRMS","CorrenteRMS"],
            ["currentKur","CorrenteCurtose"],
            ["currentSkew","CorrenteAssimetria"],
            ["currentShape","CorrenteForma"],
            ["currentTHD","CorrenteTHD"],
            ["currentCrest","CorrentePico"],
            ["currentPeak","CorrenteCrista"],
            ["currentVar","CorrenteVariancia"],
            ["currentStd","CorrenteDesvio"],
            ["vibRigRMS","VibracaoBancadaInferiorRMS"],
            ["vibLongitudinalRMS","VibracaoCalotaSuperiorRMS"],
            ["vibLongitudinalKur","VibracaoCalotaSuperiorCurtose"],
            ["vibLongitudinalSkew","VibracaoCalotaSuperiorAssimetria"],
            ["vibLongitudinalShape","VibracaoCalotaSuperiorForma"],
            ["vibLongitudinalTHD","VibracaoCalotaSuperiorTHD"],
            ["vibLongitudinalCrest","VibracaoCalotaSuperiorPico"],
            ["vibLongitudinalPeak","VibracaoCalotaSuperiorCrista"],
            ["vibLongitudinalVar","VibracaoCalotaSuperiorVariancia"],
            ["vibLongitudinalStd","VibracaoCalotaSuperiorDesvio,"],
            ["vibLateralRMS","VibracaoCalotaInferiorRMS"],
            ["vibLateralKur","VibracaoCalotaInferiorCurtose"],
            ["vibLateralSkew","VibracaoCalotaInferiorAssimetria"],
            ["vibLateralShape","VibracaoCalotaInferiorForma"],
            ["vibLateralTHD","VibracaoCalotaInferiorTHD"],
            ["vibLateralCrest","VibracaoCalotaInferiorPico"],
            ["vibLateralPeak","VibracaoCalotaInferiorCrista"],
            ["vibLateralVar","VibracaoCalotaInferiorVariancia"],
            ["vibLateralStd","VibracaoCalotaInferiorDesvio,"],
            ["acousticEmissionsRMS","AcusticasRMS"],
            ["acousticEmissionsKur","AcusticasCurtose"],
            ["acousticEmissionsSkew","AcusticasAssimetria"],
            ["acousticEmissionsShape","AcusticasForma"],
            ["acousticEmissionsTHD","AcusticasTHD"],
            ["acousticEmissionsCrest","AcusticasPico"],
            ["acousticEmissionsPeak","AcusticasCrista"],
            ["acousticEmissionsVar","AcusticasVariancia"],
            ["acousticEmissionsStd","AcusticasDesvio"],

            ["power","Potencia"] # Multiply voltage and current

            ["presSuction","PressaoSuccao"], # Select based on available readings
            ]

_ALL_PROCESS = ["RMS", "Kur", "Skew", "Shape", "THD","Crest","Peak","Var","Std"]

def possibleVars(test):
    # Checks all available vars for a given test
    pass

def hdf2csv(pathHDF, target:dict = None, tStart = None, tEnd = None, vars = None, discardEnd = True):
    
    # Check time conditions:
    if tStart is None:
        tStart = 0
    if tEnd is None:
        tEnd = float("inf")
    if tStart > tEnd:
        raise Exception("Starting time should be smaller than end time.")
    
    if vars is None:
        # Selects all possible variables if None is given
        vars = [varName[0] for varName in _ALL_VARS]

    with RunIn_File(pathHDF) as file:
        if target is None:
            target = dict()
            for unit in file:
                target[str(unit)] = str(unit.tests)

        for unit,tests in target.items():
            for test in tests:
                testVars = possibleVars(test)
                pass

if __name__ == "__main__":
    path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"
    hdf2csv(path)
    
