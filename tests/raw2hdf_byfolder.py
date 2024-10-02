# Call functions from root folder
import sys, os
sys.path.append(os.path.abspath('.'))

import pyRInDB as ridb

# foldersIn = [r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A1",
#              r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A2",
#              r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A3",
#              r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A4",
#              r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A5"]
# fileOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\ModelA.hdf5"
# ridb.RunIn_File.convertModel(foldersIn, fileOut, "ModelA")

foldersIn = [r"\\LIAE-SANTINHO\Backups\Desgaste_DadosBrutos\Unidade C1"]
fileOut = r"\\LIAE-SANTINHO\Backups\Desgaste_DatabaseFull\ModelC.hdf5"  
ridb.RunIn_File.convertModel(foldersIn, fileOut, "ModelC")