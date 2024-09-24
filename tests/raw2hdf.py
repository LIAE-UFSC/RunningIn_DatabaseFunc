# Call functions from root folder
import sys, os
sys.path.append(os.path.abspath('.'))

import pyRInDB as ridb

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull"
ridb.RunIn_File.convertFolders(folderIn, folderOut, supressWarnings = False)