# Call functions from root folder
import sys, os
sys.path.append(os.path.abspath('.'))

from pyRInDB.dbcreate import convertFolders

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull"
convertFolders(folderIn, folderOut, supressWarnings = False)