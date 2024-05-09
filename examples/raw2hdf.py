# Call functions from root folder
import sys, os
sys.path.append(os.path.abspath('.'))

from source.convertDB import *
from source.formatDB_MIMICRI import *

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull"
convertFolders(folderIn, folderOut, supressWarnings = False)