from source.convertDB import *

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull"

convertFolder(folderIn, folderOut, supressWarnings = False)