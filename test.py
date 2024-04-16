from source.convertDB import *
from source.formatDB_MIMICRI import *

# folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos"
# folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull"
# convertFolders(folderIn, folderOut, supressWarnings = False)

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"

formatDB_MIMICRI(folderIn, folderOut)

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelB.hdf5"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelB.hdf5"

formatDB_MIMICRI(folderIn, folderOut)

# folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelC.hdf5"
# folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelC.hdf5"

# formatDB_MIMICRI(folderIn, folderOut)