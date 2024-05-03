# Para poder chamar o modulo da outra pasta
import sys, os
sys.path.append(os.path.abspath('.'))

from source.formatDB_MIMICRI import *

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelA.hdf5"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"

formatDB_MIMICRI(folderIn, folderOut)

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelB.hdf5"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelB.hdf5"

formatDB_MIMICRI(folderIn, folderOut)

folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseFull\datasetModelC.hdf5"
folderOut = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelC.hdf5"

formatDB_MIMICRI(folderIn, folderOut)