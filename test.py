import pyRInDB as ridb

folders = [r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A1",
           r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos\Unidade A2"]

# Create a new RInDB object
RInDB = ridb.RunIn_File.convertModel(folders, "database.hdf5", "A")

# folderIn = r"\\LIAE-SANTINHO\Backups\Amaciamento_DadosBrutos"
# folderOut = r""

# # Create a new RInDB object
# RInDB = ridb.RunIn_File.convertFolders(folderIn, folderOut)