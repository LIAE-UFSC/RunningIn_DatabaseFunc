# RunningIn_DatabaseFunc

## Description
`pyRInDB` is a Python library for handling and processing HDF5 files from compressor tests with LabVIEW integration. It provides utilities for managing running-in databases, extracting data, and converting models. 
The main functionalities provided are:
* Converting LabVIEW waveform files to hdf5 dataset;
* Extract data from hdf5 dataset to pandas dataframe and to dict.

## Installation
To use the scripts in this repository, you'll need to install the required dependencies. You can do this using pip and the provided requirements.txt file:

```bash
pip install -r pyRInDB/requirements.txt
```

Using the LabVIEW waveform conversion functionalities also requires LabVIEW Runtime 2022. Visit [the official NI website](https://www.ni.com/en/support/downloads/software-products/download.labview-runtime.html#301740) for download options. The bit architecture should be the same as the python interpreter.

If a different LabVIEW version is required, the LabVIEW project in the `build_dll` folder should be rebuild. [This guide](https://forums.ni.com/t5/Community-Documents/Creating-a-DLL-from-LabVIEW-code/ta-p/3514929) provides a tutorial on building DLLs in LabVIEW. The resulting dll should be saved as either `source\WvfRead.dll` or `source\WvfRead64.dll` depending on the LabVIEW and python bit architecture.

## Usage

### Processing raw test data

Raw test data is comprised of multiple tests of multiple compressor units. 

A root test folder should contain one or more unit folders, named `Unidade XN` where X is the compressor model and N is the unit number. In hdf5 conversion, each compressor model will be saved in a different file.

Each unit test folder should contain the following:
* One or more test data folders, named `N_YYYY_MM_DD` or `A_YYYY_MM_DD`. YYYY_MM_DD is the test date, and N and A indicate if the test comes from a new compressor unit or from an already run-in one, respectivelly.
* A `modelInfo.txt` file, which contains attributes to be added to the hdf5 database. Each attribute should be written in a newline, in the format `attributeName:attributeValue`.

Each test folder should contain the following:
* A csv file called `medicoesGerais.dat`, with rows being measurements and columns being variables.
* Zero or more folders containing high-frequency measurements. The measurements should be stored as LabVIEW waveform files, and supported folders are:
  * corrente: current measurements.
  * vibracao: vibration measurements (2 or 3 channels).
  * acusticas: acoustic emissions measurements.
  * tensao: voltage measurements.

To process test data and convert them to HDF5 format, use the `convertFolders` method:

```python
# Example code for processing LabVIEW files
from source.convertDB import convertFolders

convertFolders('path/to/tests', 'path/to/output')
```

### Extracting Data from HDF5 Database

#### Printing the name of all tests in the HDF5 file

```python
# Example code for printing the name of all tests in the HDF5 file
from pyRInDB import RunIn_File

# Open HDF5 file as a running-in database
with RunIn_File(path) as testFile:
    for unit in testFile:
        for test in unit:
            print(f"Unit: {unit} | Test: {test}")
```

#### Extracting data from the database

```python
# Example code for extracting data from the database
from pyRInDB import RunIn_File

# Open HDF5 file as a running-in database
with RunIn_File("path/to/file.hdf5") as testFile:
    unit = testFile["A1"]  # Selects unit A1 from the database. Can also be accessed by indexing
    test = unit[0]  # Selects the first test of the unit. Can also be accessed by date ("YYYY_MM_DD")

    # Extracts dataframe with vibration and discharge pressure data, for test measurements 0, 100 and 200. See documentation for full list of variables.
    dados = test.to_dataframe(vars=["vibrationRAWLateral", "presDischarge"], indexes=[0, 100, 200])
```

### Generating datasets

#### Generating a dataset using convertModel

```python
# Example code for generating a dataset using convertModel
from pyRInDB import RunIn_File

folders_in = ["path/to/unit1", "path/to/unit2"]
file_out = "path/to/output/model.hdf5"
model_name = "ModelA"

# Convert the folders into a single HDF5 file
run_in_file = RunIn_File.convertModel(folders_in, file_out, model_name)

print(f"Model dataset created at: {file_out}")

print(dados)
```

#### Generating a dataset using convertFolders

```python
# Example code for generating datasets using convertFolders
from pyRInDB import RunIn_File

folder_in = "path/to/input/folder"
folder_out = "path/to/output/folder"
file_prefix = "Model"

# Convert the folders into multiple HDF5 files
run_in_files = RunIn_File.convertFolders(folder_in, folder_out, file_prefix)

for file in run_in_files:
    print(f"Model dataset created at: {file}")
```


## Contributing

Contribution to this repository is restricted to researchers from LIAE/LABMETRO. Access should be requested via e-mail.

## Contact

LIAE/LABMETRO: liae@labmetro.ufsc.br
