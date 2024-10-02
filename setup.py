from setuptools import setup, find_packages

setup(
    name="pyRInDB",
    version="2.0.0",
    description="A project for handling and processing HDF5 files for compressor tests with LabVIEW integration.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gabriel Thaler",
    author_email="gabriel.thaler@labmetro.ufsc.br",
    url="https://github.com/LIAE-UFSC/RunningIn_DatabaseFunc",
    packages=find_packages(where="pyRInDB"),
    package_dir={"": "pyRInDB"},
    include_package_data=True,
    package_data={
        "pyRInDB": [
            "labview_buildDLL/LV2022_x86/*.vi",
            "utils/*.dll"
        ]
    },
    install_requires=[
        "h5py>=3.3.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "typing_extensions>=3.10.0.0"
    ],
    entry_points={
        "console_scripts": [
            "pyRInDB=pyRInDB.main_class:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)