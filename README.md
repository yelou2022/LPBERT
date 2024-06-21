# LPBERT

## Dependencies
* ProteinBERT  Refer to the official installation tutorial. [click here](https://github.com/nadavbra/protein_bert). However, please note that the official recommended version of tensorflow is 2.4.0, and LPBERT uses version 2.6.0
* tensorflow 2.6.0
* python 3.9
* sklearn

## Directory Structure
* **data**: Contains datasets collected from other papers and self built datasets
* **environment_test.py**: ProteinBERT environment test file. Please note that the environment configuration is successful only when there are no errors when running this file.
* **encode.py**:  Obtaining global and local representations of protein sequences from ProteinBERT
* **train.py**: Training files for LPBERT
* **utility**: Methods used in LPBERT
* **params.py**: Parameter optimization file
