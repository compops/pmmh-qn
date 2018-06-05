# Python code

## Installation
This code was developed using Anaconda 3 and Python 3.6.3. To satisfy these requirements, please download Anaconda from https://www.anaconda.com/ for your OS and some extra libraries by running
``` bash
conda install quandl palettable cython numpy scipy pandas
```
If you opt to install python yourself some additional libraries are required. These can be installed by executing the following in the root directory:
``` bash
pip install -r requirements.txt
```
see the file for the exact versions used for running the experiments in the paper.

### Cython
Some of the code is written in Cython and requires compilation before it can be run. Please execute the following from the `python` directory
``` bash
python setup.py build_ext --inplace
```
to compile this code. The number of particles and observations are hard-coded into the C-code due to the use of static arrays for speed. To change this, open the file corresponding to the model of interest and change the constants `NPART` and `NOBS` in the beginning of the file. Note that `NOBS` is T+1 (as we include the unknown initial state).
