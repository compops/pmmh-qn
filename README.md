# Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals
This code was downloaded from https://github.com/compops/pmmh-qn and contains the code and data used to produce the results in the papers:

J. Dahlin, A. Wills and B. Ninness, **Correlated pseudo-marginal Metropolis-Hastings using quasi-Newton proposals**. Pre-print, arXiv:1807:nnnnn, 2018.

The paper is available as a preprint from https://arxiv.org/abs/1807.nnnnn.

## Python code (python/)
This code is used to set-up and run all the experiments in the paper. The code can possibly also be modified for other models. See the `README.md` file for more information.

### Docker
A simple method to reproduce the results is to make use of the Docker container build from the code in this repository when the paper was published. Docker enables you to recreate the computational environment used to create the results in the paper. Hence, it automatically downloads the correct version of Python and all dependencies.

First, you need to download and installer Docker on your OS. Please see https://docs.docker.com/engine/installation/ for instructions on how to do this. Secondly, you can run the Docker container by running the command
``` bash
docker run --name pmmh-qn-run compops/pmmh-qn:draft1
```
This will download the code and execute it on your computer. The progress will be printed to the screen. Note that the runs will take a day or two to complete. Thirdly, The results can then be access by
``` bash
docker cp pmmh-qn-run:/app/pmmh-qn-results.tgz .
```
which copies a tarball of the results into the current folder. To reproduce the plots from the paper, extract the tarball and move the contents of the results folder into a folder called results in a cloned version of the GitHub repository. Follow the instruction for the R code to create pdf versions of the plots.

## R code (r/)
This code is used to generate diagnostic plot as well as plots and table for the paper. See the `README.md` file for more information.

## Binaries and results from simulations
The data generated from each run of the algorithm is rather large and cannot be easily distributed via GitHub. Please contact the authors if you would like to receive a copy of the output from the simulations. Otherwise, you should be able to reproduce all runs yourself within a few hours by running the Docker container.

## License
This source code is distributed under the GPLv3 license with (c) Johan Dahlin 2018 and comes with ABSOLUTELY NO WARRANTY. See the file `LICENSE` for more information.
``` python
###############################################################################
#    Constructing Metropolis-Hastings proposals using damped BFGS updates
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################
```