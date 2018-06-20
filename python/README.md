# Python code

## Installation
This code was developed using Anaconda 3 and Python 3.6.3. To satisfy these requirements, please download Anaconda from https://www.anaconda.com/ for your OS. Then install the libraries used in the paper by executing in the root folder of the repo
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

## Reproducing the results in the paper
The results in the paper can be reproduced by running the scripts found in the folder `scripts/`. Here, we discuss each of the three examples in details and provide some additional supplementary details, which are not covered in the paper. The results from each script is saved in the folder `results/` under sub-folders corresponding to the three different examples.

Examples 1 and 2 are repeated using different random seeds 25 times in a Monte Carlo simulation. The simplest way to execute these is to call the script `run_script.sh`, which will run all the experiments (note that this will take at least a few hours). Another way to execute a single experiment is to call

``` bash
python run_script.py experiment_number
```

where `experiment_number` is 1, 2 or 3. Note that this will still mean that 25 experiments are run for examples 1 and 2. To run only one repetition for a single experiment, change the code in `run_script.py` by removing the for-loop.

### Example 1: Selection correlation and memory lengths

### Example 2: Sub-sampling for logistic regression model

### Example 3: Particle filtering for stochastic volatility model


## File structure
An overview of the file structure of the code base is found below.

* **helpers/** contains helpers for models, parameterisations, distributions, file management and connection to databases. These should not require any alterations.
* **models/** contains the different models used in the paper. It is here you need to add your own models (see below) if you want to use the code for some other problem.
* **parameter/** contains the MH algorithm and quasi-Newton methods for estimating the Hessian.
* **scripts/** contains helper scripts for the examples in the paper.
* **state/** contains the Cython code subsampling via stratified resampling, importance sampling, bootstrap particle filter and fixed-lag particle smoother.

## Modifying the code for other models
This code is fairly general and can be used for inference in most models with latent states (e.g., state space models) expressed by densities and with a scalar state.

The models are defined by files in models/. To implement a new model you can alter the existing models and re-define the functions `generate_initial_state`, `generate_state`, `evaluate_state`,  `generate_obs`, `evaluate_obs` and `check_parameters`. The names of these methods and their arguments should be self-explanatory. Furthermore, the Cython code for importance sampling and particle filter/smoothing should be altered. The main things to change is how the propagation and weighting of particles is carried out as well as the gradients of the log joint distribution of states and measurements.

In the paper, all model parameters are unrestricted and can assume any real value in the MH algorithm. This is enabled by reparametersing the model, which is always recommended for MH algorithms. This results in that the reparameterisation must be encoded in the methods `transform_params_to_free` and `transform_params_from_free`, where free parameters are the unrestricted versions. This also introduces a Jacobian factor into the acceptance probability encoded by `log_jacobian` as well as extra terms in the gradients and Hessians of both the log joint distribution of states and observations as well as the log priors.

### Calibration of user settings
Furthermore, some alterations are probably required to the settings used in the quasi-Newton algorithm such as initial guess of the Hessian, a standard step length, memory length, etc.

Please, let me know if you need any help with this and I will try my best to sort it out.
