###############################################################################
#    Correlated pseudo-marginal Metropolis-Hastings using
#    quasi-Newton proposals
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

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import os.path
import urllib.request

#from lightning.classification import SGDClassifier
from helpers.file_system import ensure_dir

from models.logistic_regression import LogisticRegressionModel
from state.direct.standard import DirectComputation

from parameter.mcmc.mh_zero_order import ZeroOrderMetropolisHastings as MetropolisHastings0
from parameter.mcmc.mh_first_order import FirstOrderMetropolisHastings as MetropolisHastings1
from parameter.mcmc.mh_second_order import SecondOrderMetropolisHastings as MetropolisHastings2
from parameter.mcmc.mh_quasi_newton import QuasiNewtonMetropolisHastings
from parameter.mcmc.mh_quasi_newton_benchmark import QuasiNewtonMetropolisHastingsBenchmark


def run(mh_version, mh_settings, data, use_all_data=False, seed_offset=0, alg_type=None):
    # Set random seed
    np.random.seed(87655678 + int(seed_offset))

    # System model
    sys_model = LogisticRegressionModel()
    sys_model.load_data_object(data, add_intercept=True)
    sys_model.params = np.zeros(sys_model.no_params)

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model()

    # Filter and smoother
    dc_settings = {'no_particles': 1000}
    dc = DirectComputation(
        sys_model, new_settings=dc_settings, use_all_data=use_all_data)

    # Metropolis-Hastings algorithm
    if mh_version is 'mh0':
        sampler = MetropolisHastings0(sys_model, mh_settings)
        file_path = '../results/example2-higgs/mh0'
    elif mh_version is 'mh1':
        sampler = MetropolisHastings1(sys_model, mh_settings)
        file_path = '../results/example2-higgs/mh1'
    elif mh_version is 'mh2':
        sampler = MetropolisHastings2(sys_model, mh_settings)
        file_path = '../results/example2-higgs/mh2'
    elif mh_version is 'qmh':
        sampler = QuasiNewtonMetropolisHastings(
            sys_model, mh_settings, qn_method=alg_type)
        file_path = '../results/example2-higgs/qmh-' + alg_type
    elif mh_version is 'qmhb':
        sampler = QuasiNewtonMetropolisHastingsBenchmark(
            sys_model, mh_settings)
        file_path = '../results/example2-higgs/qmhb'
    else:
        raise NameError("Unknown MH method...")

    # Run sampler
    sampler.run(dc)
    print(np.diag(sampler.estimate_hessian()))

    # Save results to file
    if alg_type:
        sim_name = 'example2-' + mh_version + \
            '-' + alg_type + '-' + str(seed_offset)
    else:
        sim_name = 'example2-' + mh_version + '-' + str(seed_offset)
    sampler.save_to_file(file_path=file_path, sim_name=sim_name, sim_desc="")


def load_data(file_path='.', subset=None):
    """Loads the Higgs data."""

    x = np.load(file_path + 'higgs_x.npy')[:, :21]
    y = np.load(file_path + 'higgs_y.npy')

    if subset:
        x = x[0:subset, :]
        y = y[0:subset]
        print("Using a subset of the data set.")

    data = {'x': x, 'y': y}
    print("Higgs data loaded...")
    print("{} observations and {} covariates.".format(x.shape[0], x.shape[1]))
    return data


def get_data(file_path=".", subset=None):
    """Reformats the Higgs data."""

    if os.path.isfile(file_path + 'HIGGS.csv.gz') and os.path.isfile(file_path + "higgs_x.npy") and os.path.isfile(file_path + "higgs_y.npy"):
        return None

    # Get the data
    file_name = file_path + 'HIGGS.csv.gz'

    if not os.path.isfile(file_name):
        print("No data found so downloading. About 3 GB so this might take some time.")
        ensure_dir(file_name)
        urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', file_name)
    else:
        print("Found data so skipping download.")

    data = np.loadtxt(file_name, delimiter=",")
    print("Data loaded to memory.")

    x = data[:, 1:22]
    y = data[:, 0].flatten()
    one_column = np.ones((x.shape[0], 1))
    x = np.hstack((one_column, x))

    if subset:
        x_subset = x[0:subset, :]
        y_subset = y[0:subset]
    else:
        x_subset = x
        y_subset = y

    # Set classifier options.
    #print("Running SGD algorithm.")
    #clf = SGDClassifier(penalty="l2", loss="log", max_iter=1000, alpha=1e-2)
    #print("SGD done.")

    # Train the model.
    #clf.fit(x_subset, y_subset)

    #print(clf.coef_)
    #print(clf.coef_[0][1])
    #print(clf.coef_[0][17])

    print("Saving data to file.")
    ensure_dir(file_path + "higgs_x")
    np.save(file_path + "higgs_x", x)
    np.save(file_path + "higgs_y", y)
