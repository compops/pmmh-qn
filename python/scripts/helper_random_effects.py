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

import numpy as np

from models.random_effects import RandomEffectsModel
from state.importance_sampling.cython import ImportanceSamplingCython

from parameter.mcmc.mh_zero_order import ZeroOrderMetropolisHastings as MetropolisHastings0
from parameter.mcmc.mh_first_order import FirstOrderMetropolisHastings as MetropolisHastings1
from parameter.mcmc.mh_second_order import SecondOrderMetropolisHastings as MetropolisHastings2
from parameter.mcmc.mh_quasi_newton import QuasiNewtonMetropolisHastings

def run(mh_version, mh_settings, seed_offset=0, alg_type=None, folder_tag=None):

    # Set random seed
    np.random.seed(87655678 + int(seed_offset))

    # System model
    sys_model = RandomEffectsModel()
    no_obs = 100
    sys_model.params['mu'] = 1.0
    sys_model.params['sigma'] = 0.2
    sys_model.generate_data(no_obs=no_obs)

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model()
    print(sys_model)

    # Setup filtering/smoothing algorithm
    smoother = ImportanceSamplingCython(sys_model)

    # Metropolis-Hastings algorithm
    if mh_version is 'mh0':
        sampler = MetropolisHastings0(sys_model, mh_settings)
        file_path = '../results/example1-random-effects/mh0'
    elif mh_version is 'mh1':
        sampler = MetropolisHastings1(sys_model, mh_settings)
        file_path = '../results/example1-random-effects/mh1'
    elif mh_version is 'qmh':
        sampler = QuasiNewtonMetropolisHastings(sys_model, mh_settings, qn_method=alg_type)
        file_path = '../results/example1-random-effects/qmh-' + alg_type
    else:
        raise NameError("Unknown MH method...")

    # Add folder tag if provided
    if folder_tag:
        file_path += '/' + folder_tag

    # Run sampler
    sampler.run(smoother)

    # Save results to file
    if alg_type:
        sim_name = 'example1-' + mh_version + '-' + alg_type + '-' + str(seed_offset)
    else:
        sim_name = 'example1-' + mh_version + '-' + str(seed_offset)
    sampler.save_to_file(file_path=file_path, sim_name=sim_name, sim_desc="")
