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

"""Script for reproducing example 2a in paper."""
import copy
import numpy as np
import scripts.helper_higgs as mh

def main(data, seed_offset=0, use_all_data=False):
    """Runs the experiment."""
    no_regressors = 22
    initial_params = np.zeros(no_regressors)
    hessian_guess = 1e-3 * np.eye(no_regressors)

    mh_settings = {'no_iters': 30000,
                   'no_burnin_iters': 3000,
                   'adapt_step_size': False,
                   'adapt_step_size_initial': 0.1,
                   'adapt_step_size_rate': 0.5,
                   'adapt_step_size_target': 0.6,
                   'step_size_gradient': 0.5,
                   'step_size_hessian': 0.5,
                   'initial_params': initial_params,
                   'min_no_samples_hessian_estimate': len(initial_params) + 2,
                   'no_iters_between_progress_reports': 1000,
                   'correlated_rvs': True,
                   'correlated_rvs_sigma': 0.05,
                   'remove_overflow_iterations': False,
                   'memory_length': 50,
                   'accept_first_iterations': 50,
                   'hessian': hessian_guess,
                   'hess_corr_fallback': hessian_guess,
                   'hess_corr_method': 'flip',
                   'ls_regularisation_parameter': 0.1,
                   'sr1_trust_region': True,
                   'sr1_trust_region_scale': 1.0,
                   'sr1_trust_region_cov': hessian_guess
                   }

    new_mh_settings = copy.deepcopy(mh_settings)
    mh.run('qmhb',
           mh_settings=new_mh_settings,
           data=data,
           use_all_data=use_all_data,
           seed_offset=seed_offset
           )

    return None

