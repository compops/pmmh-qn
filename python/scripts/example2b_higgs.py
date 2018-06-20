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

"""Script for reproducing example 2 in paper."""
import copy
import numpy as np
import scripts.helper_higgs as mh

def main(data, seed_offset=0, use_all_data=False):
    """Runs the experiment."""
    no_regressors = 22
    initial_params = np.zeros(no_regressors)
    hessian_guess = 1e-3 * np.eye(no_regressors)
    hessian_estimate = np.diag((0.0224175946, 0.0026805251, 0.0008686880,
                                0.0008478166, 0.0026625858, 0.0008161794,
                                0.0051553222, 0.0008272772, 0.0008740550,
                                0.0011803427, 0.0049723278, 0.0009728194,
                                0.0008679611, 0.0014321208, 0.0048149487,
                                0.0008054738, 0.0008019967, 0.0010184522,
                                0.0035317832, 0.0008455518, 0.0007616436,
                                0.0006226389))

    mh_settings = {'no_iters': 30000,
                   'no_burnin_iters': 3000,
                   'adapt_step_size': True,
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
                   'memory_length': 30,
                   'accept_first_iterations': 30,
                   'hessian': hessian_guess,
                   'hess_corr_fallback': hessian_guess,
                   'hess_corr_method': 'flip'
                   }


    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'hessian': hessian_estimate})
    new_mh_settings.update({'adapt_step_size': False})
    new_mh_settings.update({'step_size_hessian': 0.5 * 2.562 / np.sqrt(no_regressors)})
    mh.run('mh0',
           mh_settings=new_mh_settings,
           data=data,
           use_all_data=use_all_data,
           seed_offset=seed_offset)


    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'hessian': hessian_estimate})
    new_mh_settings.update({'adapt_step_size': False})
    new_mh_settings.update({'step_size_gradient': 0.5})
    new_mh_settings.update({'step_size_hessian': 0.5})
    mh.run('mh2',
           mh_settings=new_mh_settings,
           data=data,
           use_all_data=use_all_data,
           seed_offset=seed_offset)


    new_mh_settings.update({'adapt_step_size_target': 0.2})
    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_initial': 0.1})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           data=data,
           use_all_data=use_all_data,
           seed_offset=seed_offset,
           alg_type='bfgs')


    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.3})
    new_mh_settings.update({'sr1_trust_region': True})
    new_mh_settings.update({'sr1_trust_region_scale': 1.0})
    new_mh_settings.update({'sr1_trust_region_cov': hessian_guess})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           data=data,
           use_all_data=use_all_data,
           seed_offset=seed_offset,
           alg_type='sr1')


    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.2})
    new_mh_settings.update({'ls_regularisation_parameter': 0.1})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           data=data,
           use_all_data=use_all_data,
           seed_offset=seed_offset,
           alg_type='ls')


    return None

