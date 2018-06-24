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

"""Script for reproducing example 3 in paper."""
import copy
import numpy as np
import scripts.helper_stochastic_volatility as mh

def main(seed_offset=0):
    """Runs the experiment."""
    initial_params = (2.0, 0.9, 0.4, -0.2)
    hessian_estimate = np.array([[ 0.07878946,  0.01361605, -0.00747067, -0.00202019],
                                 [ 0.01361605,  0.03555009, -0.01868784,  0.00127068],
                                 [-0.00747067, -0.01868784,  0.02052527, -0.00058821],
                                 [-0.00202019,  0.00127068, -0.00058821,  0.00147995]])
    hessian_guess = np.diag((0.01, 0.01, 0.01, 0.001))

    mh_settings = {'no_iters': 30000,
                   'no_burnin_iters': 3000,
                   'adapt_step_size': True,
                   'adapt_step_size_initial': 0.1,
                   'adapt_step_size_rate': 0.5,
                   'adapt_step_size_target': 0.6,
                   'initial_params': initial_params,
                   'no_iters_between_progress_reports': 1000,
                   'correlated_rvs': True,
                   'correlated_rvs_sigma': 0.5,
                   'memory_length': 40,
                   'accept_first_iterations': 40,
                   'hessian': hessian_guess,
                   'hess_corr_fallback': hessian_guess,
                   'hess_corr_method': 'flip'
    }

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'hessian': hessian_guess})
    new_mh_settings.update({'adapt_step_size': False})
    new_mh_settings.update({'step_size_gradient': 1.0})
    new_mh_settings.update({'step_size_hessian': 1.0})
    mh.run('mh2',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset)

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.2})
    new_mh_settings.update({'adapt_step_size_initial': 0.1})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset,
           alg_type='bfgs')

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.3})
    new_mh_settings.update({'sr1_trust_region': True})
    new_mh_settings.update({'sr1_trust_region_scale': 1.0})
    new_mh_settings.update({'sr1_trust_region_cov': hessian_guess})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset,
           alg_type='sr1')

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.2})
    new_mh_settings.update({'ls_regularisation_parameter': 0.1})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset,
           alg_type='ls')

    return None


if __name__ == '__main__':
    main()
