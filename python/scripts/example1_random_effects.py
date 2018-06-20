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

"""Script for reproducing example 1 in paper."""
import copy
import numpy as np
import scripts.helper_random_effects as mh


def main(seed_offset=0):
    """Runs the experiment."""
    hessian_estimate = np.diag((0.01, 1.48))
    hessian_guess = 0.1 * np.eye(2)
    initial_params = (1.0, 0.2)

    mh_settings = {'no_iters': 30000,
                   'no_burnin_iters': 3000,
                   'adapt_step_size': True,
                   'adapt_step_size_initial': 0.5,
                   'adapt_step_size_rate': 0.5,
                   'adapt_step_size_target': 0.6,
                   'initial_params': initial_params,
                   'min_no_samples_hessian_estimate': 2,
                   'no_iters_between_progress_reports': 1000,
                   'correlated_rvs': True,
                   'correlated_rvs_sigma': 0.1,
                   }

    correlated_rvs_sigma_vector = np.round(np.arange(0.00, 1.05, 0.05), 2)
    memory_length_vector = np.arange(5, 45, 5)

    for i, sigmau in enumerate(correlated_rvs_sigma_vector):
        for j, mem_length in enumerate(memory_length_vector):
                folder_tag = 'sigmau-' + str(sigmau) + '/' + 'M-' + str(mem_length)

                new_mh_settings = copy.deepcopy(mh_settings)
                new_mh_settings.update({'adapt_step_size_initial': 0.1})
                new_mh_settings.update({'memory_length': mem_length})
                new_mh_settings.update({'accept_first_iterations': mem_length})
                new_mh_settings.update({'hessian': hessian_guess})
                new_mh_settings.update({'hess_corr_fallback': hessian_guess})
                new_mh_settings.update({'hess_corr_method': 'flip'})
                new_mh_settings.update({'correlated_rvs_sigma': sigmau})
                mh.run('qmh',
                        mh_settings=new_mh_settings,
                        seed_offset=seed_offset,
                        alg_type='bfgs',
                        folder_tag=folder_tag)

                new_mh_settings = copy.deepcopy(mh_settings)
                new_mh_settings.update({'adapt_step_size_initial': 0.25})
                new_mh_settings.update({'memory_length': mem_length})
                new_mh_settings.update({'accept_first_iterations': mem_length})
                new_mh_settings.update({'hessian': hessian_guess})
                new_mh_settings.update({'hess_corr_fallback': hessian_guess})
                new_mh_settings.update({'hess_corr_method': 'flip'})
                new_mh_settings.update({'sr1_trust_region': True})
                new_mh_settings.update({'sr1_trust_region_scale': 1.0})
                new_mh_settings.update({'sr1_trust_region_cov': hessian_guess})
                new_mh_settings.update({'correlated_rvs_sigma': sigmau})
                mh.run('qmh',
                        mh_settings=new_mh_settings,
                        seed_offset=seed_offset,
                        alg_type='sr1',
                        folder_tag=folder_tag)

                new_mh_settings = copy.deepcopy(mh_settings)
                new_mh_settings.update({'adapt_step_size_initial': 0.15})
                new_mh_settings.update({'memory_length': mem_length})
                new_mh_settings.update({'accept_first_iterations': mem_length})
                new_mh_settings.update({'hessian': hessian_guess})
                new_mh_settings.update({'hess_corr_fallback': hessian_guess})
                new_mh_settings.update({'hess_corr_method': 'flip'})
                new_mh_settings.update({'ls_regularisation_parameter': 0.1})
                new_mh_settings.update({'correlated_rvs_sigma': sigmau})
                mh.run('qmh',
                        mh_settings=new_mh_settings,
                        seed_offset=seed_offset,
                        alg_type='ls',
                        folder_tag=folder_tag)

    return None


if __name__ == '__main__':
    main()
