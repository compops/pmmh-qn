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
"""First-order Metropolis-Hastings (MALA) using gradient information
    in the proposal distribution for the parameters."""

import warnings
import numpy as np
from scipy.stats import multivariate_normal as mvn

from parameter.mcmc.base_class import MarkovChainMonteCarlo


class FirstOrderMetropolisHastings(MarkovChainMonteCarlo):
    """First-order Metropolis-Hastings (MALA) using gradient information
       in the proposal distribution for the parameters."""

    current_iter = 0
    start_time = 0
    time_offset = 0
    run_time = 0
    time_per_iter = 0

    def __init__(self, model, settings=None):
        super().__init__(model, settings)

        self.type = 'mh1'
        self.alg_type = 'mh1'
        self.name = "First-order Metropolis-Hastings (MALA)"
        self.using_gradients = True
        self.using_hessians = False
        self.hessians_computed_by = None
        model.using_gradients = self.using_gradients
        model.using_hessians = self.using_hessians

    def _set_settings(self, new_settings):
        self.settings = {'no_iters': 1000,
                         'no_burnin_iters': 300,
                         'adapt_step_size': False,
                         'adapt_step_size_initial': 0.1,
                         'adapt_step_size_rate': 2.0 / 3.0,
                         'adapt_step_size_target': 0.25,
                         'correlated_rvs': False,
                         'correlated_rvs_sigma': 0.5,
                         'accept_first_iterations': 100,
                         'step_size_gradient': 0.1,
                         'step_size_hessian': 0.1,
                         'hessian': np.eye(self.no_params_to_estimate) * 0.01**2,
                         'initial_params': np.random.uniform(size=self.no_params_to_estimate),
                         'no_iters_between_progress_reports': 100,
                         'remove_overflow_iterations': True
                         }
        self.settings.update(new_settings)

    def _propose_parameters(self, current_state, proposed_state):
        step_size = 0.5 * self.settings['step_size_gradient']
        curr_params = current_state['params_free']
        curr_grad = step_size * current_state['gradient']
        curr_hess = current_state['hessian']

        prop_params = mvn.rvs(curr_params + curr_grad, curr_hess)
        proposed_state['params_free'] = prop_params

        if np.isfinite(prop_params).all():
            return True
        else:
            return False

    def _estimate_state(self, estimator, proposed_state, state_history):
        # Get adapted step sizes (if there are any) otherwise use fixed
        if 'adapted_step_size' in proposed_state:
            step_size_gradient = 0.5 * proposed_state['adapted_step_size']**2
            step_size_hessian = proposed_state['adapted_step_size']**2
        else:
            step_size_gradient = 0.5 * self.settings['step_size_gradient']**2
            step_size_hessian = self.settings['step_size_hessian']**2

        # Save the current candidate parameters to model
        warnings.filterwarnings("error")
        try:
            self.model.store_free_params(proposed_state['params_free'])
            log_jacobian = self.model.log_jacobian()
            _, log_prior = self.model.log_prior()
        except:
            return False
        hess = self.settings['hessian'] * step_size_hessian

        # Run the smoother to get likelihood and state estimate
        if self.settings['correlated_rvs']:
            rvs = {'rvs': proposed_state['rvs']}
            smoother_completed = estimator.smoother(self.model, rvs=rvs)
        else:
            smoother_completed = estimator.smoother(self.model)

        if not smoother_completed:
            return False

        try:
            log_like = estimator.results['log_like']
            state_trajectory = estimator.results['state_trajectory']
            grad = estimator.results['gradient_internal']
            grad *= step_size_gradient
        except Exception as e:
            print("State estimation failed with error...")
            print(e)
            return False

        proposed_state.update({'params': self.model.get_params()})
        proposed_state.update({'state_trajectory': state_trajectory})
        proposed_state.update({'log_like': log_like})
        proposed_state.update({'log_jacobian': log_jacobian})
        proposed_state.update({'log_prior': log_prior})
        proposed_state.update({'log_target': log_prior + log_like})
        proposed_state.update({'gradient': grad})
        proposed_state.update({'hessian': hess})
        return True

    def _compute_accept_prob(self, current_state, proposed_state):
        current = current_state['params_free']
        current_mean = current + current_state['gradient']
        current_hess = current_state['hessian']

        proposed = proposed_state['params_free']
        proposed_mean = proposed + proposed_state['gradient']
        proposed_hess = proposed_state['hessian']

        try:
            proposed_probability = mvn.logpdf(
                proposed, current_mean, current_hess)
            current_probability = mvn.logpdf(
                current, proposed_mean, proposed_hess)

            tar_diff = proposed_state['log_target'] - \
                current_state['log_target']
            jac_diff = proposed_state['log_jacobian'] - \
                current_state['log_jacobian']
            pro_diff = current_probability - proposed_probability

            accept_prob = np.min((1.0, np.exp(tar_diff + jac_diff + pro_diff)))
        except:
            if self.settings['remove_overflow_iterations']:
                return False
            else:
                proposed_state.update({'accept_prob': 1.0})
                return True

        proposed_state.update({'accept_prob': accept_prob})
        return True
