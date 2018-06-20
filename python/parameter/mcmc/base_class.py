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
"""Base class for a MCMC sampler."""

import time
import copy

import numpy as np

from parameter.base_parameter_inference import BaseParameterInference

from parameter.mcmc.helpers import estimate_hessian
from parameter.mcmc.output import print_greeting
from parameter.mcmc.output import print_progress_report
from parameter.mcmc.output import save_to_file
from parameter.mcmc.output import plot


class MarkovChainMonteCarlo(BaseParameterInference):
    """Base class for a MCMC sampler."""
    current_iter = 0
    start_time = 0
    time_offset = 0
    run_time = 0
    time_per_iter = 0
    state_history = None
    settings = None

    def __init__(self, model, settings):
        """Constructor augumented by the different sub-classes."""
        self.model = model
        self.no_params_to_estimate = self.model.no_params_to_estimate
        self._set_settings(settings)

    def run(self, estimator, verbose=False):
        """Run the MCMC algorithm."""
        no_iters = self.settings['no_iters']
        no_burnin_iters = self.settings['no_burnin_iters']
        assert no_iters > no_burnin_iters
        self.current_iter = 0

        proposed_state = {}
        current_state = {}
        state_history = {}

        self.adapted_step_sizes = []
        self.estimator_type = estimator.alg_type
        self.dim_rvs = estimator.dim_rvs
        self.verbose = verbose
        self.start_time = time.time()
        self.time_offset = 0.0
        self._print_greeting(estimator)
        proposed_state = self._initialise_sampler(estimator, state_history)

        for i in range(len(state_history), no_iters):
            self.current_iter = i
            current_state = self._get_current_state(state_history)
            self._initialise_iteration(state_history)

            proposed_state = self._adapt_step_size(
                current_state, proposed_state)

            if not self._propose_parameters(current_state, proposed_state):
                proposed_state.update({'accept_prob': 0.0})
                print("Error in proposing parameters.")
                self._reject_proposed_state(
                    current_state, proposed_state, state_history, reset_params=True)
                continue

            self._propose_rvs(current_state, proposed_state)

            if not self._estimate_state(estimator, proposed_state, state_history):
                print("Error in state estimation.")
                proposed_state.update({'accept_prob': 0.0})
                self._reject_proposed_state(
                    current_state, proposed_state, state_history, reset_params=True)
                continue

            if not self._compute_accept_prob(current_state, proposed_state):
                print("Error in accept probability computation.")
                proposed_state.update({'accept_prob': 0.0})
                self._reject_proposed_state(
                    current_state, proposed_state, state_history, reset_params=True)
                continue

            if 'accept_first_iterations' in self.settings:
                if i < self.settings['accept_first_iterations']:
                    proposed_state.update({'accept_prob': 1.0})

            checked_proposed = self._check_proposed_state(proposed_state)

            if np.random.random(1) < proposed_state['accept_prob'] and checked_proposed:
                self._accept_proposed_state(proposed_state, state_history)
            else:
                self._reject_proposed_state(
                    current_state, proposed_state, state_history)

            self._delete_corr_rvs_history(state_history)
            self._print_progress_report(proposed_state, state_history)

        self.run_time = time.time() - self.start_time
        self.time_per_iter = (self.run_time - self.time_offset) / no_iters

        self.state_history = state_history
        print("Run of MCMC algorithm complete...")
        print("It took: {:.2f} seconds to run this code.".format(
            self.run_time))

    def _set_settings(self, settings):
        """Sets the settings for the algorithm using the settings dict.
           Note that this method is overloaded by each type of algorithm."""
        raise NotImplementedError

    def _initialise_sampler(self, estimator, state_history):
        """Initialises the algorithm, same for all versions."""
        estimator.settings['estimate_gradient'] = self.using_gradients
        estimator.settings['estimate_hessian'] = self.using_hessians
        self.model.store_params(self.settings['initial_params'])

        proposed_state = {}
        proposed_state.update({'params_free': self.model.get_free_params()})
        proposed_state.update({'params_prop': self.model.get_params()})

        if self.estimator_type is not 'kalman' and self.settings['correlated_rvs']:
            no_particles = estimator.settings['no_particles']
            no_obs = estimator.settings['no_obs']
            rvs = np.random.normal(size=self.dim_rvs)
            proposed_state.update({'rvs': rvs})
            rvs = np.random.normal(size=self.dim_rvs)
            proposed_state.update({'rvs_aux': rvs})

        if self._estimate_state(estimator, proposed_state, state_history):
            proposed_state.update({'accept_prob': 1.0})
            proposed_state.update({'accepted': 1.0})

            if 'hessian' in self.settings:
                hess = self.settings['hessian'] * \
                    self.settings['step_size_hessian']**2
            elif 'hess_corr_fallback' in self.settings:
                hess = self.settings['hess_corr_fallback'] * \
                    self.settings['step_size_hessian']**2
            else:
                raise NameError(
                    "An initial guess of the Hessian must be supplied as the setting hessian or hess_corr_fallback.")

            proposed_state.update({'hessian': hess})
            state_history.update({0: proposed_state})
        else:
            raise NameError("MCMC: Initialisation failed, check parameters.")

        return proposed_state

    def _get_current_state(self, state_history):
        """Returns the current state. Differs for the quasi-Newton methods."""
        current_state = copy.deepcopy(state_history[(self.current_iter - 1)])

        # Checks
        if not np.isfinite(current_state['params']).all():
            raise ValueError("Non-finite parameters have been accepted.")

        if 'nat_gradient' in current_state:
            if not np.isfinite(current_state['nat_gradient']).all():
                raise ValueError(
                    "Non-finite gradients have been accepted. If using QN proposal try to decrease the size of the setting hess_corr_fallback.")

        return current_state

    def _propose_rvs(self, current_state, proposed_state):
        """Proposes correlated random variables for the particle filter."""

        if self.estimator_type is 'kalman':
            return None

        if self.settings['correlated_rvs']:
            # Crank-Nicolson proposal for the rvs
            sigma_u = self.settings['correlated_rvs_sigma']
            if 'rvs' in current_state:
                rvs = current_state['rvs']
                mean = np.sqrt(1.0 - sigma_u**2) * rvs
                prop_rvs = mean + sigma_u * np.random.normal(size=self.dim_rvs)
                proposed_state['rvs'] = prop_rvs

            if 'rvs_aux' in current_state:
                rvs_aux = current_state['rvs_aux']
                mean = np.sqrt(1.0 - sigma_u**2) * rvs_aux
                prop_aux_rvs = mean + sigma_u * \
                    np.random.normal(size=self.dim_rvs)
                proposed_state['rvs_aux'] = prop_aux_rvs

        else:
            # Rvs are generated by the particle filter internally
            proposed_state['rvs'] = None
            proposed_state['rvs_aux'] = None

    def _check_proposed_state(self, proposed_state):
        """Checks that the proposed state is finite."""

        if not np.isfinite(proposed_state['params']).all():
            return False

        if not np.isfinite(proposed_state['params_free']).all():
            return False

        if 'nat_gradient' in proposed_state:
            if not np.isfinite(proposed_state['nat_gradient']).all():
                return False

            if not np.isfinite(proposed_state['hessian']).all():
                return False

        return True

    def _accept_proposed_state(self, proposed_state, state_history):
        """Stores the candidate state as it is accepted."""
        i = self.current_iter
        state_history.update({i: copy.deepcopy(proposed_state)})
        state_history[i].update({'accepted': 1.0})
        state_history[i].update({'params_prop': proposed_state['params']})

    def _reject_proposed_state(self, current_state, proposed_state, state_history, reset_params=False):
        """Stores the current state as the candidate state is rejected."""
        i = self.current_iter
        state_history.update({i: copy.deepcopy(current_state)})
        state_history[i].update({'accepted': 0.0})
        if 'params' in proposed_state:
            state_history[i].update({'params_prop': proposed_state['params']})

        if 'accept_first_iterations' in self.settings and reset_params:
            if i < 4 * self.settings['accept_first_iterations']:
                self.model.store_params(self.settings['initial_params'])
                state_history[i].update(
                    {'params_free': self.model.get_free_params()})
                state_history[i].update(
                    {'params_prop': self.model.get_params()})
                print("Reset Markov chain due to problems in log-target estimator.")

    def _delete_corr_rvs_history(self, state_history):
        """Removes history for correlated rvs to save memory."""
        i = self.current_iter
        if 'memory_length' in self.settings:
            offset = self.settings['memory_length']
        else:
            offset = 1

        if self.settings['correlated_rvs'] and i >= offset:
            del(state_history[i - offset]['rvs'])
            del(state_history[i - offset]['rvs_aux'])

    def _adapt_step_size(self, current_state, proposed_state):
        accept_prob = np.copy(current_state['accept_prob'])
        new_proposed_state = {}
        offset = 0

        if not self.settings['adapt_step_size']:
            self.adapted_step_sizes.append(0.0)
            return new_proposed_state

        if 'accept_first_iterations' in self.settings:
            offset = self.settings['accept_first_iterations'] - 1
            if self.current_iter < self.settings['accept_first_iterations']:
                new_proposed_state.update(
                    {'adapted_step_size': self.settings['adapt_step_size_initial']})
                self.adapted_step_sizes.append(
                    self.settings['adapt_step_size_initial'])
                return new_proposed_state

        alpha = self.settings['adapt_step_size_rate']
        target_rate = self.settings['adapt_step_size_target']
        adapt_rate = (self.current_iter - offset)**(-alpha)

        if 'adapted_step_size' in current_state:
            cur_step_size = np.log(current_state['adapted_step_size'])
        else:
            cur_step_size = np.log(self.settings['adapt_step_size_initial'])

        diff = np.min((accept_prob - target_rate, 1.0))
        new_step_size = float(np.exp(cur_step_size + adapt_rate * diff))

        if np.isfinite(new_step_size) and new_step_size > 0.0 and new_step_size < 10.0:
            new_proposed_state.update({'adapted_step_size': new_step_size})
            self.adapted_step_sizes.append(new_step_size)
        else:
            new_proposed_state.update(
                {'adapted_step_size': np.exp(cur_step_size)})
            self.adapted_step_sizes.append(cur_step_size)

        return new_proposed_state

    def _initialise_iteration(self, state_history):
        """Runs as the first step in each iteration of the MCMC algorithm."""
        pass

    def _propose_parameters(self, current_state, proposed_state):
        """Proposes the parameters. Different for each version."""
        raise NotImplementedError

    def _estimate_state(self, estimator, proposed_state, state_history):
        """State estimator to give an estimate of the target, gradients and
           Hessians. Different for each version."""
        raise NotImplementedError

    def _compute_accept_prob(self, current_state, proposed_state):
        """Computes acceptance probabilities. Different for each version."""
        raise NotImplementedError

    estimate_hessian = estimate_hessian
    save_to_file = save_to_file
    plot = plot
    _print_greeting = print_greeting
    _print_progress_report = print_progress_report
