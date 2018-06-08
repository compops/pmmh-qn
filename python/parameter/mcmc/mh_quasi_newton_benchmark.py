import time
import copy
import warnings

import numpy as np

from scipy.stats import multivariate_normal as mvn

from helpers.cov_matrix import correct_hessian
from helpers.distributions import product_multivariate_gaussian as pmvn
from parameter.mcmc.mh_quasi_newton import QuasiNewtonMetropolisHastings

class QuasiNewtonMetropolisHastingsBenchmark(QuasiNewtonMetropolisHastings):
    current_iter = 0
    start_time = 0
    time_offset = 0
    run_time = 0
    time_per_iter = 0
    no_hessians_corrected = 0
    iter_hessians_corrected = []


    def __init__(self, model, settings=None):
        super().__init__(model, settings)
        self.type = 'qmh_benchmark'
        self.alg_type = 'qmh_benchmark'

    def _estimate_state(self, estimator, proposed_state, state_history):
        # Get adapted step sizes (if there are any) otherwise use fixed
        if 'adapted_step_size' in proposed_state:
            step_size_gradient = 0.5 * proposed_state['adapted_step_size']**2
            step_size_hessian = proposed_state['adapted_step_size']**2
        else:
            step_size_gradient = 0.5 * self.settings['step_size_gradient']**2
            step_size_hessian = self.settings['step_size_hessian']**2

        # Check if there is an empirical estimate of the Hessian to use
        # as the fallback
        if type(self.emp_hessian) is np.ndarray:
            alt_hess = self.emp_hessian
        else:
            alt_hess = self.settings['hess_corr_fallback']

        hess_corr = self.settings['hess_corr_method']

        ## Run the smoother to get likelihood and state estimate
        warnings.filterwarnings("error")
        try:
            self.model.store_free_params(proposed_state['params_free'])
            log_jacobian = self.model.log_jacobian()
            _, log_prior = self.model.log_prior()
        except:
            print("MH-QN-benchmark: Storing parameters failed...")
            return False

        if self.settings['correlated_rvs'] and estimator.alg_type is not 'kalman':
            rvs = {'rvs': proposed_state['rvs']}
            smoother_completed = estimator.smoother(self.model, compute_hessian=True, rvs=rvs)
        else:
            smoother_completed = estimator.smoother(self.model, compute_hessian=True)

        if not smoother_completed:
            print("MH-QN-benchmark: Smoother failed...")
            return False

        log_like = estimator.results['log_like']
        state_trajectory = estimator.results['state_trajectory']
        grad = estimator.results['gradient_internal']
        hess = np.linalg.inv(estimator.results['hessian_internal'])
        grad_copy = np.array(grad, copy=True)

        # Run benchmark with different Quasi-Newton proposals
        memory_length_vector = (5, 10, 15, 20, 25, 30, 35, 40)
        error_bfgs_fro = []
        error_ls_fro = []
        error_sr1_fro = []

        if self.current_iter > self.settings['memory_length']:
            for i, memory_length in enumerate(memory_length_vector):
                params_diffs, grads_diffs = self._qn_compute_diffs(state_history, memory_length=memory_length)

                init_hessian = self._qn_init_hessian(grad)
                init_hessian_ls = state_history

                hess_bfgs, _ = self._qn_bfgs(params_diffs, grads_diffs, init_hessian)
                hess_ls, _ = self._qn_ls(params_diffs, grads_diffs, init_hessian_ls)
                hess_sr1, _ = self._qn_sr1(params_diffs, grads_diffs, init_hessian)

                hess_direct = np.linalg.inv(estimator.results['hessian_internal_noprior'])

                error_bfgs_fro.append(np.linalg.norm(hess_direct - hess_bfgs, 'fro'))
                error_ls_fro.append(np.linalg.norm(hess_direct - hess_ls, 'fro'))
                error_sr1_fro.append(np.linalg.norm(hess_direct - hess_sr1, 'fro'))

        hess, fixed_hess = correct_hessian(hess, alt_hess, hess_corr, verbose=False)

        grad = estimator.results['gradient_internal']
        nat_grad = hess @ grad
        if np.isfinite(step_size_hessian) and np.isfinite(step_size_gradient):
            output_hess = np.array(hess, copy=True) * step_size_hessian
            output_nat_grad = np.array(nat_grad, copy=True) * step_size_gradient
        else:
            print("MH-QN-benchmark: Gradient or Hessian not finite.")
            return False

        proposed_state.update({'params': self.model.get_params()})
        proposed_state.update({'state_trajectory': state_trajectory})
        proposed_state.update({'log_like': log_like})
        proposed_state.update({'log_jacobian': log_jacobian})
        proposed_state.update({'log_prior': log_prior})
        proposed_state.update({'log_target': log_prior + log_like})
        proposed_state.update({'gradient': grad_copy})
        proposed_state.update({'nat_gradient': output_nat_grad})
        proposed_state.update({'hessian': output_hess})
        proposed_state.update({'hessian_corrected': fixed_hess})
        proposed_state.update({'error_bfgs_fro': np.array(error_bfgs_fro)})
        proposed_state.update({'error_ls_fro': np.array(error_ls_fro)})
        proposed_state.update({'error_sr1_fro': np.array(error_sr1_fro)})
        return True

    def _qn_compute_diffs(self, state_history, memory_length):
        no_params = self.no_params_to_estimate

        # Extract parameters, gradients and log-target for the current length
        # of memory
        params = np.zeros((memory_length - 1, no_params))
        grads = np.zeros((memory_length - 1, no_params))
        losses = np.zeros((memory_length - 1, 1))
        j = 0
        for i in range(self.current_iter - memory_length + 1, self.current_iter):
            params[j, :] = state_history[i]['params_free'].flatten()
            grads[j, :] = state_history[i]['gradient'].flatten()
            losses[j, :] = float(state_history[i]['log_target'])
            losses[j, :] += float(state_history[i]['log_prior'])
            j += 1

        # Sort and compute differences
        idx = np.argsort(losses.flatten())
        params = params[idx, :]
        grads = grads[idx, :]

        params_diffs = np.zeros((memory_length - 2, no_params))
        grads_diffs = np.zeros((memory_length - 2, no_params))
        for i in range(len(idx) - 1):
            params_diffs[i, :]= params[i + 1, :] - params[i, :]
            grads_diffs[i, :] = grads[i + 1, :] - grads[i, :]

        return params_diffs, grads_diffs