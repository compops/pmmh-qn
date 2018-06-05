import time
import copy
import warnings

import numpy as np

from scipy.stats import multivariate_normal as mvn

from helpers.cov_matrix import correct_hessian
from helpers.distributions import product_multivariate_gaussian as pmvn
from parameter.mcmc.base_class import MarkovChainMonteCarlo

class QuasiNewtonMetropolisHastings(MarkovChainMonteCarlo):
    current_iter = 0
    start_time = 0
    time_offset = 0
    run_time = 0
    time_per_iter = 0
    no_hessians_corrected = 0
    iter_hessians_corrected = []


    def __init__(self, model, settings=None, qn_method='bfgs'):
        super().__init__(model, settings)

        self.type = 'qmh'
        self.alg_type = 'qmh'
        self.qn_method = qn_method
        self.using_gradients = True
        self.using_hessians = True
        self.emp_hessian = None
        model.using_gradients = self.using_gradients
        model.using_hessians = self.using_hessians

        if qn_method is 'bfgs':
            self.name = "Quasi-Newton Metropolis-Hastings using BFGS"
            self.hessians_computed_by = 'quasi-newton-bfgs'
            self.qn_estimator = self._qn_bfgs
        elif qn_method is 'sr1':
            self.name = "Quasi-Newton Metropolis-Hastings using SR1"
            self.hessians_computed_by = 'quasi-newton-sr1'
            self.qn_estimator = self._qn_sr1
        elif qn_method is 'ls':
            self.name = "Quasi-Newton Metropolis-Hastings using LS"
            self.hessians_computed_by = 'quasi-newton-ls'
            self.qn_estimator = self._qn_ls
        else:
            raise NameError("Unknown Quasi-Newton method.")


    def _initialise_sampler(self, estimator, state_history):
        proposed_state = super()._initialise_sampler(estimator, state_history)
        self.no_hessians_corrected = 0
        self.iter_hessians_corrected = []
        return proposed_state


    def _set_settings(self, new_settings):
        self.settings = {'no_iters': 1000,
                         'no_burnin_iters': 300,
                         'adapt_step_size': False,
                         'adapt_step_size_initial': 0.1,
                         'adapt_step_size_rate': 2.0/3.0,
                         'adapt_step_size_target': 0.25,
                         'correlated_rvs': False,
                         'correlated_rvs_sigma': 0.5,
                         'accept_first_iterations': 100,
                         'memory_length': 20,
                         'step_size_gradient': 0.1,
                         'step_size_hessian': 0.1,
                         'hess_corr_fallback': np.eye(self.no_params_to_estimate) * 0.01**2,
                         'hess_corr_method': 'flip',
                         'min_no_samples_hessian_estimate': 2.0 * self.no_params_to_estimate,
                         'initial_params': np.random.uniform(size=self.no_params_to_estimate),
                         'no_iters_between_progress_reports': 100,
                         'ls_regularisation_parameter': 0.1,
                         'ls_help_settings_regularisation_parameter': False,
                         'sr1_skip_limit': 1e-4,
                         'sr1_trust_region': False,
                         'sr1_trust_region_scale': 1.0,
                         'sr1_trust_region_cov': np.eye(self.no_params_to_estimate),
                         'bfgs_curv_cond': 'damped',
                         'initial_hessian_scaling': 0.01,
                         'remove_overflow_iterations': True,
                         'show_overflow_warnings': False
                         }
        self.settings.update(new_settings)
        self.settings.update({'accept_first_iterations': self.settings['memory_length']})

        assert len(self.settings['initial_params']) == self.no_params_to_estimate
        assert self.settings['hess_corr_fallback'].shape[0] == self.no_params_to_estimate
        assert self.settings['hess_corr_fallback'].shape[1] == self.no_params_to_estimate
        assert self.settings['sr1_trust_region_cov'].shape[0] == self.no_params_to_estimate
        assert self.settings['sr1_trust_region_cov'].shape[1] == self.no_params_to_estimate


    def _initialise_algorithm(self):
        self.no_hessians_corrected = 0
        self.iter_hessians_corrected = []
        self.time_offset = 0.0


    def _get_current_state(self, state_history):
        """Returns the current state. Differs for the quasi-Newton methods."""
        if self.current_iter > self.settings['memory_length']:
            mem_length = self.settings['memory_length']
        else:
            mem_length = 1
        current_state = copy.deepcopy(state_history[int(self.current_iter - mem_length)])

        # Checks
        if not np.isfinite(current_state['params']).all():
            raise ValueError("Non-finite parameters have been accepted.")

        if 'nat_gradient' in current_state:
            if not np.isfinite(current_state['nat_gradient']).all():
                raise ValueError("Non-finite gradients have been accepted.")

        return current_state


    def _initialise_iteration(self, state_history):
        # Compute empirical Hessian estimate for hybrid regularisation method
        # Estimate computed using the latter part of the burn-in
        no_burnin_iters = self.settings['no_burnin_iters']
        if self.current_iter == no_burnin_iters:
            trace = np.zeros((int(0.5 * no_burnin_iters), self.no_params_to_estimate))
            for i in range(int(0.5 * no_burnin_iters)):
                j = int(0.5 * no_burnin_iters) + i
                trace[i, :] = state_history[j]['params_free']
            self.emp_hessian = np.cov(trace, rowvar=False)
            print("Iteration: {}. Computed empirical Hessian matrix.".format(self.current_iter))


    def _propose_parameters(self, current_state, proposed_state):
        if type(self.emp_hessian) is np.ndarray:
            sr1_trust_region_cov = self.settings['sr1_trust_region_scale'] * self.emp_hessian
        else:
            sr1_trust_region_cov = self.settings['sr1_trust_region_cov']

        current = current_state['params_free']
        current_mean = current_state['params_free'] + current_state['nat_gradient']

        if self.qn_method is 'sr1' and self.settings['sr1_trust_region']:
            prop_params = pmvn.rvs(current_mean,
                                   current,
                                   current_state['hessian'],
                                   sr1_trust_region_cov)
        else:
            prop_params = mvn.rvs(current_mean,
                                  current_state['hessian'])

        if 'accept_first_iterations' in self.settings\
            and self.current_iter < self.settings['accept_first_iterations']:
            prop_params = mvn.rvs(current, current_state['hessian'])
            print("Used random walk proposal for initial step.")

        proposed_state['params_free'] = prop_params

        if np.isfinite(proposed_state['params_free']).all():
            return True
        else:
            print("")
            print("Iteration: {}.".format(self.current_iter))
            print("current_parm: {}".format(current_state['params']))
            print("current_grad: {}".format(current_state['nat_gradient']))
            print("current_hess {}".format(np.diag(current_state['hessian'])))
            print("")
            return False
            # Removing this line introduces numerical errors for some reason
            # probably due to some error in the compiler.
            print(current_state)
            print(proposed_state)


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
            print("MH-QN: Storing parameters failed...")
            return False

        if self.settings['correlated_rvs'] and estimator.alg_type is not 'kalman':
            rvs = {'rvs': proposed_state['rvs']}
            smoother_completed = estimator.smoother(self.model, rvs=rvs)
        else:
            smoother_completed = estimator.smoother(self.model)

        if not smoother_completed:
            print("MH-QN: Smoother failed...")
            return False

        log_like = estimator.results['log_like']
        state_trajectory = estimator.results['state_trajectory']
        grad = estimator.results['gradient_internal']
        grad_copy = np.array(grad, copy=True)

        # Run Quasi-Newton proposals to try to estimate Hessian
        if self.current_iter > self.settings['memory_length']:
            params_diffs, grads_diffs = self._qn_compute_diffs(state_history)
            if self.qn_method is 'ls':
                init_hessian = state_history
            else:
                init_hessian = self._qn_init_hessian(grad)
            hess, no_samples = self.qn_estimator(params_diffs, grads_diffs, init_hessian)
            using_qn_hessian = True

            # If not enough samples to estimate a good Hessian replace
            # with alt_hessian and no gradients
            if no_samples < int(self.settings['min_no_samples_hessian_estimate']):
                print("Not enough samples ({}) to estimate Hessian using QN.".format(no_samples))
                hess = np.array(alt_hess, copy=True)
                using_qn_hessian = False
        else:
            hess = np.array(alt_hess, copy=True)
            grad = np.zeros(self.no_params_to_estimate)
            no_samples = 0
            using_qn_hessian = False

        proposed_state.update({'hessian_samples': no_samples})
        proposed_state.update({'hessian_estimated': using_qn_hessian})

        # Auxiliary gradient computation
        offset_to_substract = time.time()

        if self.settings['correlated_rvs'] and estimator.alg_type is not 'kalman':
            rvs = {'rvs': proposed_state['rvs_aux']}
            smoother_completed = estimator.smoother(self.model, rvs=rvs)
        else:
            smoother_completed = estimator.smoother(self.model)

        offset_to_substract = time.time() - offset_to_substract

        hess, fixed_hess = correct_hessian(hess, alt_hess, hess_corr, verbose=False)

        if not type(hess) is np.ndarray:
            print("MH-QN: Not a valid Hessian estimate...")
            return False

        grad = estimator.results['gradient_internal']
        nat_grad = hess @ grad
        if np.isfinite(step_size_hessian) and np.isfinite(step_size_gradient):
            output_hess = np.array(hess, copy=True) * step_size_hessian
            output_nat_grad = np.array(nat_grad, copy=True) * step_size_gradient
        else:
            print("MH-QN: Gradient or Hessian not finite.")
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

        self.time_offset += offset_to_substract

        if fixed_hess:
            self.no_hessians_corrected += 1
            self.iter_hessians_corrected.append(self.current_iter)

        return True


    def _compute_accept_prob(self, current_state, proposed_state):
        if self.current_iter < 2.0 * self.settings['memory_length']:
            proposed_state.update({'accept_prob': 1.0})
            return True

        if type(self.emp_hessian) is np.ndarray:
            sr1_trust_region_cov = self.settings['sr1_trust_region_scale'] * self.emp_hessian
        else:
            sr1_trust_region_cov = self.settings['sr1_trust_region_cov']

        try:
            if self.qn_method is 'sr1' and self.settings['sr1_trust_region']:
                current = current_state['params_free']
                current_mean = current + current_state['nat_gradient']

                proposed = proposed_state['params_free']
                proposed_mean = proposed + proposed_state['nat_gradient']

                proposed_probability = pmvn.logpdf(proposed, current_mean,
                                                current, current_state['hessian'],
                                                sr1_trust_region_cov)

                current_probability = pmvn.logpdf(current, proposed_mean,
                                                proposed, proposed_state['hessian'],
                                                sr1_trust_region_cov)
            else:
                current = current_state['params_free']
                proposed = proposed_state['params_free']
                current_mean = current + current_state['nat_gradient']
                current_hess = current_state['hessian']

                proposed_mean = proposed + proposed_state['nat_gradient']
                proposed_hess = proposed_state['hessian']

                proposed_probability = mvn.logpdf(proposed, current_mean, current_hess)
                current_probability = mvn.logpdf(current, proposed_mean, proposed_hess)

            tar_diff = proposed_state['log_target'] - current_state['log_target']
            jac_diff = proposed_state['log_jacobian'] - current_state['log_jacobian']
            pro_diff = current_probability - proposed_probability

            accept_prob = np.min((1.0, np.exp(tar_diff + jac_diff + pro_diff)))

        except Exception as e:
            if self.settings['show_overflow_warnings']:
                current_hess = current_state['hessian']
                proposed_hess = proposed_state['hessian']
                print("")
                print("Iteration: {}. Overflow in accept prob calculation.".format(self.current_iter))
                print("This is probably due to a mismatch in the current and proposed Hessians.")
                print("Diag of current Hessian: {}.".format(np.diag(current_hess)))
                print("Diag of candidate Hessian: {}.".format(np.diag(proposed_hess)))
                print("")
            if self.settings['remove_overflow_iterations']:
                return False
            else:
                proposed_state.update({'accept_prob': 1.0})
                return True

        proposed_state.update({'accept_prob': accept_prob})
        return True


    def _qn_compute_diffs(self, state_history):
        memory_length = self.settings['memory_length']
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


    def _qn_init_hessian(self, prop_grad):
        scaling = self.settings['initial_hessian_scaling']
        ident_mat = np.eye(self.no_params_to_estimate)
        grad_norm = np.linalg.norm(prop_grad, 2)
        if grad_norm > 0.0:
            return ident_mat * scaling / grad_norm
        else:
            return ident_mat * scaling


    def _qn_bfgs(self, params_diffs, grads_diffs, initial):
        curv_cond = self.settings['bfgs_curv_cond']
        estimate = np.array(initial, copy=True)
        ident_mat = np.eye(self.no_params_to_estimate)
        no_grads = params_diffs.shape[0]
        no_samples = 0

        for i in range(no_grads):
            do_update = True

            if curv_cond is 'enforce':
                params_diffs[i] = -params_diffs[i, :]
                if np.dot(params_diffs[i, :], grads_diffs[i, :]) > 0.0:
                    new_grads_diffs = grads_diffs[i, :]
                else:
                    do_update = False

            elif curv_cond is 'damped':
                params_diffs[i] = -params_diffs[i, :]
                try:
                    inverse_hessian = np.linalg.inv(estimate)
                except:
                    print("BFGS: Singular initialisation, using identity instead.")
                    inverse_hessian = np.eye(self.no_params_to_estimate)
                term1 = np.dot(params_diffs[i, :], grads_diffs[i, :])
                term2 = np.dot(params_diffs[i, :], inverse_hessian)
                term2 = np.dot(term2, params_diffs[i, :])

                if term1 > 0.2 * term2:
                    theta = 1.0
                else:
                    if term2 - term1 != 0.0:
                        theta = 0.8 * term2 / (term2 - term1)
                    else:
                        do_update = False
                        theta = 1.0

                grad_guess = np.dot(inverse_hessian, params_diffs[i, :])
                new_grads_diffs = theta * grads_diffs[i, :] + (1.0 - theta) * grad_guess

            elif curv_cond is 'ignore':
                new_grads_diffs = grads_diffs[i, :]

            else:
                raise NameError("BFGS: unknown curv_cond given.")

            if do_update:
                rho = 1.0 / np.dot(new_grads_diffs, params_diffs[i, :])
                if rho < 1.0 and rho > 0.0:
                    term1 = np.outer(params_diffs[i, :], new_grads_diffs)
                    term1 = ident_mat - rho * term1
                    term2 = np.outer(new_grads_diffs, params_diffs[i, :])
                    term2 = ident_mat - rho * term2
                    term3 = rho * np.outer(params_diffs[i, :], params_diffs[i, :])

                    tmp_term1 = np.matmul(term1, estimate)
                    estimate = np.matmul(tmp_term1, term2) + term3

                    no_samples += 1

        if curv_cond is not 'damped':
            estimate *= -1.0

        return estimate, no_samples


    def _qn_sr1(self, params_diffs, grads_diffs, initial):
        no_samples = 0
        skip_limit = self.settings['sr1_skip_limit']
        estimate = np.array(initial, copy=True)
        no_grads = params_diffs.shape[0]

        for i in range(no_grads):
            diff_term = params_diffs[i] - np.dot(estimate, grads_diffs[i])
            term1 = np.abs(np.dot(grads_diffs[i], diff_term))
            term2 = np.linalg.norm(grads_diffs[i], 2)
            term2 *= np.linalg.norm(diff_term, 2)
            term2 *= skip_limit

            if term1 > term2:
                if np.dot(diff_term, grads_diffs[i]) != 0.0:
                    rank1_update = np.outer(diff_term, diff_term)
                    rank1_update /= np.dot(diff_term, grads_diffs[i])
                    estimate += rank1_update
                    no_samples += 1
            else:
                no_samples += 1

        estimate *= -1.0
        return estimate, no_samples

    def _qn_ls(self, params_diffs, grads_diffs, state_history):
        lam = self.settings['ls_regularisation_parameter']
        memory_length = self.settings['memory_length']
        no_params = self.no_params_to_estimate

        if type(self.emp_hessian) is np.ndarray:
            hessian_prior = -self.emp_hessian
        else:
            hessian_prior = -self.settings['hess_corr_fallback']

        term1 = lam * np.eye(no_params) + grads_diffs.T @ grads_diffs
        term2 = lam * hessian_prior + grads_diffs.T @ params_diffs
        estimate = np.linalg.inv(term1) @ term2

        if self.settings['ls_help_settings_regularisation_parameter']:
            print("LS reg term: {}".format(np.diag(grads_diffs.T @ params_diffs)))

        estimate = 0.5 * (estimate + estimate.transpose())
        return -estimate, grads_diffs.shape[0]