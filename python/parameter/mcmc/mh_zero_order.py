import warnings
import numpy as np
from scipy.stats import multivariate_normal as mvn

from parameter.mcmc.base_class import MarkovChainMonteCarlo

class ZeroOrderMetropolisHastings(MarkovChainMonteCarlo):
    current_iter = 0
    start_time = 0
    time_offset = 0
    run_time = 0
    time_per_iter = 0

    def __init__(self, model, settings=None):
        super().__init__(model, settings)

        self.type = 'mh'
        self.alg_type = 'mh0'
        self.name = "Random walk Metropolis-Hastings (Zeroth order)"
        self.using_gradients = False
        self.using_hessians = False
        self.hessians_computed_by = None
        model.using_gradients = self.using_gradients
        model.using_hessians = self.using_hessians

    def _set_settings(self, new_settings):
        self.settings = {'no_iters': 1000,
                         'no_burnin_iters': 300,
                         'adapt_step_size': False,
                         'adapt_step_size_initial': 0.1,
                         'adapt_step_size_rate': 2.0/3.0,
                         'adapt_step_size_target': 0.25,
                         'correlated_rvs': False,
                         'correlated_rvs_sigma': 0.5,
                         'step_size_hessian': 0.1,
                         'hessian': np.eye(self.no_params_to_estimate) * 0.01**2,
                         'initial_params': np.random.uniform(size=self.no_params_to_estimate),
                         'no_iters_between_progress_reports': 100,
                         'remove_overflow_iterations': True
                         }
        self.settings.update(new_settings)


    def _propose_parameters(self, current_state, proposed_state):
        curr_params = current_state['params_free']
        curr_hess = current_state['hessian']

        prop_params = mvn.rvs(curr_params, curr_hess)
        proposed_state['params_free'] = prop_params

        if self.verbose:
            print("prop_params: {}".format(prop_params))

        if np.isfinite(prop_params).all():
            return True
        else:
            return False


    def _estimate_state(self, estimator, proposed_state, state_history):
        # Get adapted step sizes (if there are any) otherwise use fixed
        if 'adapted_step_size' in proposed_state:
            step_size_hessian = proposed_state['adapted_step_size']**2
        else:
            step_size_hessian = self.settings['step_size_hessian']**2

        # Save the current candidate parameters to model
        warnings.filterwarnings("error")
        try:
            self.model.store_free_params(proposed_state['params_free'])
            log_jacobian = self.model.log_jacobian()
            _, log_prior = self.model.log_prior()
        except:
            print("Cannot store parameters, compute Jacobian or prior.")
            return False
        hess = self.settings['hessian'] * step_size_hessian

        # Run the filter to get likelihood and state estimate
        if self.settings['correlated_rvs']:
            rvs = {'rvs': proposed_state['rvs']}
            filter_completed = estimator.filter(self.model, rvs=rvs)
        else:
            filter_completed = estimator.filter(self.model)

        if not filter_completed:
            print("Problems when estimating state using filter.")
            return False

        log_like = estimator.results['log_like']
        state_trajectory = estimator.results['state_trajectory']

        proposed_state.update({'params': self.model.get_params()})
        proposed_state.update({'state_trajectory': state_trajectory})
        proposed_state.update({'log_like': log_like})
        proposed_state.update({'log_jacobian': log_jacobian})
        proposed_state.update({'log_prior': log_prior})
        proposed_state.update({'log_target': log_prior + log_like})
        proposed_state.update({'hessian': hess})

        return True


    def _compute_accept_prob(self, current_state, proposed_state):
        current_mean = current_state['params_free']
        current_hess = current_state['hessian']
        proposed_mean = proposed_state['params_free']
        proposed_hess = proposed_state['hessian']

        try:
            proposed_probability = mvn.logpdf(proposed_mean, current_mean, current_hess)
            current_probability = mvn.logpdf(current_mean, proposed_mean, proposed_hess)

            tar_diff = proposed_state['log_target'] - current_state['log_target']
            jac_diff = proposed_state['log_jacobian'] - current_state['log_jacobian']
            pro_diff = current_probability - proposed_probability

            if self.verbose:
                print("tar_diff: {}".format(tar_diff))
                print("jac_diff: {}".format(jac_diff))
                print("pro_diff: {}".format(pro_diff))

            accept_prob = np.min((1.0, np.exp(tar_diff + jac_diff + pro_diff)))
        except:
            if self.settings['remove_overflow_iterations']:
                return False
            else:
                proposed_state.update({'accept_prob': 1.0})
                return True

        if self.verbose:
            print("accept_prob: {}".format(accept_prob))

        proposed_state.update({'accept_prob': accept_prob})
        return True