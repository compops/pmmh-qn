"""Direct methods."""
import numpy as np

from scipy.stats import norm

from state.base_state_inference import BaseStateInference
from state.direct.subsampling import stratified
from state.direct.subsampling import get_settings


class DirectComputation(BaseStateInference):
    """Direct log-likelihood and gradient computations."""

    def __init__(self, model, new_settings=None, use_all_data=False):
        self.alg_type = 'direct'
        self.settings = {'no_particles': 100,
                         'no_obs': model.no_obs,
                         'use_all_data': False
                         }
        if new_settings:
            self.settings.update(new_settings)
        if use_all_data:
            self.settings.update({'no_particles': model.no_obs})
            self.settings.update({'use_all_data': True})
        self._init_direct_computation(model)
        self.results = {}

    def filter(self, model, **kwargs):
        """Direct log-likelihood and gradient computation."""

        if not self.settings['use_all_data']:
            if 'rvs' in kwargs:
                rvs = np.sort(norm.cdf(kwargs['rvs']['rvs'].flatten()))
                idx = np.array(stratified(rvs)).astype(int)
            else:
                idx = np.random.choice(model.no_obs, self.no_particles)
        else:
            idx = np.arange(model.no_obs).astype(int)

        try:
            results = model.get_loglike_gradient(idx=idx)
            self.results.update({'filt_state_est': 0.0})
            self.results.update({'state_trajectory': 0.0})
            self.results.update({'log_like': float(results['log_like'])})
            return True
        except Exception as e:
            # Smoother did not run properly, return False
            print("Error in computation of likelihood.")
            print(e)
            return False

    def smoother(self, model, compute_hessian=False, **kwargs):
        """Direct log-likelihood and gradient computation."""
        if not self.settings['use_all_data']:
            if 'rvs' in kwargs:
                rvs = np.sort(norm.cdf(kwargs['rvs']['rvs'].flatten()))
                idx = np.array(stratified(rvs)).astype(int)
            else:
                idx = np.random.choice(model.no_obs, self.no_particles)
        else:
            idx = np.arange(model.no_obs).astype(int)

        # try:
        results = model.get_loglike_gradient(compute_gradient=True, compute_hessian=compute_hessian, idx=idx)

        self.results.update({'filt_state_est': 0.0})
        self.results.update({'state_trajectory': 0.0})
        self.results.update({'log_like': float(results['log_like'])})

        gradient_internal = np.array(results['gradient_internal'])
        gradient_internal += model.log_prior_gradient()

        hessian_internal = np.array(results['hessian_internal'])
        self.results.update({'hessian_internal_noprior': np.copy(hessian_internal)})
        hessian_internal += model.log_prior_hessian()

        self.results.update({'gradient_internal': gradient_internal})
        self.results.update({'gradient': np.array(results['gradient'])})
        self.results.update({'hessian_internal': hessian_internal})
        self.results.update({'hessian': np.array(results['hessian'])})
        return True

        # except Exception as e:
        #     # Smoother did not run properly, return False
        #     print("Error in computation of likelihood and its gradient.")
        #     print(e)
        #     return False

    def _init_direct_computation(self, model):
        no_obs, no_particles = get_settings()
        assert no_obs == model.no_obs

        self.name = "Direct log-likelihood and gradient computations for " + model.short_name + " model"
        self.alg_type = 'direct'
        self.no_obs = no_obs
        self.no_particles = no_particles
        self.dim_rvs = no_particles
        self.settings.update({'no_obs': no_obs, 'no_particles': no_particles})

        print("-------------------------------------------------------------------")
        print("Direct log-likelihood and gradient computations for " + model.short_name + " initialised.")
        print("")
        print("The settings are as follows: ")
        for key in self.settings:
            print("{}: {}".format(key, self.settings[key]))
        print("")
        print("-------------------------------------------------------------------")
        print("")