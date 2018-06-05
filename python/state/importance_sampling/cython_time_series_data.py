"""Importance sampling methods."""
import numpy as np
from scipy.stats import norm
from state.base_state_inference import BaseStateInference

from state.importance_sampling.random_effects import get_settings as c_get_settings_random
from state.importance_sampling.random_effects import importance_discrete as c_filter_random


class ImportanceSamplingCython(BaseStateInference):
    """Importance sampling methods."""

    def __init__(self, model):
        self.alg_type = 'importance'
        if model.short_name is 'random_effects':
            self.c_filter = c_filter_random
            self.c_get_settings = c_get_settings_random
        else:
            raise NameError ("Cython implementation for model missing.")

        self._init_importance_sampler(model)
        self.results = {}

    def filter(self, model, **kwargs):
        """Importance sampling using Cython."""
        obs = np.array(model.obs)
        no_obs = model.no_obs
        params = model.get_all_params()

        try:
            if 'rvs' in kwargs:
                rvs = kwargs['rvs']['rvs']
                rv_r = norm.cdf(rvs[:, 0][0])
                rv_p = rvs[:, 1:].flatten()
            else:
                rv_r = np.random.uniform()
                rv_p = np.random.normal(size=(self.no_obs, self.no_particles)).flatten()

            xf, ll, xtraj, _ = self.c_filter(obs, params=params, rvr=rv_r, rvp=rv_p)
            self.results.update({'filt_state_est': np.array(xf).flatten()})
            self.results.update({'state_trajectory': np.array(xtraj).flatten()})
            self.results.update({'log_like': float(ll)})
            return True

        except Exception as e:
            # filter did not run properly, return False
            print("Error in Cython code for importance sampler filter.")
            print(e)
            return False

    def smoother(self, model, **kwargs):
        """Importance sampling using Cython."""
        obs = np.array(model.obs)
        no_obs = model.no_obs
        params = model.get_all_params()

        try:
            if 'rvs' in kwargs:
                rvs = kwargs['rvs']['rvs']
                rv_r = norm.cdf(rvs[:, 0][0])
                rv_p = rvs[:, 1:].flatten()
            else:
                rv_r = np.random.uniform()
                rv_p = np.random.normal(size=(self.no_obs, self.no_particles)).flatten()

            xf, ll, xtraj, grad = self.c_filter(obs, params=params, rvr=rv_r, rvp=rv_p)
            self.results.update({'filt_state_est': np.array(xf).flatten()})
            self.results.update({'state_trajectory': np.array(xtraj).flatten()})
            self.results.update({'log_like': float(ll)})
            self.results.update({'log_joint_gradient_estimate': np.array(grad).flatten()})
            if self._estimate_gradient_and_hessian(model):
                return True
            else:
                # Gradient or Hessian estimates are complex, inf or nan
                return False
        except Exception as e:
            # filter did not run properly, return False
            print("Error in Cython code for importance sampler.")
            print(e)
            return False

    def _init_importance_sampler(self, model):
        no_obs, no_particles = self.c_get_settings()
        assert no_obs == model.no_obs

        self.name = "Importance sampling (Cython) for " + model.short_name + " model"
        self.alg_type = 'particle'
        self.settings = {'no_particles': no_particles,
                         'no_obs': no_obs,
                         'estimate_gradient': False
                         }
        self.no_obs = no_obs
        self.no_particles = no_particles
        self.dim_rvs = (no_obs, no_particles + 1)

        print("-------------------------------------------------------------------")
        print("Cython importance sampling implementation for " + model.short_name + " initialised.")
        print("")
        print("The settings are as follows: ")
        for key in self.settings:
            print("{}: {}".format(key, self.settings[key]))
        print("")
        print("-------------------------------------------------------------------")
        print("")