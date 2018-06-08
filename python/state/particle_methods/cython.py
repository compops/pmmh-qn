"""Particle methods."""
import numpy as np
from scipy.stats import norm
from state.base_state_inference import BaseStateInference

from state.particle_methods.stochastic_volatility import bpf_sv_corr as c_filter_sv
from state.particle_methods.stochastic_volatility import flps_sv_corr as c_smoother_sv
from state.particle_methods.stochastic_volatility import get_settings as c_get_settings_sv

from state.particle_methods.earthquake import bpf_eq as c_filter_earthquake
from state.particle_methods.earthquake import bpf_eq_corr as c_filter_corr_earthquake
from state.particle_methods.earthquake import flps_eq as c_smoother_earthquake
from state.particle_methods.earthquake import flps_eq_corr as c_smoother_corr_earthquake
from state.particle_methods.earthquake import get_settings as c_get_settings_earthquake

class ParticleMethodsCython(BaseStateInference):
    """Particle methods."""

    def __init__(self, model):
        self.alg_type = 'particle'
        if model.short_name is 'earthquake':
            self.c_filter = c_filter_earthquake
            self.c_filter_corr = c_filter_corr_earthquake
            self.c_smoother = c_smoother_earthquake
            self.c_smoother_corr = c_smoother_corr_earthquake
            self.c_get_settings = c_get_settings_earthquake
        elif model.short_name is 'sv':
            self.c_filter = c_filter_sv
            self.c_filter_corr = c_filter_sv
            self.c_smoother = c_smoother_sv
            self.c_smoother_corr = c_smoother_sv
            self.c_get_settings = c_get_settings_sv
        else:
            raise NameError ("Cython implementation for model missing.")

        self._init_particle_method(model)
        self.results = {}

    def filter(self, model, **kwargs):
        """Bootstrap particle filter using Cython."""
        obs = np.array(model.obs.flatten()).astype(np.float)
        params = model.get_all_params()

        try:
            if 'rvs' in kwargs:
                rvs = kwargs['rvs']['rvs'].flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]
                xf, ll, xtraj = self.c_filter_corr(obs, params=params, rvr=rv_r, rvp=rv_p)
            else:
                rvs = np.random.normal(size=self.dim_rvs).flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]
                xf, ll, xtraj = self.c_filter_corr(obs, params=params, rvr=rv_r, rvp=rv_p)

            self.results.update({'filt_state_est': np.array(xf).flatten()})
            self.results.update({'state_trajectory': np.array(xtraj).flatten()})
            self.results.update({'log_like': float(ll)})
            return True
        except Exception as e:
            # filter did not run properly, return False
            print("Error in Cython code for particle filter.")
            print(e)
            return False

    def smoother(self, model, **kwargs):
        """Fixed-lag particle smoother using Cython."""
        obs = np.array(model.obs.flatten()).astype(np.float)
        params = model.get_all_params()

        try:
            if 'rvs' in kwargs:
                rvs = kwargs['rvs']['rvs'].flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]
                xf, xs, ll, grad, xtraj = self.c_smoother_corr(obs, params=params, rvr=rv_r, rvp=rv_p)
            else:
                rvs = np.random.normal(size=self.dim_rvs).flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]
                xf, ll, xtraj = self.c_filter_corr(obs, params=params, rvr=rv_r, rvp=rv_p)

            # Compute estimate of gradient and Hessian
            if model.using_gradients or model.using_hessians:
                grad = np.array(grad).reshape((model.no_params, model.no_obs+1))
                grad[np.isinf(grad)] = 0.0
                grad[np.isnan(grad)] = 0.0
                grad_est = np.nansum(grad, axis=1)

            if model.using_hessians:
                part1 = np.mat(grad).transpose()
                part1 = np.dot(np.mat(grad), part1)
                part2 = np.mat(grad_est)
                part2 = np.dot(np.mat(grad_est).transpose(), part2)
                hessian_est = part1 - part2 / model.no_obs

            self.results.update({'filt_state_est': np.array(xf).flatten()})
            self.results.update({'state_trajectory': np.array(xtraj).flatten()})
            self.results.update({'smo_state_est': np.array(xs).flatten()})
            self.results.update({'log_like': float(ll)})

            if model.using_gradients or model.using_hessians:
                self.results.update({'log_joint_gradient_estimate': grad_est})
            if model.using_hessians:
                self.results.update({'log_joint_hessian_estimate': hessian_est})

            if self._estimate_gradient_and_hessian(model):
                return True
            else:
                # Gradient or Hessian estimates are complex, inf or nan
                return False
        except Exception as e:
            # Smoother did not run properly, return False
            print("Error in Cython code for particle smoother.")
            print(e)
            return False

    def _init_particle_method(self, model):
        no_obs, no_particles, fixed_lag = self.c_get_settings()
        assert no_obs == model.no_obs + 1

        self.name = "Particle method (Cython) for " + model.short_name + " model"
        self.alg_type = 'particle'
        self.settings = {'no_particles': no_particles,
                         'no_obs': no_obs,
                         'resampling_method': 'systematic',
                         'fixed_lag': fixed_lag,
                         'initial_state': 0.0,
                         'generate_initial_state': True,
                         'estimate_gradient': True,
                         'estimate_hessian': True
                         }
        self.no_obs = no_obs
        self.no_particles = no_particles
        self.dim_rvs = (no_obs, no_particles + 1)

        print("-------------------------------------------------------------------")
        print("Cython particle smoothing implementation for " + model.short_name + " initialised.")
        print("")
        print("The settings are as follows: ")
        for key in self.settings:
            print("{}: {}".format(key, self.settings[key]))
        print("")
        print("-------------------------------------------------------------------")
        print("")
