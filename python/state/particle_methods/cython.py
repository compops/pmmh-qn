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
"""Particle methods."""

import numpy as np
from scipy.stats import norm
from state.base_state_inference import BaseStateInference

from state.particle_methods.stochastic_volatility import bpf_sv_corr as c_filter_sv
from state.particle_methods.stochastic_volatility import flps_sv_corr as c_smoother_sv
from state.particle_methods.stochastic_volatility import get_settings as c_get_settings_sv


class ParticleMethodsCython(BaseStateInference):
    """Particle methods."""

    def __init__(self, model):
        self.alg_type = 'particle'
        if model.short_name is 'sv':
            self.c_filter = c_filter_sv
            self.c_filter_corr = c_filter_sv
            self.c_smoother = c_smoother_sv
            self.c_smoother_corr = c_smoother_sv
            self.c_get_settings = c_get_settings_sv
        else:
            raise NameError("Cython implementation for model missing.")

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
                xf, ll, xtraj = self.c_filter_corr(
                    obs, params=params, rvr=rv_r, rvp=rv_p)
            else:
                rvs = np.random.normal(size=self.dim_rvs).flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]
                xf, ll, xtraj = self.c_filter_corr(
                    obs, params=params, rvr=rv_r, rvp=rv_p)

            self.results.update({'filt_state_est': np.array(xf).flatten()})
            self.results.update(
                {'state_trajectory': np.array(xtraj).flatten()})
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

        if model.using_hessians:
            hessian_flag = 1
        else:
            hessian_flag = 0

        try:
            if 'rvs' in kwargs:
                rvs = kwargs['rvs']['rvs'].flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]
            else:
                rvs = np.random.normal(size=self.dim_rvs).flatten()
                rv_r = norm.cdf(rvs[0:len(obs)]).flatten()
                rv_p = rvs[len(obs):]

            xf, xs, ll, grad, xtraj, hess1, hess2 = self.c_smoother_corr(
                obs, params=params, rvr=rv_r, rvp=rv_p, compute_hessian=hessian_flag)

            # Compute estimate of gradient and Hessian
            if model.using_gradients or model.using_hessians:
                grad = np.array(grad).reshape(
                    (model.no_params, model.no_obs + 1))
                grad[np.isinf(grad)] = 0.0
                grad[np.isnan(grad)] = 0.0
                grad_est = np.nansum(grad, axis=1)

            if model.using_hessians:
                part1 = np.inner(grad_est, grad_est)
                part2 = np.array(hess1).reshape(
                    (model.no_params, model.no_params))
                part2 += np.array(hess2).reshape((model.no_params,
                                                  model.no_params))
                hessian_est = part1 - part2

            self.results.update({'filt_state_est': np.array(xf).flatten()})
            self.results.update(
                {'state_trajectory': np.array(xtraj).flatten()})
            self.results.update({'smo_state_est': np.array(xs).flatten()})
            self.results.update({'log_like': float(ll)})

            if model.using_gradients or model.using_hessians:
                self.results.update({'log_joint_gradient_estimate': grad_est})
            if model.using_hessians:
                self.results.update(
                    {'log_joint_hessian_estimate': -hessian_est})

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

        self.name = "Particle method (Cython) for " + \
            model.short_name + " model"
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
        print("Cython particle smoothing implementation for " +
              model.short_name + " initialised.")
        print("")
        print("The settings are as follows: ")
        for key in self.settings:
            print("{}: {}".format(key, self.settings[key]))
        print("")
        print("-------------------------------------------------------------------")
        print("")
