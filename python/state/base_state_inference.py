###############################################################################
#    Constructing Metropolis-Hastings proposals using damped BFGS updates
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

"""The base state inference object."""

import warnings
warnings.filterwarnings("error")
import numpy as np

class BaseStateInference(object):
    name = []
    settings = {}
    results = {}
    model = {}

    no_obs = 0
    log_like = []
    gradient = []
    gradient_internal = []
    hessian_internal = []

    def __repr__(self):
        self.name

    def _estimate_gradient_and_hessian(self, model):
        """Inserts gradients and Hessian of the log-priors into the estimates
        of the gradient and Hessian of the log-likelihood."""

        if 'log_joint_gradient_estimate' in self.results:
            gradient_estimate = self.results['log_joint_gradient_estimate']
            estimate_gradients = True
        else:
            estimate_gradients = False

        if 'log_joint_hessian_estimate' in self.results:
            hessian_estimate = self.results['log_joint_hessian_estimate']
            idx = model.params_to_estimate_idx
            self.results.update({'hessian_internal_noprior': np.copy(hessian_estimate[np.ix_(idx, idx)])})
            estimate_gradients = True
            estimate_hessian = True
        else:
            estimate_hessian = False

        if not estimate_gradients and not estimate_hessian:
            return True

        # Add the log-prior derivatives
        gradient = {}
        gradient_internal = []
        log_prior_gradient = model.log_prior_gradient()

        if type(model.params) is dict:
            i = 0
            for param in model.params.keys():
                gradient_estimate[i] += log_prior_gradient[param]
                if param in model.params_to_estimate:
                    gradient.update({param: gradient_estimate[i]})
                    gradient_internal.append(gradient_estimate[i])
                i += 1
        else:
            for i in range(model.no_params):
                gradient_estimate[i] += log_prior_gradient[i]
                if i in model.params_to_estimate:
                    gradient.update({i: gradient_estimate[i]})
                    gradient_internal.append(gradient_estimate[i])

        if estimate_hessian:
            log_prior_hessian = model.log_prior_hessian()
            if type(model.params) is dict:
                i = 0
                for param in model.params.keys():
                    hessian_estimate[i, i] -= log_prior_hessian[param]
                    i += 1
            else:
                for i in range(model.no_params):
                    hessian_estimate[i, i] -= log_prior_hessian[i]

        # Compile output
        self.results.update({'gradient_internal': np.array(gradient_internal)})
        self.results.update({'gradient': gradient})

        if estimate_hessian:
            idx = model.params_to_estimate_idx
            self.results.update({'hessian_internal': np.array(hessian_estimate[np.ix_(idx, idx)])})
            self.results.update({'hessian_internal_prior': np.array(hessian_estimate[np.ix_(idx, idx)])})

        return True