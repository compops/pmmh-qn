"""System model class for a random effects model."""
import numpy as np
from scipy.stats import norm
from scipy.stats import poisson

from models.base_model import BaseModel
from helpers.distributions import normal
from helpers.distributions import gamma
from helpers.distributions import cauchy
from helpers.distributions import multivariate_gaussian

class RandomEffectsModel(BaseModel):
    """ System model class for a random effects model.

        Encodes the model with the parameterisation:

        alpha_i ~ N(mu, sigma^2)
        y_it | alpha_i ~ N(alpha_i, 1)

        The parameters of the model are (mu, sigma)

        Furthermore, the inference model is reparameterised to enable all
        parameters to be unrestricted (can assume any real value) in the
        inference step. Hence, the reparameterised model is given by

        sigma = exp(psi)

        where (psi) now as assume any real value. This results in that
        this transformation needs to be taken into account when computing
        gradients as well as in the Jacobian.

    """

    def __init__(self):
        self.name = "Random effects model."
        self.file_prefix = "random_effects"
        self.short_name = "random_effects"

        self.params = {'mu': 0.0,
                       'sigma': 1.0
                       }

        self.free_params = {'mu': 0.0,
                            'sigma': 0.0
                           }

        self.params_prior = {'mu': (normal, 0.0, 1.0),
                             'sigma': (cauchy, 0.0, 1.0),
                             }

        self.no_params = len(self.params)
        self.initial_state = []
        self.no_obs = None
        self.states = []
        self.inputs = []
        self.obs = []
        self.params_to_estimate_idx = []
        self.no_params_to_estimate = 0
        self.params_to_estimate = []
        self.true_params = []

    def generate_data(self, no_obs):
        self.no_obs = no_obs

        # Generate random effects
        alpha = self.params['sigma'] * np.random.normal(size=no_obs)
        alpha += self.params['mu']

        y = np.zeros(no_obs)
        for t in range(no_obs):
            y[t] = alpha[t] + np.random.normal()

        self.obs = y
        self.states = alpha

    def importance_sampling(self, settings, rvs=None):
        raise NotImplementedError("Use the Cython implementation instead.")

    def check_parameters(self):
        """" Checks if parameters satisfies hard constraints on the parameters.

                Returns:
                    Boolean to indicate if the current parameters results in
                    a stable system and obey the constraints on their values.

        """
        if self.params['sigma'] < 0.0:
            parameters_are_okey = False
        else:
            parameters_are_okey = True
        return True

    def log_prior_gradient(self):
        """ Returns the logarithm of the prior distribution.

            Returns:
                First value: a dict with an entry for each parameter.
                Second value: the sum of the log-prior for all variables.

        """
        gradients = super(RandomEffectsModel, self).log_prior_gradient()

        gradients['sigma'] *= self.params['sigma']
        return gradients

    def log_prior_hessian(self):
        """ The Hessian of the logarithm of the prior.

            Returns:
                A dict with an entry for each parameter.

        """
        gradients = super(RandomEffectsModel, self).log_prior_gradient()
        hessians = super(RandomEffectsModel, self).log_prior_hessian()

        gradients['sigma'] *= self.params['sigma']
        hessians['sigma'] *= self.params['sigma']
        hessians['sigma'] += gradients['sigma']
        return hessians

    def transform_params_to_free(self):
        """ Computes and store the values of the reparameterised parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the reparameterised parameters are computed by applying
            the transformation to the current standard parameters stored in
            the model object.

        """
        for param in self.params:
            if param is 'sigma':
                self.free_params.update({param: np.log(self.params[param])})
            else:
                self.free_params.update({param: self.params[param]})

    def transform_params_from_free(self):
        """ Computes and store the values of the standard parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the standard parameters are computed by applying
            the transformation to the current reparameterised parameters stored
            in the model object.

        """
        for param in self.params:
            if param is 'sigma':
                self.params.update({param: np.exp(self.free_params[param])})
            else:
                self.params.update({param: self.free_params[param]})

    def log_jacobian(self):
        """ Computes the sum of the log-Jacobian.

            These Jacobians are dictated by the transformations for the model.
            See the docstring for the model class for more information.

            Returns:
                the sum of the logarithm of the Jacobian of the parameter
                transformation for the parameters under inference as listed
                in params_to_estimate.

        """
        try:
            jacobian = {}
            jacobian.update({'mu': 0.0})
            jacobian.update({'sigma': np.log(self.params['sigma'])})
        except RuntimeWarning:
            return -np.inf

        return self._compile_log_jacobian(jacobian)