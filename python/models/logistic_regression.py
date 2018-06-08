"""System model class for a logistic regression."""
import copy
import numpy as np

from scipy.stats import norm

from models.base_model import BaseModel
from helpers.distributions import multivariate_gaussian
from helpers.distributions.logit import logit


class LogisticRegressionModel(BaseModel):
    """ System model class for a logistic regression.

        Encodes the model with the parameterisation:

        y | X  ~ logit(beta' * X)

        The parameters of the model are (beta)

    """

    def __init__(self, no_regressors=None):
        self.name = "Logistic regression model."
        self.file_prefix = "logistic_regression"
        self.short_name = "logistic"
        self.no_regressors = no_regressors

        if no_regressors:
            self.params = np.zeros(no_regressors)
            self.free_params = np.zeros(no_regressors)
            self.no_params = no_regressors
        else:
            self.params = {}
            self.free_params = {}
            self.no_params = 0

        self.initial_state = []
        self.no_obs = []
        self.states = []
        self.inputs = []
        self.obs = []
        self.params_to_estimate_idx = []
        self.no_params_to_estimate = 0
        self.params_to_estimate = []
        self.true_params = []

    def load_data_object(self, data, add_intercept=False):
        x = data['x']
        y = data['y']

        if add_intercept:
            one_column = np.ones((x.shape[0], 1))
            x = np.hstack((one_column, x))

        self.no_obs = x.shape[0]
        self.no_params = x.shape[1]
        self.no_regressors = x.shape[1]

        self.params = np.zeros(self.no_params)
        self.free_params = np.zeros(self.no_params)
        self.obs = y.flatten()
        self.regressors = x

    def generate_data(self, no_obs):
        """ Generates observations and regressors from the model.

                Inputs:
                    no_obs: number of observations to simulate (positive integer)

                Returns:
                    Boolean to indicate if the current parameters results in
                    a stable system and obey the constraints on their values.

        """
        self.no_obs = no_obs
        beta = self.params

        # Generate regressors
        x = np.random.normal(size=(no_obs, self.no_regressors))

        # Generate observations
        eta = logit(np.sum(beta * x, axis=1))
        y = np.random.binomial(1, eta)

        self.obs = y
        self.regressors = x

    def get_loglike_gradient(self, compute_gradient=False, compute_hessian=False, idx=None):
        """ Computes the log-likelihood and gradient of the data.

                Inputs:
                    idx: optional vector of indices used to extract a subset.

                Returns:
                    A dictionary with the log-likelihood (log_like) and the
                    gradient of the log-likelihood wrt. theta (gradient).

        """
        np.seterr(over='ignore')
        np.seterr(divide='ignore')

        beta = self.params
        if type(idx) is type(None):
            x = self.regressors
            y = self.obs
        else:
            x = self.regressors[idx, :]
            y = self.obs[idx]

        # Compute the latent variable
        eta = 1.0 / (1.0 + np.exp(-1.0 * np.sum(beta * x, axis=1)))

        # Compute the likelihood
        eta_1 = np.log(eta)
        eta_0 = np.log(1.0 - eta)
        eta_1[np.isinf(eta_1)] = 0.0
        eta_0[np.isinf(eta_0)] = 0.0

        log_like = np.sum(y * eta_1 + (1.0 - y) * eta_0)

        # Compute the gradient
        if compute_gradient:
            grad_1 = x.T / (1.0 + np.exp(np.sum(beta * x, axis=1)))
            grad_0 = -x.T / (1.0 + np.exp(-1.0 * np.sum(beta * x, axis=1)))

            gradient = np.sum(y * grad_1 + (1.0 - y) * grad_0, axis=1)
            gradient_internal = gradient[self.params_to_estimate_idx]

        else:
            gradient = np.zeros(len(beta))
            gradient_internal = np.zeros(len(beta))

        # Compute Hessian
        if compute_hessian:
            hess = np.zeros((len(beta), len(beta)))
            scale_0 = -(1.0 + np.exp(-np.sum(beta * x, axis=1)))**(-2)
            scale_0 *= np.exp(-np.sum(beta * x, axis=1))
            scale_1 = -(1.0 + np.exp(np.sum(beta * x, axis=1)))**(-2)
            scale_1 *= np.exp(np.sum(beta * x, axis=1))

            outer_product = np.einsum('ij...,i...->ij...', x, x)
            hess = y * scale_1 + (1.0 - y) * scale_0
            hess = np.sum(hess * outer_product.T, axis=2)
            hessian_internal = hess[0:self.no_params_to_estimate, 0:self.no_params_to_estimate]

        else:
            hess = np.zeros(len(beta))
            hessian_internal = np.zeros(len(beta))

        return {'log_like': log_like,
                'gradient': gradient,
                'gradient_internal': gradient_internal,
                'hessian': hess,
                'hessian_internal': hessian_internal
                }

    def check_parameters(self):
        """" Checks if parameters satisfies hard constraints on the parameters.

                Returns:
                    Boolean to indicate if the current parameters results in
                    a stable system and obey the constraints on their values.

        """
        return True

    def log_prior(self):
        """ Returns the logarithm of the prior distribution.

            Returns:
                First value: a dict with an entry for each parameter.
                Second value: the sum of the log-prior for all variables.

        """
        grad = multivariate_gaussian.logpdf(self.params, 0.0, 1.0)
        return {}, grad

    def log_prior_gradient(self):
        """ Returns the logarithm of the prior distribution.

            Returns:
                First value: a dict with an entry for each parameter.
                Second value: the sum of the log-prior for all variables.

        """
        grad = multivariate_gaussian.logpdf_gradient(self.params[0:self.no_params_to_estimate], 0.0, 1.0)
        return grad

    def log_prior_hessian(self):
        """ Returns the logarithm of the prior distribution.

            Returns:
                First value: a dict with an entry for each parameter.
                Second value: the sum of the log-prior for all variables.

        """
        hess = multivariate_gaussian.logpdf_hessian(self.params[0:self.no_params_to_estimate], 0.0, 1.0)
        return np.diag(hess)

    def transform_params_to_free(self):
        """ Computes and store the values of the reparameterised parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the reparameterised parameters are computed by applying
            the transformation to the current standard parameters stored in
            the model object.

        """
        self.free_params = np.copy(self.params)

    def transform_params_from_free(self):
        """ Computes and store the values of the standard parameters.

            These transformations are dictated directly from the model. See
            the docstring for the model class for more information. The
            values of the standard parameters are computed by applying
            the transformation to the current reparameterised parameters stored
            in the model object.

        """
        self.params = np.copy(self.free_params)

    def log_jacobian(self):
        """ Computes the sum of the log-Jacobian.

            These Jacobians are dictated by the transformations for the model.
            See the docstring for the model class for more information.

            Returns:
                the sum of the logarithm of the Jacobian of the parameter
                transformation for the parameters under inference as listed
                in params_to_estimate.

        """
        return 0.0

    def store_free_params(self, new_params):
        """ Stores reparameterised parameters to the model.

            Stores new reparameterised (unrestricted) values of the parameters
            to the model object. The restricted parameters are updated accordingly.
            Only the parameters used for inference are required. The remaining
            parameters are copied from the trueParams attribute.

            Args:
                new_params: an array with the new reparameterised parameters. The
                            order must be as in the list params_to_estimate.

            Returns:
            Nothing.

        """
        self.params = copy.deepcopy(self.true_params)
        self.transform_params_to_free()
        self.free_params[self.params_to_estimate_idx] = new_params
        self.transform_params_from_free()

    def store_params(self, new_params):
        """ Stores (restricted) parameters to the model.

            Stores new values of the parameters to the model object. The
            reparameterised (unrestricted) parameters are updated accordingly.
            Only the parameters used for inference are required. The remaining
            parameters are copied from the trueParams attribute.

            Args:
                new_params: an array with the new parameters. The order must be
                            the same as in the list params_to_estimate.

            Returns:
            Nothing.

        """
        self.params = copy.deepcopy(self.true_params)
        self.params[self.params_to_estimate_idx] = new_params
        self.transform_params_to_free()

    def get_free_params(self):
        """ Returns the reparameterised parameters under inference in the model.

            Returns:
            An array with the current reparameterised values for the parameters
            under inference in the model. The order is the same as in the list
            params_to_estimate.

        """
        return np.array(self.free_params[self.params_to_estimate_idx])

    def get_params(self):
        """ Returns the parameters under inference in the model.

            Returns:
            An array with the current values for the parameters under inference
            in the model. The order is the same as in the list params_to_estimate.

        """
        return np.array(self.params[self.params_to_estimate_idx])

    def get_all_params(self):
        """ Returns all the parameters in the model.

            Returns:
            An array with the current values of all parameters in the model.

        """
        return np.array(self.params)

    def create_inference_model(self, params_to_estimate=None):
        """ Transforms a model object into an inference object.

            Adds additional information into a system model to enable it to be
            used for inference. This information includes the parameters to
            estimate in the model.

            Args:
                params_to_estimate: list of parameters to estimate. For example
                                    params_to_estimate = ('mu', 'phi').

            Returns:
            Nothing.

        """

        self.model_type = "Inference model"

        if isinstance(params_to_estimate, type(None)):
            self.no_params_to_estimate = self.no_regressors
            self.params_to_estimate = range(self.no_regressors)
            self.params_to_estimate_idx = np.arange(self.no_regressors).astype(int)
        else:
            if type(params_to_estimate) is tuple or type(params_to_estimate) is np.ndarray:
                self.no_params_to_estimate = len(params_to_estimate)
                self.params_to_estimate = params_to_estimate
                self.params_to_estimate_idx = np.asarray(params_to_estimate)
            else:
                self.no_params_to_estimate = 1
                self.params_to_estimate = [params_to_estimate]
                self.params_to_estimate_idx = [params_to_estimate]

    def fix_true_params(self):
        """ Creates a copy of the true parameters into the model object. """
        self.true_params = copy.deepcopy(self.params)
