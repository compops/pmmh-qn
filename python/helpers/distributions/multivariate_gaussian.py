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

"""Helpers for the multivariate Gaussian distribution."""
import numpy as np
from scipy.stats import multivariate_normal


def pdf(parm, mean, cov_matrix):
    """ Computes the pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            cov_matrix: covariance matrix

        Returns:
            A scalar with the value of the pdf.

        Note that if mean and cov_matrix are scalar, then an independent
        Gaussian is assumed with equal mean and variances.
    """
    if type(mean) is float and type(cov_matrix) is float:
        n = len(parm)
        mu = mean * np.ones(n)
        Sigma = cov_matrix * np.eye(n)
        parm = np.array(parm).flatten()
        return multivariate_normal.pdf(parm, mu, Sigma)
    else:
        return multivariate_normal.pdf(parm, mean, cov_matrix)


def logpdf(parm, mean, cov_matrix):
    """ Computes the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean vector
            cov_matrix: covariance matrix

        Returns:
            A scalar with the value of the log-pdf.

        Note that if mean and cov_matrix are scalar, then an independent
        Gaussian is assumed with equal mean and variances.
    """
    if type(mean) is float and type(cov_matrix) is float:
        n = len(parm)
        mu = mean * np.ones(n)
        Sigma = cov_matrix * np.eye(n)
        parm = np.array(parm).flatten()
        return multivariate_normal.logpdf(parm, mu, Sigma)
    else:
        return my_logpdf(parm, mean, cov_matrix)


def my_logpdf(parm, mean, cov_matrix):
    """ Computes the log-pdf of the multivariate Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean vector
            cov_matrix: covariance matrix

        Returns:
            A scalar with the value of the pdf.

    """
    no_dimensions = len(cov_matrix)

    norm_coeff = no_dimensions * np.log(2.0 * np.pi)
    norm_coeff += np.linalg.slogdet(cov_matrix)[1]
    error = parm - mean

    quad_term = np.dot(error, np.linalg.pinv(cov_matrix))
    quad_term = np.dot(quad_term, error.transpose())
    return -0.5 * (norm_coeff + quad_term)


def logpdf_gradient(parm, mean, cov_matrix):
    """ Computes the gradient of the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            cov_matrix: covariance matrix

        Returns:
            A vector with the value of the gradient of the log-pdf.

        Note that if mean and cov_matrix are scalar, then an independent
        Gaussian is assumed with equal mean and variances.
    """
    if type(mean) is float and type(cov_matrix) is float:
        n = len(parm)
        parm = np.array(parm).flatten()
        return -np.ones(n) * (mean - parm) / cov_matrix**2
    else:
        raise NotImplementedError


def logpdf_hessian(parm, mean, cov_matrix):
    """ Computes the Hessian of the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            cov_matrix: covariance matrix

        Returns:
            A matrix with the value of the Hessian of the log-pdf.

        Note that if mean and cov_matrix are scalar, then an independent
        Gaussian is assumed with equal mean and variances.
    """
    if type(mean) is float and type(cov_matrix) is float:
        n = len(parm)
        return -np.eye(n) / cov_matrix**2
    else:
        raise NotImplementedError


def rv(mean, cov_matrix, print_warnings=False):
    no_param = len(mean)

    if no_param == 1:
        return mean + np.sqrt(np.abs(cov_matrix)) * np.random.normal()
    else:
        try:
            return np.random.multivariate_normal(mean, cov_matrix)
        except RuntimeWarning:
            if print_warnings:
                print("Warning raised in np.random.multivariate_normal " +
                      "so using Cholesky to generate random variables.")

            cov_matrix_root = np.linalg.cholesky(cov_matrix)
            rv = np.random.multivariate_normal(np.zeros(no_param),
                                               np.eye(no_param))
            return mean + np.matmul(cov_matrix_root, rv)
