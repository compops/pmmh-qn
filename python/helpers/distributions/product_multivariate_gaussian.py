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

"""Helpers for the product of two multivariate Gaussian distributions."""
import numpy as np
from scipy.stats import multivariate_normal as mvn


def compute_statistics(mean1, mean2, cov_matrix1, cov_matrix2):
    """Computes the statistics for the product Gaussian."""
    inverse_term = np.linalg.inv(cov_matrix1 + cov_matrix2)
    cov_matrix = cov_matrix1 @ inverse_term @ cov_matrix2
    mean = cov_matrix2 @ inverse_term @ mean1
    mean += cov_matrix1 @ inverse_term @ mean2
    return mean, cov_matrix


def logpdf(parm, mean1, mean2, cov_matrix1, cov_matrix2):
    """ Computes the log-pdf of the product Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean1 and mean2: means of the two distributions.
            cov_matrix1: and cov_matrix2: covariance matrices for the two distributions.

        Returns:
            A scalar with the value of the log-pdf.

    """
    mean, cov_matrix = compute_statistics(
        mean1, mean2, cov_matrix1, cov_matrix2)
    return mvn.logpdf(parm, mean, cov_matrix)


def rvs(mean1, mean2, cov_matrix1, cov_matrix2):
    """ Generates a random variable from the product Gaussian distribution.

        Args:
            mean1 and mean2: means of the two distributions.
            cov_matrix1: and cov_matrix2: covariance matrices for the two distributions.

        Returns:
            A scalar with the value of the log-pdf.

    """
    mean, cov_matrix = compute_statistics(
        mean1, mean2, cov_matrix1, cov_matrix2)
    return mvn.rvs(mean, cov_matrix)
