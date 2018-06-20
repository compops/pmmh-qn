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

"""Helpers for the Gamma distribtion."""
import numpy as np
import scipy as sp


def pdf(param, shape, rate):
    """ Computes the pdf of the Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the pdf.

    """
    coef = rate**shape / sp.special.gamma(shape)
    return coef * param**(shape - 1.0) * np.exp(-rate * param)


def logpdf(param, shape, rate):
    """ Computes the log-pdf of the Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the log-pdf.

    """
    coef = shape * np.log(rate) - sp.special.gammaln(shape)
    return coef + (shape - 1.0) * np.log(param) - rate * param


def logpdf_gradient(param, shape, rate):
    """ Computes the gradient of the log-pdf of the Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    return (shape - 1.0) / param - rate


def logpdf_hessian(param, shape, rate):
    """ Computes the Hessian of the log-pdf of the Gamma distribution.

        Args:
            param: value to evaluate in
            shape: shape parameter
            rate: rate parameter

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    return - (shape - 1.0) / (param**2)
