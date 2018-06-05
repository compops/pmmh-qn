"""Helpers for the Poisson distribtion."""
import numpy as np
from scipy.misc import factorial


def pdf(param, lam):
    """ Computes the pdf of the Poisson distribution.

        Args:
            param: value to evaluate in
            lam: rate parameter

        Returns:
            A scalar with the value of the pdf.

    """
    return lam**param * exp(-lam) / factorial(param)


def logpdf(param, lam):
    """ Computes the log-pdf of the Poisson distribution.

        Args:
            param: value to evaluate in
            lam: rate parameter

        Returns:
            A scalar with the value of the log-pdf.

    """
    return param * np.log(lam) - lam - np.log(factorial(param))


def logpdf_gradient(param, lam):
    """ Computes the gradient of the log-pdf of the Poisson distribution.

        Args:
            param: value to evaluate in
            lam: rate parameter

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    return param / lam - 1.0


def logpdf_hessian(param, lam):
    """ Computes the Hessian of the log-pdf of the Poisson distribution.

        Args:
            param: value to evaluate in
            lam: rate parameter

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    return -param * lam**(-2)