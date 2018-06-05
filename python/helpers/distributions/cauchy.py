"""Helpers for the Cauchy distribtion."""

import numpy as np

def pdf(param, loc, scale):
    """ Computes the pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the pdf.

    """
    term1 = np.pi * scale
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return term1**(-1) * term2**(-1)

def logpdf(param, loc, scale):
    """ Computes the log-pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the log-pdf.

    """
    term1 = np.pi * scale
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return -np.log(term1) - np.log(term2)

def logpdf_gradient(param, loc, scale):
    """ Computes the gradient of the log-pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return -scale**(-2) * (param - loc) / term2

def logpdf_hessian(param, loc, scale):
    """ Computes the Hessian of the log-pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return -scale**(-2) / term2 + scale**(-4) * (param - loc)**2 / term2**2
