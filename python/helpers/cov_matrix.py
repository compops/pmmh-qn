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
"""Helpers for checking covariance matrices."""

import numpy as np
from scipy.linalg import eigh
from scipy.stats._multivariate import _eigvalsh_to_eps


def is_psd(hessian):
    """ Checks if positive semi-definite matrix.

        Computes the eigenvalues and checks for negative ones.

        Args:
            hessian: a matrix to be checked.

        Returns:
           True if the array is positive semi-definite and False otherwise.

    """
    # Check for NaNs or Infs
    if isinstance(hessian, np.ndarray):
        if np.any(np.isinf(hessian)) or np.any(np.isnan(hessian)):
            return False

    return np.all(np.linalg.eigvals(hessian) > 0)


def problematic_gradient(gradient):
    """ Checks if the gradient is problematic

        Args:
            gradient: a vector to be checked.

        Returns:
           True if the array is valid gradient and False otherwise.

    """
    flag = False

    # Check for nan and inf
    if not np.isfinite(gradient).all():
        flag = True

    # Check for complex elements
    if np.iscomplex(gradient).any():
        flag = True

    return flag


def problematic_hessian(hessian, check_pd=True, check_sing=True, verbose=False):
    """ Checks if the Hessian (covariance matrix) is problematic.

        Computes the eigenvalues and checks for negative ones. Also checks
        if the matrix is singular.

        Args:
            hessian: a matrix to be checked.

        Returns:
           True if the array is valid covariance matrix and False otherwise.

    """
    if hessian is None:
        return True

    # Check for nan and inf
    if np.isnan(hessian).any():
        if verbose:
            print("Warning: some Hessian elements are not finite...")
        return True

    # Check for nan and inf
    if not np.all(np.isfinite(hessian)):
        if verbose:
            print("Warning: some Hessian elements are not finite...")
        return True

    # Check for nan and inf
    if np.any(np.isnan(hessian)):
        if verbose:
            print("Warning: some Hessian elements are NaN...")
        return True

    # Check for complex elements
    if np.any(np.iscomplex(hessian)):
        if verbose:
            print("Warning: some Hessian elements are complex...")
        return True

    # Check for large elements
    if np.any(np.diag(hessian) > 1e20):
        if verbose:
            print("Warning: some Hessian elements are too large...")
        return True

    # Singular matrix (too small eigenvalues)
    # Negative eigenvalues
    flag = False
    try:
        eig_values = eigh(hessian, lower=True, check_finite=True)[0]
        eps = _eigvalsh_to_eps(eig_values, None, None)

        if check_sing and (np.abs(eig_values) < eps).any():
            flag = True
            if verbose:
                print("Warning: Hessian has very small eigenvalues: {}.".format(
                    np.min(eig_values)))

        large_eig_values = eig_values[np.abs(eig_values) > eps]
        if check_sing and len(large_eig_values) < len(eig_values):
            if verbose:
                print("Warning: Hessian is illconditioned with eigenvalues:")
                print(eig_values)
            flag = True

        if check_pd and np.min(eig_values) < 0.0:
            if verbose:
                neg_eig_values = eig_values[eig_values < 0.0]
                print("Warning: Hessian has negative eigenvalues: {}".format(
                    neg_eig_values))
            flag = True

    except Exception as e:
        print(e)
        raise Warning(
            "Numerical issues in eigenvalue computations, rejecting.")
        flag = True

    return flag


def problematic_hessian_notfinite(hessian, verbose=False):
    """Wrapper for just checking for complex, inf, and nans in Hessian."""
    return problematic_hessian(hessian, check_pd=False, check_sing=False, verbose=verbose)


def correct_hessian(estimate, fallback_hessian, strategy='flip', verbose=False):
    """ Corrects non positive-definite Hessians using different strategies.

        Args:
            estimate: Hessian estimate to be checked and possibly corrected.
            fallback_hessian: a fallback if the Hessian cannot be corrected.
            strategy: the correction method used (see below).

        Returns:
           A valid covariance matrix and the flag True if the Hessian has been
           corrected and False otherwise.

        The Hessian be be corrected using replace, regularise and flip.

            replace: replaces the Hessian with fallback_hessian.
            regularise: adds a diagonal matrix that flips the smallest
                        eigenvalue to be positive.
            flip: makes use of an eigenvalue decomposition to make sure that
                  all the eigenvalues are positive and larger than flip_limit.

    """
    flip_limit = 1e-4

    if problematic_hessian_notfinite(estimate, verbose=verbose):
        return fallback_hessian, True

    if not strategy or not problematic_hessian(estimate, verbose=verbose):
        # No correction required
        return estimate, False

    if strategy is 'replace' or estimate is None:
        # Replace the Hessian estimate with another estimate
        corr_estimate = fallback_hessian
        # if verbose:
        print("Corrected Hessian: replaced with fallback Hessian.")

    elif strategy is 'regularise':
        # Shift the eigenvalues by adding a diagonal matrix
        corr_estimate = estimate - 2.0 * \
            np.min(np.linalg.eig(estimate)[0]) * np.eye(estimate.shape[0])
        if verbose:
            print(
                "Corrected Hessian: added diagonal matrix to shift negative eigenvalues.")

    elif strategy is 'flip':
        # Flip negative eigenvalues by an eigenvalue/eigenvector decomposition
        evd = np.linalg.eig(estimate)
        ev_matrix = np.abs(evd[0])
        for i, value in enumerate(ev_matrix):
            ev_matrix[i] = np.max((value, flip_limit))
        corr_estimate = evd[1] @ np.diag(ev_matrix) @ np.linalg.inv(evd[1])
        if verbose:
            print("Corrected Hessian: by using spectral decomposition.")
    else:
        raise ValueError("Unknown Hessian correction strategy...")

    # Check if the Hessian is correct now.
    if problematic_hessian(corr_estimate, verbose=verbose):
        return fallback_hessian, True
    else:
        return corr_estimate, True
