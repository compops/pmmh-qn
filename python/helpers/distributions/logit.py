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
"""Helper for the logit function."""

import numpy as np


def logit(x):
    """Computes the logit of x."""
    return (1 + np.exp(x))**(-1)


def invlogit(x):
    """Computes the inverse logit of x."""
    assert x < 1.0
    assert x > 0.0
    return np.log(x / (1.0 - x))


def gradient_log_logit(f, fprime):
    """Computes the gradient of the logit of x."""
    return fprime * (1.0 + np.exp(-f))**(-2.0) * np.exp(-f)
