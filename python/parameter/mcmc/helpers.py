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
"""Helpers for MCMC samplers."""

import numpy as np


def estimate_hessian(mcmc):
    """Estimates the Hessian using the trace from the burn-in."""
    if not hasattr(mcmc, 'state_history'):
        raise NameError("No state history found in MCMC object.")

    no_iters = mcmc.current_iter
    no_burn_in = mcmc.settings['no_burnin_iters']
    no_effective_iters = no_iters - no_burn_in

    free_params = np.zeros((no_effective_iters, mcmc.no_params_to_estimate))

    for i in range(no_effective_iters):
        j = i + no_burn_in
        free_params[i, :] = mcmc.state_history[j]['params_free']

    return np.cov(free_params, rowvar=False)


def get_history(mcmc, state_history, lag=None):
    """Helper to extract the history of the Markov chain."""
    no_iters = mcmc.current_iter
    no_burnin_iters = mcmc.settings['no_burnin_iters']
    if lag:
        no_effective_iters = lag
    else:
        no_effective_iters = no_iters - no_burnin_iters
    no_params = mcmc.no_params_to_estimate

    params_free = np.zeros((no_effective_iters, no_params))
    for i in range(no_effective_iters):
        if lag:
            j = int(no_iters - i)
        else:
            j = int(i + no_burnin_iters)
        if j < 0:
            continue
        params_free[i, :] = state_history[j]['params_free']
    return params_free


def compute_acf(data, max_lag=100):
    """Helper for computing the empirical ACF."""
    no_data = len(data)
    variance = np.var(data)
    data = data - np.mean(data)
    correlations = np.correlate(data, data, mode='full')[-no_data:]
    result = correlations / (variance * (np.arange(no_data, 0, -1)))
    if not max_lag:
        max_lag = np.where(np.abs(result) < 1.96 / np.sqrt(no_data))
        if len(max_lag[0] > 0):
            max_lag = max_lag[0][0]
        else:
            max_lag = len(result)
    return result[0:max_lag]


def compute_iact(mcmc, state_history, lag=None, max_lag=100):
    """Helper for computing the IACT."""
    params_free = get_history(mcmc, state_history, lag=lag)
    iact = np.zeros(mcmc.no_params_to_estimate)

    try:
        for i in range(mcmc.no_params_to_estimate):
            acf = compute_acf(params_free[:, i], max_lag)
            iact[i] = 1.0 + 2.0 * np.sum(acf)
    except:
        iact *= np.nan

    return iact


def compute_ess(mcmc, state_history, lag=None, max_lag=100):
    """Helper for computing the effective sample size."""
    return mcmc.current_iter / compute_iact(mcmc, state_history, lag, max_lag)


def compute_sjd(mcmc, state_history, lag=None):
    """Helper for computing the squared jump distance."""
    params_free = get_history(mcmc, state_history, lag)
    squared_jumps = np.linalg.norm(np.diff(params_free, axis=0), 2, axis=1)**2
    return np.mean(squared_jumps)
