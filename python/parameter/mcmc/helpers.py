import numpy as np

def estimate_hessian(mcmc):
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

def compute_acf(data, max_lag=None):
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

def compute_iact(mcmc, state_history, lag=None, max_lag=250):
    params_free = get_history(mcmc, state_history, lag=lag)
    iact = np.zeros(mcmc.no_params_to_estimate)

    try:
        for i in range(mcmc.no_params_to_estimate):
            acf = compute_acf(params_free[:, i], max_lag)
            iact[i] = 1.0 + 2.0 * np.sum(acf)
    except:
        iact *= np.nan

    return iact

def compute_ess(mcmc, state_history, lag=None, max_lag=250):
    return  mcmc.current_iter / compute_iact(mcmc, state_history, lag, max_lag)

def compute_sjd(mcmc, state_history, lag=None):
    params_free = get_history(mcmc, state_history, lag)
    squared_jumps = np.linalg.norm(np.diff(params_free, axis=0), 2, axis=1)**2
    return np.mean(squared_jumps)
