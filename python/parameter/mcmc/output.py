"""Helpers for displaying and storing results from MCMC algorithms."""
import sys
import copy
import time

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from helpers.file_system import write_to_json, write_to_feather
from parameter.mcmc.helpers import compute_acf
from parameter.mcmc.helpers import compute_iact
from parameter.mcmc.helpers import compute_sjd

from palettable.colorbrewer.qualitative import Dark2_8


def print_progress_report(mcmc, proposed_state, state_history, lag=1000):
    i = mcmc.current_iter
    iters_between_reports = mcmc.settings['no_iters_between_progress_reports']

    if (i + 1) % iters_between_reports is not 0:
        return None

    mean_time_per_iter = (time.time() - mcmc.start_time) / i
    est_time_remaining = mean_time_per_iter * (mcmc.settings['no_iters'] - i)

    avg_params = np.zeros(mcmc.no_params_to_estimate)
    for j in range(i):
        avg_params += state_history[j]['params'] / (i + 1)

    avg_accept_rate = 0.0
    if i > (lag + 1):
        for j in range(i - lag + 1, i):
            avg_accept_rate += state_history[j]['accepted'] / float(lag)
    else:
        for j in range(i):
            avg_accept_rate += state_history[j]['accepted'] / (i + 1)

    print("###################################################################")
    print(" Iteration: " + str(i + 1) + " of : "
        + str(mcmc.settings['no_iters']) + " completed.")
    print(" Time per iteration: {:.3f}s and estimated time remaining: {:.3f}s.".format(mean_time_per_iter, est_time_remaining))
    print("")
    print(" Current state of the Markov chain:")
    print( ["%.4f" % v for v in state_history[i-1]['params']])
    print("")
    print(" Proposed next state of the Markov chain:")
    print( ["%.4f" % v for v in proposed_state['params']])
    print("")
    print(" Current posterior mean estimate: ")
    print( ["%.4f" % v for v in avg_params])
    print("")
    print(" Current acceptance rate (over last 1,000 iterations):")
    print(" %.4f" % avg_accept_rate)
    print("")
    print(" Current IACT values (over last 1,000 iterations):")
    print( ["%.2f" % v for v in compute_iact(mcmc, state_history, lag=lag)])
    if mcmc.alg_type is 'qmh':
        if (i > mcmc.settings['memory_length']):
            avg_samples_hessian = 0.0
            avg_hessian_estimated = 0.0
            avg_hessians_corrected = 0.0
            for j in range(i):
                if state_history[j]['hessian_samples'] > 0:
                    avg_samples_hessian += state_history[j]['hessian_samples'] / i
                    avg_hessian_estimated += state_history[j]['hessian_estimated'] * 1.0 / i
                    avg_hessians_corrected += state_history[j]['hessian_corrected'] * 1.0 / i
            print("")
            print(" Mean number of samples for Hessian estimate:")
            print(" {:.4f}".format(avg_samples_hessian))
            print("")
            print(" Rate of successful QN proposal computations:")
            print(" {:.4f}".format(avg_hessian_estimated))
            print("")
            print(" Rate of Hessian correction:")
            print(" {:.4f}".format(avg_hessians_corrected))
    if 'adapted_step_size' in proposed_state:
        print("")
        print(" Current adapted step size (with target accept prob: {}):".format(mcmc.settings['adapt_step_size_target']))
        print(" {:.4f}".format(proposed_state['adapted_step_size']))
    print("")
    print("###################################################################")
    print("")



def print_greeting(mcmc, estimator):
    print("")
    print("###################################################################")
    print("Starting MH algorithm...")
    print("")
    print("Sampling from the parameter posterior using: " + mcmc.name)
    print("in the model: " + mcmc.model.name)
    print("")
    if mcmc.using_gradients:
        print("Likelihood and gradients estimated using:")
        print(estimator.name)
        if mcmc.hessians_computed_by:
            print("Hessian is estimated using: " + mcmc.hessians_computed_by)
    else:
        print("Likelihood estimated using:")
        print(estimator.name)
    print("")
    if 'hessian' in mcmc.settings:
        step_size = mcmc.settings['step_size_hessian']
        eff_step_size = np.sqrt(np.diag(mcmc.settings['hessian'])) * step_size
        print("The proposal has effective random step sizes: ")
        i = 0
        for param in mcmc.model.params_to_estimate:
            print("{}: {:.3f}".format(param, eff_step_size[i]))
            i += 1
        print("")
    print("Running MH for {} iterations.".format(mcmc.settings['no_iters']))
    print("")
    print("The rest of the settings are as follows: ")
    for key in mcmc.settings:
        print("{}: {}".format(key, mcmc.settings[key]))
    print("")
    print("###################################################################")



def plot(mcmc, max_acf_lag = 500):
    """ Plots results to the screen after a run of an MCMC algorithm. """
    no_iters = mcmc.settings['no_iters']
    no_burnin_iters = mcmc.settings['no_burnin_iters']
    effective_iters = no_iters - no_burnin_iters
    no_params = mcmc.model.no_params_to_estimate
    param_names = mcmc.model.params_to_estimate
    no_bins = int(np.sqrt(effective_iters))

    accept_rate = params = np.zeros(effective_iters)
    params = np.zeros((effective_iters, no_params))
    prop_params = np.zeros((effective_iters, no_params))
    nat_gradient = np.zeros((effective_iters, no_params))
    no_nat_gradients = True

    for i in range(effective_iters):
        j = i + no_burnin_iters
        accept_rate[i] = mcmc.state_history[j]['accepted']
        params[i, :] = mcmc.state_history[j]['params']
        prop_params[i, :] = mcmc.state_history[j]['params_prop']
        if 'nat_gradient' in mcmc.state_history[j]:
            nat_gradient[i, :] = mcmc.state_history[j]['nat_gradient']
            no_nat_gradients = False
        else:
            no_nat_gradients = True

    accept_rate = np.cumsum(accept_rate) / np.arange(1, effective_iters+1)

    plt.figure(1)
    plt.plot(accept_rate)
    plt.ylabel("Running mean acceptance rate")
    plt.xlabel("iter")

    plt.figure(2)
    for i in range(no_params):
        col = Dark2_8.mpl_colors[i % 8]

        plt.subplot(no_params, 5, 5 * i + 1)
        plt.hist(params[:, i], bins=no_bins, color = col)
        plt.ylabel("Marginal posterior probability of " + str(param_names[i]))
        plt.xlabel("iter")

        plt.subplot(no_params, 5, 5 * i + 2)
        plt.plot(params[:, i], color = col)
        plt.ylabel("Parameter trace of " + str(param_names[i]))
        plt.xlabel("iter")

        plt.subplot(no_params, 5, 5 * i + 3)
        acf = compute_acf(params[:, i], max_lag=max_acf_lag)
        plt.plot(np.arange(max_acf_lag), acf, color = col)
        plt.ylabel("ACF of " + str(param_names[i]))
        plt.xlabel("iter")

        plt.subplot(no_params, 5, 5 * i + 4)
        plt.plot(prop_params[:, i], color = col)
        plt.ylabel("Proposed trace of " + str(param_names[i]))
        plt.xlabel("iter")

        plt.subplot(no_params, 5, 5 * i + 5)
        if not no_nat_gradients:
            plt.plot(nat_gradient[:, i], color = col)
            plt.ylabel("natural gradient of " + str(param_names[i]))
            plt.xlabel("iter")
    plt.show()



def compile_results(mcmc, sim_name=None):
    no_iters = mcmc.settings['no_iters']
    no_burnin_iters = mcmc.settings['no_burnin_iters']
    no_effective_iters = no_iters - no_burnin_iters
    idx = range(no_burnin_iters, no_iters)
    current_time = time.strftime("%c")

    mcmcout = {}
    mcmcout.update({'simulation_name': sim_name})
    mcmcout.update({'simulation_time': current_time})
    mcmcout.update({'time_per_iteration': mcmc.time_per_iter})

    df = pd.DataFrame(mcmc.state_history)
    field_names = ['params', 'params_prop', 'nat_gradient', 'accepted', 'state_trajectory']

    for field in field_names:
        if not field in mcmc.state_history[0]:
            print("Skipping saving " + field + " to file.")
            continue

        if type(df[0][field]) is float or type(df[0][field]) is np.float64:
            tmp = np.zeros((no_effective_iters, 1))
        else:
            tmp = np.zeros((no_effective_iters, len(df[0][field])))

        for i in range(no_effective_iters):
            foo = df[i + no_burnin_iters][field]

            if type(foo) is not np.ndarray:
                if np.isnan(foo): foo = 0.0
                if np.isposinf(foo): foo = 10000000.0
                if np.isneginf(foo): foo = -10000000.0

            if type(foo) is np.ndarray:
                idx = np.where(np.isnan(foo))
                foo[idx] = 0.0
                idx = np.where(np.isposinf(foo))
                foo[idx] = 10000000.0
                idx = np.where(np.isneginf(foo))
                foo[idx] = -10000000.0

            tmp[i, :] = foo

        mcmcout.update({field: tmp})

    if hasattr(mcmc, 'adapted_step_sizes'):
        mcmcout.update({'adapted_step_sizes': mcmc.adapted_step_sizes})

    if hasattr(mcmc, 'no_hessians_corrected'):
        mcmcout.update({'no_hessians_corrected': mcmc.no_hessians_corrected})

    data = {}
    data.update({'observations': mcmc.model.obs})
    if mcmc.model.states is None:
        data.update({'states': mcmc.model.states})
    data.update({'simulation_name': sim_name})
    data.update({'simulation_time': current_time})

    settings = copy.deepcopy(mcmc.settings)
    recasted_settings = {}
    for key in settings:
        recasted_settings.update({key: str(settings[key])})
    recasted_settings.update({'sampler_name': mcmc.name})
    recasted_settings.update({'simulation_name': sim_name})
    recasted_settings.update({'simulation_time': current_time})

    return mcmcout, data, recasted_settings, df



def save_to_file(mcmc, file_path, sim_name=None, sim_desc=None):

    mcout, data, settings, mcoutdf = compile_results(mcmc, sim_name=sim_name)

    if sim_desc:
        desc = {'description': sim_desc,
                'time': settings['simulation_time']
            }
        write_to_json(desc, file_path, sim_name, 'description.txt')

    try:
        write_to_json(mcout, file_path, sim_name, 'mcmc_output.json')
    except:
        print("Error exporting JSON. Probably due to NaN or Inf. Use feather instead.")
    #write_to_feather(mcoutdf, file_path, sim_name, 'mcmc_output.feather')
    write_to_json(data, file_path, sim_name, 'data.json')
    write_to_json(settings, file_path, sim_name, 'settings.json')