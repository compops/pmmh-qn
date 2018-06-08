"""Script for reproducing example 3 in paper."""
import copy
import numpy as np
import scripts.helper_earthquake as mh

def main(seed_offset=0):
    """Runs the experiment."""
    initial_params = (3.0, 0.9, 0.2)
    hessian_estimate = np.diag((0.02, 0.05, 0.03))
    hessian_guess = np.diag((0.005, 0.005, 0.05))

    mh_settings = {'no_iters': 30000,
                   'no_burnin_iters': 10000,
                   'adapt_step_size': True,
                   'adapt_step_size_initial': 0.25,
                   'adapt_step_size_rate': 0.5,
                   'adapt_step_size_target': 0.6,
                   'initial_params': initial_params,
                   'no_iters_between_progress_reports': 1000,
                   'correlated_rvs': True,
                   'correlated_rvs_sigma': 0.5,
                   'memory_length': 20,
                   'accept_first_iterations': 20,
                   'hessian': hessian_guess,
                   'hess_corr_fallback': hessian_guess,
                   'hess_corr_method': 'flip'
    }

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'hessian': hessian_estimate})
    new_mh_settings.update({'adapt_step_size': False})
    new_mh_settings.update({'step_size_gradient': 0.0})
    new_mh_settings.update({'step_size_hessian': 0.5 * 2.562 / np.sqrt(3)})
    mh.run('mh0',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset)

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'hessian': hessian_estimate})
    new_mh_settings.update({'adapt_step_size': False})
    new_mh_settings.update({'step_size_gradient': 0.01 *  1.125 / np.sqrt(3**(1.0 / 3.0))})
    new_mh_settings.update({'step_size_hessian': 0.01 *  1.125 / np.sqrt(3**(1.0 / 3.0))})
    mh.run('mh1',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset)

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.2})
    new_mh_settings.update({'adapt_step_size_initial': 0.1})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset,
           alg_type='bfgs')

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.3})
    new_mh_settings.update({'sr1_trust_region': True})
    new_mh_settings.update({'sr1_trust_region_scale': 1.0})
    new_mh_settings.update({'sr1_trust_region_cov': hessian_guess})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset,
           alg_type='sr1')

    new_mh_settings = copy.deepcopy(mh_settings)
    new_mh_settings.update({'adapt_step_size_target': 0.2})
    new_mh_settings.update({'ls_regularisation_parameter': 0.1})
    mh.run('qmh',
           mh_settings=new_mh_settings,
           seed_offset=seed_offset,
           alg_type='ls')

    return None


if __name__ == '__main__':
    main()
