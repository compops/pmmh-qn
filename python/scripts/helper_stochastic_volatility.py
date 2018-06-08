import numpy as np

from models.stochastic_volatility import StochasticVolatilityModel
from state.particle_methods.cython import ParticleMethodsCython

from parameter.mcmc.mh_zero_order import ZeroOrderMetropolisHastings as MetropolisHastings0
from parameter.mcmc.mh_first_order import FirstOrderMetropolisHastings as MetropolisHastings1
from parameter.mcmc.mh_second_order import SecondOrderMetropolisHastings as MetropolisHastings2
from parameter.mcmc.mh_quasi_newton import QuasiNewtonMetropolisHastings

def run(mh_version, mh_settings, seed_offset=0, alg_type=None):

    # Set random seed
    np.random.seed(87655678 + int(seed_offset))

    # System model
    sys_model = StochasticVolatilityModel()
    sys_model.import_data_quandl(handle="BITSTAMP/USD",
                                 start_date="2016-11-07",
                                 end_date="2017-11-07",
                                 variable='VWAP')

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model(params_to_estimate = ('mu', 'phi', 'sigma_v', 'rho'))

    # Setup filtering/smoothing algorithm
    smoother = ParticleMethodsCython(sys_model)

    # Metropolis-Hastings algorithm
    if mh_version is 'mh0':
        sampler = MetropolisHastings0(sys_model, mh_settings)
        file_path = '../results/example3-stochastic-volatility/mh0'
    elif mh_version is 'mh1':
        sampler = MetropolisHastings1(sys_model, mh_settings)
        file_path = '../results/example3-stochastic-volatility/mh1'
    elif mh_version is 'mh2':
        sampler = MetropolisHastings2(sys_model, mh_settings)
        file_path = '../results/example3-stochastic-volatility/mh2'
    elif mh_version is 'qmh':
        sampler = QuasiNewtonMetropolisHastings(sys_model, mh_settings, qn_method=alg_type)
        file_path = '../results/example3-stochastic-volatility/qmh_' + alg_type
    else:
        raise NameError("Unknown MH method...")

    # Run sampler
    sampler.run(smoother)

    # Save results to file
    if alg_type:
        sim_name = 'example3-' + mh_version + '_' + alg_type + '_' + str(seed_offset)
    else:
        sim_name = 'example3-' + mh_version + '_' + str(seed_offset)
    sampler.save_to_file(file_path=file_path, sim_name=sim_name, sim_desc="")