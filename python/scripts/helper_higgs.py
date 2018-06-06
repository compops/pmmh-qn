import numpy as np

from models.logistic_regression import LogisticRegressionModel
from state.direct.standard import DirectComputation

from parameter.mcmc.mh_zero_order import ZeroOrderMetropolisHastings as MetropolisHastings0
from parameter.mcmc.mh_first_order import FirstOrderMetropolisHastings as MetropolisHastings1
from parameter.mcmc.mh_second_order import SecondOrderMetropolisHastings as MetropolisHastings2
from parameter.mcmc.mh_quasi_newton import QuasiNewtonMetropolisHastings

def run(mh_version, mh_settings, data, use_all_data=False, seed_offset=0, alg_type=None):
    # Set random seed
    np.random.seed(87655678 + int(seed_offset))

    # System model
    sys_model = LogisticRegressionModel()
    sys_model.load_data_object(data, add_intercept=True)
    sys_model.params = np.zeros(sys_model.no_params)

    # Inference model
    sys_model.fix_true_params()
    sys_model.create_inference_model()

    # Filter and smoother
    dc_settings = {'no_particles': 1000}
    dc = DirectComputation(sys_model, new_settings=dc_settings, use_all_data=use_all_data)

    # Metropolis-Hastings algorithm
    if mh_version is 'mh0':
        sampler = MetropolisHastings0(sys_model, mh_settings)
        file_path = '../results-draft1/example3-higgs/mh0'
    elif mh_version is 'mh1':
        sampler = MetropolisHastings1(sys_model, mh_settings)
        file_path = '../results-draft1/example3-higgs/mh1'
    elif mh_version is 'qmh':
        sampler = QuasiNewtonMetropolisHastings(sys_model, mh_settings, qn_method=alg_type)
        file_path = '../results-draft1/example3-higgs/qmh-' + alg_type
    else:
        raise NameError("Unknown MH method...")

    # Run sampler
    sampler.run(dc)
    print(np.diag(sampler.estimate_hessian()))

    # Save results to file
    if alg_type:
        sim_name = 'example3-' + mh_version + '-' + alg_type + '-' + str(seed_offset)
    else:
        sim_name = 'example3-' + mh_version + '-' + str(seed_offset)
    sampler.save_to_file(file_path=file_path, sim_name=sim_name, sim_desc="")
