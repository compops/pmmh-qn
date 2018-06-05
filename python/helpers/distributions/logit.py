import numpy as np

def logit(x):
    return (1 + np.exp(x))**(-1)

def invlogit(x):
    assert x < 1.0
    assert x > 0.0
    return np.log(x / (1.0 - x))

def gradient_log_logit(f, fprime):
    return fprime * (1.0 + np.exp(-f))**(-2.0) * np.exp(-f)
