from __future__ import absolute_import
import cython

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, exp, isfinite
from libc.float cimport FLT_MAX
from libc.stdlib cimport malloc, free

DEF NPART = 100
DEF NOBS = 100
DEF NPARAMS = 2
DEF PI = 3.1415

@cython.cdivision(True)
@cython.boundscheck(False)
def get_settings():
    return NOBS, NPART

@cython.cdivision(True)
@cython.boundscheck(True)
def importance_discrete(double [:] obs, double [:] params, double rvr, double [:] rvp):
    # Define counters
    cdef int i
    cdef int j
    cdef int k
    cdef int t
    cdef int idx

    # Initialise variables
    cdef double *particles = <double *>malloc(NPART * NOBS * sizeof(double))
    cdef double[NOBS] filt_state_est, state_trajectory
    cdef double[NPART] weights, unnorm_weights, shifted_weights
    cdef double[NPARAMS] gradient
    cdef double sub_gradient_mu, sub_gradient_sigma

    for i in range(NOBS):
        filt_state_est[i] = 0.0
        state_trajectory[i] = 0.0

    for j in range(NPART):
        weights[j] = 0.0
        unnorm_weights[j] = 0.0
        shifted_weights[j] = 0.0

    for k in range(NPARAMS):
        gradient[k] = 0.0

    # Define parameters
    cdef double mu, sigma
    mu = params[0]
    sigma = params[1]

    # Generate the particles
    for i in range(NOBS):
        for j in range(NPART):
            particles[i + j * NOBS] = mu + sigma * rvp[i + j * NOBS]

    # Define helpers
    cdef double max_weight, norm_factor, log_like, foo_double
    cdef double unnorm_weights_g

    # Compute weights
    for i in range(NOBS):
        for j in range(NPART):
            unnorm_weights_g = norm_logpdf(obs[i], particles[i + j * NOBS], 1.0)
            if isfinite(unnorm_weights_g):
                unnorm_weights[j] += unnorm_weights_g

    max_weight = my_max(unnorm_weights)
    norm_factor = 0.0
    for j in range(NPART):
        shifted_weights[j] = exp(unnorm_weights[j] - max_weight)
        if isfinite(shifted_weights[j]):
            norm_factor += shifted_weights[j]
        else:
            shifted_weights[j] = 0.0

    # Estimate log-likelihood
    log_like = max_weight + log(norm_factor) - NOBS * log(NPART)

    # Normalise weights and compute state filtering estimate
    for j in range(NPART):
        weights[j] = shifted_weights[j] / norm_factor
        for i in range(NOBS):
            filt_state_est[i] += weights[j] * particles[i + j * NOBS]

    # Sample trajectory
    idx = sampleParticle_corr(weights, rvr)
    for i in range(NOBS):
        state_trajectory[i] = particles[i + idx * NOBS]

    # Compute gradients
    for j in range(NPART):
        for i in range(NOBS):
            sub_gradient_mu = sigma**(-2) * (particles[i + j * NOBS] - mu)
            gradient[0] += weights[j] * sub_gradient_mu

            sub_gradient_sigma = sigma**(-2) * (particles[i + j * NOBS] - mu)**2 - 1.0
            gradient[1] += weights[j] * sub_gradient_sigma

    free(particles)

    # Compile the rest of the output
    return filt_state_est, log_like, state_trajectory, gradient

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double norm_logpdf(double x, double m, double s):
    """Helper for computing the log of the Gaussian pdf."""
    cdef double part1 = -0.91893853320467267
    cdef double part2 = -log(s)
    cdef double part3 = -0.5 * (x - m) * (x - m) / (s * s)
    return part1 + part2 + part3

@cython.boundscheck(False)
cdef double my_max(double weights[NPART]):
    cdef int idx = 0
    cdef int i = 0
    cdef double current_largest = weights[0]

    for i in range(1, NPART):
        if weights[i] > current_largest and isfinite(weights[i]):
            idx = i
    return weights[idx]

@cython.cdivision(True)
@cython.boundscheck(False)
cdef int sampleParticle_corr(double weights[NPART], double rnd_number):
    cdef int cur_idx = 0
    cdef int j = 0
    cdef double[NPART] cum_weights
    cdef double sum_weights

    # Compute the empirical CDF of the weights
    cum_weights[0] = weights[0]
    sum_weights = weights[0]
    for j in range(1, NPART):
        cum_weights[j] = cum_weights[j-1] + weights[j]
        sum_weights += weights[j]

    for j in range(1, NPART):
        cum_weights[j] /= sum_weights

    for j in range(NPART):
        if cum_weights[cur_idx] < rnd_number:
            cur_idx += 1
        else:
            break
    return cur_idx