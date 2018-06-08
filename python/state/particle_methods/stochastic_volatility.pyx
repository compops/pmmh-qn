from __future__ import absolute_import

import cython
cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free
from cython cimport view
cnp.import_array()

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, exp, isfinite
from libc.float cimport FLT_MAX
from libc.stdlib cimport malloc, free

DEF LAG = 10
DEF NPART = 100
DEF NOBS = 361
DEF NPARAMS = 4
DEF PI = 3.1415

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct Sorter:
    int index
    double value

cdef int _compare(const_void *a, const_void *b):
    cdef double v = ((<Sorter*>a)).value-((<Sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1

cdef void cyargsort(double[:] data, Sorter * order):
    cdef int i
    cdef int n = data.shape[0]
    for i in range(n):
        order[i].index = i
        order[i].value = data[i]
    qsort(<void *> order, n, sizeof(Sorter), _compare)

cpdef argsort(double[:] data, int[:] order):
    cdef int i
    cdef int n = data.shape[0]
    cdef Sorter *order_struct = <Sorter *> malloc(n * sizeof(Sorter))
    cyargsort(data, order_struct)
    for i in range(n):
        order[i] = order_struct[i].index
    free(order_struct)

@cython.cdivision(True)
@cython.boundscheck(False)
def get_settings():
    return NOBS, NPART, LAG

@cython.cdivision(True)
@cython.boundscheck(False)
def bpf_sv_corr(double [:] obs, double [:] params, double [:] rvr, double [:] rvp):

    # Initialise variables
    cdef int *ancestry = <int *>malloc(NPART * NOBS * sizeof(int))
    cdef int *old_ancestry = <int *>malloc(NPART * NOBS * sizeof(int))
    cdef double *weights = <double *>malloc(NPART * NOBS * sizeof(double))
    cdef double *particles = <double *>malloc(NPART * NOBS * sizeof(double))
    cdef double *weights_at_t = <double *>malloc(NPART * sizeof(double))

    cdef int[NPART] ancestors
    cdef int[NPART] new_idx
    cdef double[NPART] old_particles
    cdef double[NOBS] filt_state_est
    cdef double[NOBS] state_trajectory
    cdef double[NPART] unnorm_weights
    cdef double[NPART] shifted_weights
    cdef double log_like = 0.0

    # Define parameters
    cdef double mu, phi, sigmav, rho
    mu = params[0]
    phi = params[1]
    sigmav = params[2]
    rho = params[3]

    # Define helpers
    cdef double mean = 0.0
    cdef double stDev = 0.0
    cdef double max_weight = 0.0
    cdef double norm_factor
    cdef double foo_double
    cdef int idx

    # Define counters
    cdef int i
    cdef int j
    cdef int k

    # Pre-allocate variables
    for i in range(NOBS):
        filt_state_est[i] = 0.0
        state_trajectory[i] = 0.0
        for j in range(NPART):
            ancestry[i + j * NOBS] = 0
            old_ancestry[i + j * NOBS] = 0
            particles[i + j * NOBS] = 0.0
            weights[i + j * NOBS] = 0.0

    # Generate or set initial state
    stDev = sigmav / sqrt(1.0 - (phi * phi))
    for j in range(NPART):
        particles[0 + j * NOBS] = mu + stDev * rvp[0 + j * NOBS]
        weights[0 + j * NOBS] = 1.0 / NPART
        ancestry[0 + j * NOBS] = j
        weights_at_t[j] = 0.0
        filt_state_est[0] += weights[0 + j * NOBS] * particles[0 + j * NOBS]
        old_particles[j] = particles[0 + j * NOBS]

    # Sort particles
    argsort(old_particles, new_idx)
    for j in range(NPART):
        particles[0 + j * NOBS] = old_particles[new_idx[j]]

    for i in range(1, NOBS):

        # Resample particles
        for j in range(NPART):
            weights_at_t[j] = weights[i - 1 + j * NOBS]
        systematic_corr(ancestors, weights_at_t, rvr[i])

        # Update ancestry
        for k in range(i):
            for j in range(NPART):
                old_ancestry[k + j * NOBS] = ancestry[k + j * NOBS]

        for j in range(NPART):
            ancestry[i + j * NOBS] = ancestors[j]
            for k in range(i):
                ancestry[k + j * NOBS] = old_ancestry[k + ancestors[j] * NOBS]

        # Propagate particles
        for j in range(NPART):
            mean = mu + phi * (particles[i - 1 + ancestors[j] * NOBS] - mu)
            mean += sigmav * rho * exp(-0.5 * particles[i + ancestors[j] * NOBS]) * obs[i - 1]
            stDev = sqrt(1.0 - rho * rho) * sigmav
            particles[i + j * NOBS] = mean + stDev * rvp[i + j * NOBS]

        # Sort particles
        for j in range(NPART):
            old_particles[j] = particles[i + j * NOBS]

        for k in range(i):
            for j in range(NPART):
                old_ancestry[k + j * NOBS] = ancestry[k + j * NOBS]

        argsort(old_particles, new_idx)

        for j in range(NPART):
            particles[i + j * NOBS] = old_particles[new_idx[j]]

        for j in range(NPART):
            for k in range(i+1):
                ancestry[k + j * NOBS] = old_ancestry[k + new_idx[j] * NOBS]

        # Weight particles
        for j in range(NPART):
            unnorm_weights[j] = norm_logpdf(obs[i], 0.0, exp(0.5 *particles[i + j * NOBS]))

        max_weight = my_max(unnorm_weights)
        norm_factor = 0.0
        for j in range(NPART):
            shifted_weights[j] = exp(unnorm_weights[j] - max_weight)
            if isfinite(shifted_weights[j]):
                norm_factor += shifted_weights[j]
            else:
                shifted_weights[j] = 0.0

        # Normalise weights and compute state filtering estimate
        filt_state_est[i] = 0.0
        for j in range(NPART):
            weights[i + j * NOBS] = shifted_weights[j] / norm_factor
            if isfinite(weights[i + j * NOBS] * particles[i + j * NOBS]) != 0:
                filt_state_est[i] += weights[i + j * NOBS] * particles[i + j * NOBS]

        # Estimate log-likelihood
        log_like += max_weight + log(norm_factor) - log(NPART)

    # Sample trajectory
    idx = sampleParticle_corr(weights, rvr[0])
    for i in range(NOBS):
        j = ancestry[i + idx * NOBS]
        state_trajectory[i] = particles[i + j * NOBS]

    free(particles)
    free(weights)
    free(weights_at_t)
    free(ancestry)
    free(old_ancestry)

    # Compile the rest of the output
    return filt_state_est, log_like, state_trajectory

@cython.cdivision(True)
@cython.boundscheck(False)
def flps_sv_corr(double [:] obs, double [:] params, double [:] rvr, double [:] rvp):

    # Initialise variables
    cdef int *ancestors = <int *>malloc(NPART * sizeof(int))
    cdef double *particle_history = <double *>malloc(LAG * NPART * sizeof(double))
    cdef double *old_particle_history = <double *>malloc(LAG * NPART * sizeof(double))
    cdef int *ancestry = <int *>malloc(NPART * NOBS * sizeof(int))
    cdef int *old_ancestry = <int *>malloc(NPART * NOBS * sizeof(int))

    cdef double *particles = <double *>malloc(NOBS * NPART * sizeof(double))
    cdef double *weights = <double *>malloc(NOBS * NPART * sizeof(double))
    cdef double *weights_at_t = <double *>malloc(NPART * sizeof(double))

    cdef double[NPART] old_particles
    cdef int[NPART] new_idx

    cdef double[NOBS] filt_state_est
    cdef double[NOBS] smo_state_est
    cdef double[NOBS] state_trajectory

    cdef double *unnorm_weights = <double *>malloc(NPART * sizeof(double))
    cdef double *shifted_weights = <double *>malloc(NPART * sizeof(double))

    cdef double sub_gradient[NPARAMS]
    cdef double gradient[NPARAMS][NOBS]

    cdef double sub_hessian1[NPARAMS]
    cdef double hessian1[NPARAMS][NPARAMS][NOBS]

    cdef double sub_hessian2[NPARAMS][NPARAMS]
    cdef double hessian2[NPARAMS][NPARAMS][NOBS]

    cdef double log_like = 0.0

    # Define parameters
    cdef double mu, phi, sigmav, rho
    mu = params[0]
    phi = params[1]
    sigmav = params[2]
    rho = params[3]

    # Define helpers
    cdef double mean = 0.0
    cdef double stDev = 0.0
    cdef double max_weight = 0.0
    cdef double norm_factor
    cdef double foo_double

    cdef double q_matrix = 1.0 / (sigmav * sigmav * (1.0 - rho * rho))
    cdef double rho_term = 1.0 - rho * rho
    cdef double state_quad_term = 0.0
    cdef double curr_particle = 0.0
    cdef double next_particle = 0.0

    # Define counters
    cdef int i
    cdef int j
    cdef int current_lag
    cdef int k
    cdef int idx
    cdef int idx_curr
    cdef int idx_next
    cdef int idx_t

    # Initialize ancestry
    for k in range(LAG):
        for j in range(NPART):
            particle_history[k + j * LAG] = 0.0
            old_particle_history[k + j * LAG] = 0.0

    for i in range(NOBS):
        filt_state_est[i] = 0.0
        smo_state_est[i] = 0.0
        state_trajectory[i] = 0.0
        for j in range(NPART):
            weights_at_t[j] = 0.0
            particles[i + j * NOBS] = 0.0
            weights[i + j * NOBS] = 0.0
            ancestry[i + j * NOBS] = 0
            old_ancestry[i + j * NOBS] = 0

    for i in range(NPARAMS):
        sub_gradient[i] = 0.0
        for j in range(NOBS):
            gradient[i][j] = 0.0

    # Generate initial state
    stDev = sigmav / sqrt(1.0 - (phi * phi))
    for j in range(NPART):
        particles[0 + j * NOBS] = mu + stDev * particles[0 + j * NOBS]
        weights[0 + j * NOBS] = 1.0 / NPART
        particle_history[0 + j * LAG] = particles[0 + j * NOBS]
        ancestry[0 + j * NOBS] = j
        old_particles[j] = particles[0 + j * NOBS]

    filt_state_est[0] = 0.0
    for j in range(NPART):
        filt_state_est[0] += weights[0 + j * NOBS] * particles[0 + j * NOBS]

    # Sort particles
    argsort(old_particles, new_idx)

    for j in range(NPART):
        particles[0 + j * NOBS] = old_particles[new_idx[j]]

    for i in range(1, NOBS):
        current_lag = my_min(i, LAG)

        # Resample particles
        for j in range(NPART):
            weights_at_t[j] = weights[i - 1 + j * NOBS]
        systematic_corr(ancestors, weights_at_t, rvr[i])

        # Update buffer for smoother
        for j in range(NPART):
            for k in range(LAG):
                old_particle_history[k + j * LAG] = particle_history[k + j * LAG]

        # Update ancestry
        for k in range(i):
            for j in range(NPART):
                old_ancestry[k + j * NOBS] = ancestry[k + j * NOBS]

        for j in range(NPART):
            ancestry[i + j * NOBS] = ancestors[j]
            for k in range(i):
                ancestry[k + j * NOBS] = old_ancestry[k + ancestors[j] * NOBS]

        # Propagate particles
        for j in range(NPART):
            mean = mu + phi * (particles[i - 1 + ancestors[j] * NOBS] - mu)
            mean += sigmav * rho * exp(-0.5 * particles[i - 1 + ancestors[j] * NOBS]) * obs[i - 1]
            stDev = sqrt(rho_term) * sigmav
            particles[i + j * NOBS] = mean + stDev * rvp[i + j * NOBS]
            particle_history[0 + j * LAG] = particles[i + j * NOBS]
            for k in range(1, LAG):
                particle_history[k + j * LAG] = old_particle_history[k - 1 + ancestors[j] * LAG]

        # Sort particles
        for j in range(NPART):
            old_particles[j] = particles[i + j * NOBS]

        for k in range(i):
            for j in range(NPART):
                old_ancestry[k + j * NOBS] = ancestry[k + j * NOBS]

        argsort(old_particles, new_idx)

        for j in range(NPART):
            particles[i + j * NOBS] = old_particles[new_idx[j]]
            particle_history[0 + j * LAG] = old_particles[new_idx[j]]
            for k in range(1, LAG):
                particle_history[k + j * LAG] = old_particle_history[k - 1 + ancestors[new_idx[j]] * LAG]

            for k in range(i+1):
                ancestry[k + j * NOBS] = old_ancestry[k + new_idx[j] * NOBS]

        # Weight particles
        for j in range(NPART):
            unnorm_weights[j] = norm_logpdf(obs[i], 0.0, exp(0.5 * particles[i + j * NOBS]))

        max_weight = my_max(unnorm_weights)
        norm_factor = 0.0
        for j in range(NPART):
            shifted_weights[j] = exp(unnorm_weights[j] - max_weight)
            foo_double = norm_factor + shifted_weights[j]
            if isfinite(foo_double) != 0:
                norm_factor = foo_double

        # Normalise weights and compute state filtering estimate
        for j in range(NPART):
            weights[i + j * NOBS] = shifted_weights[j] / norm_factor
            if isfinite(weights[i + j * NOBS] * particles[i + j * NOBS]) != 0:
                filt_state_est[i] += weights[i + j * NOBS] * particles[i + j * NOBS]

        # Compute smoothed state
        if i >= LAG:
            for j in range(NPART):
                curr_particle = particle_history[(LAG - 1) + j * LAG]
                next_particle = particle_history[(LAG - 2) + j * LAG]

                smo_state_est[i - LAG + 1] += weights[i + j * NOBS] * curr_particle

                state_quad_term = next_particle - mu - phi * (curr_particle - mu)
                state_quad_term -= sigmav * rho * exp(-0.5 * curr_particle) * obs[i - LAG]

                sub_gradient[0] = q_matrix * state_quad_term * (1.0 - phi)
                sub_gradient[1] = q_matrix * state_quad_term * (curr_particle - mu) * (1.0 - phi**2)
                sub_gradient[2] = q_matrix * state_quad_term * state_quad_term - 1.0
                sub_gradient[2] += q_matrix * state_quad_term * sigmav * rho * exp(-0.5 * curr_particle) * obs[i - LAG]
                sub_gradient[3] = rho
                sub_gradient[3] -= q_matrix * rho * state_quad_term * state_quad_term
                sub_gradient[3] += q_matrix * state_quad_term * sigmav * exp(-0.5 * curr_particle) * obs[i - LAG] * rho_term

                gradient[0][i - LAG + 1] += sub_gradient[0] * weights[i + j * NOBS]
                gradient[1][i - LAG + 1] += sub_gradient[1] * weights[i + j * NOBS]
                gradient[2][i - LAG + 1] += sub_gradient[2] * weights[i + j * NOBS]
                gradient[3][i - LAG + 1] += sub_gradient[3] * weights[i + j * NOBS]

        # Estimate log-likelihood
        log_like += max_weight + log(norm_factor) - log(NPART)

    # Estimate gradients of the log joint distribution
    for i in range(NOBS - LAG, NOBS):
        idx  = NOBS - i - 1
        for j in range(NPART):
            curr_particle = particle_history[idx + j * LAG]
            smo_state_est[i] +=  weights[NOBS - 1 + j * NOBS] * curr_particle

            if (idx - 1) >= 0:
                next_particle = particle_history[idx - 1 + j * LAG]
                state_quad_term = next_particle - mu - phi * (curr_particle - mu)
                state_quad_term -= sigmav * rho * exp(-0.5 * curr_particle) * obs[i - 1]

                sub_gradient[0] = q_matrix * state_quad_term * (1.0 - phi)
                sub_gradient[1] = q_matrix * state_quad_term * (curr_particle - mu) * (1.0 - phi**2)
                sub_gradient[2] = q_matrix * state_quad_term * state_quad_term - 1.0
                sub_gradient[2] += q_matrix * state_quad_term * sigmav * rho * exp(-0.5 * curr_particle) * obs[i - 1]
                sub_gradient[3] = rho
                sub_gradient[3] -= q_matrix * rho * state_quad_term * state_quad_term
                sub_gradient[3] += q_matrix * state_quad_term * sigmav * exp(-0.5 * curr_particle) * obs[i - 1] * rho_term

                gradient[0][i - LAG + 1] += sub_gradient[0] * weights[i + j * NOBS]
                gradient[1][i - LAG + 1] += sub_gradient[1] * weights[i + j * NOBS]
                gradient[2][i - LAG + 1] += sub_gradient[2] * weights[i + j * NOBS]
                gradient[3][i - LAG + 1] += sub_gradient[3] * weights[i + j * NOBS]

    # Sample trajectory
    idx = sampleParticle_corr(weights, rvr[0])
    for i in range(NOBS):
        j = ancestry[i + idx * NOBS]
        state_trajectory[i] = particles[i + j * NOBS]

    free(particles)
    free(weights)
    free(weights_at_t)
    free(particle_history)
    free(old_particle_history)
    free(ancestors)
    free(unnorm_weights)
    free(shifted_weights)
    free(ancestry)
    free(old_ancestry)

    # Compile the rest of the output
    return filt_state_est, smo_state_est, log_like, gradient, state_trajectory

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double norm_logpdf(double x, double m, double s):
    """Helper for computing the log of the Gaussian pdf."""
    cdef double part1 = -0.91893853320467267 # -0.5 * log(2 * pi)
    cdef double part2 = -log(s)
    cdef double part3 = -0.5 * (x - m) * (x - m) / (s * s)
    return part1 + part2 + part3

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void systematic(int *ancestors, double weights[NPART]):
    cdef int cur_idx = 0
    cdef int j = 0
    cdef double rnd_number = random_uniform()
    cdef double cpoint = 0.0
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
        cpoint = (rnd_number + j) / NPART
        while cum_weights[cur_idx] < cpoint and cur_idx < NPART - 1:
            cur_idx += 1
        ancestors[j] = cur_idx

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void systematic_corr(int *ancestors, double weights[NPART], double rnd_number):
    cdef int cur_idx = 0
    cdef int j = 0
    cdef double cpoint = 0.0
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
        cpoint = (rnd_number + j) / NPART
        while cum_weights[cur_idx] < cpoint and cur_idx < NPART - 1:
            cur_idx += 1
        ancestors[j] = cur_idx

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX

@cython.cdivision(True)
@cython.boundscheck(False)
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    return x1 * w

@cython.boundscheck(False)
cdef double my_max(double weights[NPART]):
    cdef int idx = 0
    cdef int i = 0
    cdef double current_largest = weights[0]

    for i in range(1, NPART):
        if weights[i] > current_largest and isfinite(weights[i]):
            idx = i
    return weights[idx]

@cython.boundscheck(False)
cdef int my_min(int x, int y):
    cdef int foo
    if x > y or x == y:
        foo = x
    else:
        foo = y
    return foo

@cython.cdivision(True)
@cython.boundscheck(False)
cdef int sampleParticle(double weights[NPART]):
    cdef int cur_idx = 0
    cdef int j = 0
    cdef double rnd_number = random_uniform()
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
