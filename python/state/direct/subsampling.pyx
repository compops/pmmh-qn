from __future__ import absolute_import
import cython
from libc.stdlib cimport malloc, free

DEF NOBS = 110000
DEF NPART = 5500

@cython.cdivision(True)
@cython.boundscheck(False)
def get_settings():
    return NOBS, NPART

@cython.cdivision(True)
@cython.boundscheck(False)
def multinomial(double [:] rnd_number):
    cdef int cur_idx, j
    cdef double *cum_weights = <double *>malloc(NOBS * sizeof(double))
    cdef int[NPART] indices

    for i in range(NOBS):
        cum_weights[i] = (i + 1.0) / NOBS

    cur_idx = 0
    for j in range(NPART):
        while cum_weights[cur_idx] < rnd_number[j] and cur_idx < NOBS - 1:
            cur_idx += 1
        indices[j] = cur_idx

    free(cum_weights)
    return indices

@cython.cdivision(True)
@cython.boundscheck(False)
def stratified(double [:] rnd_number):
    cdef int cur_idx, j
    cdef double cpoint
    cdef double *cum_weights = <double *>malloc(NOBS * sizeof(double))
    cdef int[NPART] indices

    for i in range(NOBS):
        cum_weights[i] = (i + 1.0) / NOBS

    cur_idx = 0
    for j in range(NPART):
        cpoint = (rnd_number[j] + j) / NPART
        while cum_weights[cur_idx] < cpoint and cur_idx < NOBS - 1:
            cur_idx += 1
        indices[j] = cur_idx

    free(cum_weights)
    return indices


@cython.cdivision(True)
@cython.boundscheck(False)
def systematic(double rnd_number):
    cdef int cur_idx, j
    cdef double cpoint
    cdef double *cum_weights = <double *>malloc(NOBS * sizeof(double))
    cdef int[NPART] indices

    for i in range(NOBS):
        cum_weights[i] = (i + 1.0) / NOBS

    cur_idx = 0
    for j in range(NPART):
        cpoint = (rnd_number + j) / NPART
        while cum_weights[cur_idx] < cpoint and cur_idx < NOBS - 1:
            cur_idx += 1
        indices[j] = cur_idx

    free(cum_weights)
    return indices
