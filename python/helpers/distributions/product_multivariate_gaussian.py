"""Helpers for the product of two multivariate Gaussian distributions."""
import numpy as np
from scipy.stats import multivariate_normal as mvn

def compute_statistics(mean1, mean2, cov_matrix1, cov_matrix2):
    inverse_term = np.linalg.inv(cov_matrix1 + cov_matrix2)
    cov_matrix = cov_matrix1 @ inverse_term @ cov_matrix2
    mean = cov_matrix2 @ inverse_term @ mean1
    mean += cov_matrix1 @ inverse_term @ mean2
    return mean, cov_matrix

def logpdf(parm, mean1, mean2, cov_matrix1, cov_matrix2):
    mean, cov_matrix = compute_statistics(mean1, mean2, cov_matrix1, cov_matrix2)
    return mvn.logpdf(parm, mean, cov_matrix)

def rvs(mean1, mean2, cov_matrix1, cov_matrix2):
    mean, cov_matrix = compute_statistics(mean1, mean2, cov_matrix1, cov_matrix2)
    return mvn.rvs(mean, cov_matrix)

