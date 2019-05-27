"""
Created on Mon May 27 12:45:18 2019

P. W. Davenport-Jenkins
University of Manchester
MSc Econometrics
"""

from numba import jit
import scipy.optimize as optimize
from functools import partial
import scipy.stats as stats
import numpy as np

NO_PMC = 6


def make_grid(anchor, mesh_size, size):

    mu = anchor[0]
    sigma = anchor[1]
    mu_upper = [mu + mesh_size * i for i in range(size)]
    mu_lower = [mu - mesh_size * (1+i) for i in range(size)]
    mu_array = list(reversed(mu_lower)) + mu_upper
    sigma_upper = [sigma + mesh_size * i for i in range(size)]
    sigma_lower = [sigma - mesh_size * (1+i) for i in range(size)]
    sigma_array = list(reversed(sigma_lower)) + sigma_upper

    return [mu_array, sigma_array]


# An m vector of moment restrictions
# defined by the user.
def moment_conditions_matrix(beta, z):
    """
    There are m moment restrictions.
    There are N data points, called z.
    For each data point the m moment restrictions are computed.
    The result is an N x m matrix with each column representing
    a different moment restriction.
    """
    # In this example we do the normal distribution with location 3 and
    # scale 5.
    # central moments are defined as E[(z-mu)^k]
    mu = beta[0]
    sigma = beta[1]
    moments_matrix = np.concatenate(
        (
            z - mu,
            sigma**2 - (z - mu)**2,
            (z - mu)**3,
            3 * sigma**4 - (z - mu)**4,
            (z - mu)**5,
            15 * sigma**6 - (z - mu)**6
        ),
        axis=1
    )

    return moments_matrix


# This oculd perhaps be done more efficiently.
# einstien notation?
def GMM_weighting_matrix(moment_conditions_matrix):

    number_of_restrictions = len(moment_conditions_matrix[0])

    sample_size = len(moment_conditions_matrix)

    omega = np.zeros((number_of_restrictions, number_of_restrictions))

    for moment_vector in moment_conditions_matrix:

        omega = omega + np.outer(moment_vector, moment_vector)

    return np.linalg.inv(omega/sample_size)


def GMM_objective_function(x, sample, weighting_matrix):

    sample_moments_vector = moment_conditions_matrix(x, sample).mean(axis=0)

    return sample_moments_vector.T @ weighting_matrix @ sample_moments_vector


def GMM(beta_initial, sample, number_of_moment_conditions):

    # First Stage: calculate initial beta
    beta_1 = optimize.minimize(
            GMM_objective_function,
            beta_initial,
            args=(sample, np.identity(number_of_moment_conditions)),
            method="BFGS"
    )

    beta_1 = beta_1.x

    # Use this value to compute the optimal weighting matrix
    weighting_matrix = GMM_weighting_matrix(
                                moment_conditions_matrix(beta_1, sample)
    )

    # Second stage:: use the optimal weighting matrix to compute 2S-GMM
    # estimator of beta
    beta_2 = optimize.minimize(
            GMM_objective_function,
            beta_1,
            args=(sample, weighting_matrix),
            method="BFGS"
    )

    return beta_2.x


def ET(value):

    return -np.exp(value)


def ET_cost_function(moments, x):

    return -np.mean(ET(moments @ x))


def GEL_ET(sample, beta, number_of_moment_conditions):

    """
    Here data_matrix is a 1000 x 6 matrix
    """
    moments_matrix = moment_conditions_matrix(beta, sample)

    # Small \lambda gives small \nu. Near 0 should be feasible.
    initial_params = np.zeros(number_of_moment_conditions)

    cost = partial(ET_cost_function, moments_matrix)

    result = optimize.minimize(
        cost,
        initial_params,
        method="Nelder-Mead",
        options={'maxiter': 1000}
    )
    return result.x


@jit
def brute_force(sample, beta_grid, number_of_moment_conditions):
    mu_list = beta_grid[0]
    sigma_list = beta_grid[1]
    mu_size = len(mu_list)
    sigma_size = len(sigma_list)
    lambda_dictionary = dict()
    for i in range(mu_size):
        for j in range(sigma_size):
            beta = [mu_list[i], sigma_list[j]]
            lambda_beta = GEL_ET(sample, beta, number_of_moment_conditions)
            lambda_dictionary[tuple(beta)] = lambda_beta

    objective_dict = dict()
    for i in range(mu_size):
        for j in range(sigma_size):
            beta = [mu_list[i], sigma_list[j]]
            moments = moment_conditions_matrix(beta, sample).mean(axis=0)
            lambda_beta = lambda_dictionary[tuple(beta)]
            value = -ET_cost_function(moments, lambda_beta)
            objective_dict[tuple(beta)] = value

    return objective_dict, lambda_dictionary


normal_random_variable = stats.norm(3, 5)

# generate n random values from above distribution
sample = normal_random_variable.rvs(size=(100, 1))

initial_beta = [2, 4]

beta_GMM = GMM(initial_beta, sample, NO_PMC)

# increasing the last input parameter significantly increases cost
beta_grid = make_grid(beta_GMM, 0.005, 10)

dict_to_max, lambda_dictionary = brute_force(sample, beta_grid, NO_PMC)

min_objective = min(dict_to_max.values())

for key, value in dict_to_in.items():
    if value == min_objective:
        beta_GEL = key

lambda_GEL = lambda_dictionary[beta_GEL]
