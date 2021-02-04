"""
Created on Tue Oct  9 12:58:58 2018
Updated on Wed Jun 19 13:14:19 2019
@author: Daniel Duque
@email: d.duque25@gmail.com
This module implements Normal-to-Anything (NORTA) algorithm
to generate correlated random vectors. The original paper is by
Cario and Nelson (2007).
"""
import numpy as np
import multiprocessing as mp
from scipy.linalg import cholesky
import scipy.stats as stats
from itertools import product

Z = stats.norm(0, 1)  # Standar normal
OUTPUT = 1  # Output flag. 0: no output, 1: process output
MC_SAMPLES = 1E6  # Number of samples to compute the integral
BIN_SEARCH_LAMBDA = 0.5  # Binary search parameter (~0 computes closer to u, and ~1 computes closer to l)
ZERO_TOL = 1e-4  # Tolerance for the binary search


def find_rho_z(rho_input):
    """
    Computes the correlation of the multivariate normal used in the
    generation of random variables. The correlation is found by means of
    a binary search where for each value, the target covariance between i and j
    is computed as an integral via Monte Carlo.
    
    Args:
    rho_input: a tuple with the folling entries:
        ij (tuple int): pair of indices for which the correlation is being computed
        F_invs (list of func): inverse functions for the marginal distribution of
        each random variable.
        CovX (ndarray): Covariance matrix of the input
        EX (ndarray): Mean vector of the input
    Returns:
        rho_z (float): correlation for (i,j) pair in NORTA
    """
    ij, F_invs, CovX, EX = rho_input
    i = ij[0]
    j = ij[1]
    if OUTPUT == 1:
        print("Computing rhoZ(%i,%i)" % (i, j))
    cor_dem = np.sqrt(CovX[i, i] * CovX[j, j])
    rho_z = CovX[i, j] / cor_dem
    rho_u = 1 if CovX[i, j] > 0 else 0
    rho_d = -1 if CovX[i, j] < 0 else 0
    F_i_inv = F_invs[i]
    F_j_inv = F_invs[j]
    EXi, EXj = EX[i], EX[j]
    while np.abs(rho_u - rho_d) > ZERO_TOL:
        covZ = np.array([[1, rho_z], [rho_z, 1]])
        EXiXj = montecarlo_integration(F_i_inv, F_j_inv, m=np.zeros(2), c=covZ, n=MC_SAMPLES)
        CXiXj = EXiXj - EXi * EXj
        if OUTPUT == 1:
            print(f"{i},{j} - rho_z={rho_z:10.4e} - C(i,j)={CXiXj:10.4e} - Cov={CovX[i, j]:10.4e}")
        if np.abs(CXiXj - CovX[i, j]) / cor_dem < ZERO_TOL:
            #
            return rho_z
        else:
            if CXiXj > CovX[i, j]:
                rho_u = rho_z
                rho_z = BIN_SEARCH_LAMBDA * rho_d + (1 - BIN_SEARCH_LAMBDA) * rho_u
            else:  # rhoC_ij <= rho_ij
                rho_d = rho_z
                rho_z = BIN_SEARCH_LAMBDA * rho_d + (1 - BIN_SEARCH_LAMBDA) * rho_u
    return rho_z


def montecarlo_integration(F_i_inv, F_j_inv, m, c, n):
    """
    Computes the integral for the particular function in NORTA.
    WARNING: This method is not general for other functions as it is.
    """
    rnd = np.random.RandomState(0)  # Random stream
    
    def f(z1, z2):
        # bi_normal is assumed in the sampling, and therefore omitted from the integrant.
        return F_i_inv(Z.cdf(z1)) * F_j_inv(Z.cdf(z2))
    
    z_trial = rnd.multivariate_normal(m, c, int(n))
    integral = np.sum(f(z_trial[:, 0], z_trial[:, 1]))
    return integral / int(n)


def fit_NORTA(data, n, d, F_invs=None, lambda_param=0.01, mc_samples=1E6, seed=0, output_flag=0, n_proc=4):
    """
    Computes covariance matrix for NORTA procedure.
    
    Args:
        data (ndarray): a n x d array with the data
        n (int): number of observations for each random variable
        d (int): dimension of the random vector.
        F_invs (list of func): optional parameter to specify the marginal
            distributions. Each function must support vector operations.
            Default is None, which constructs the marginals from the data.
        lambda_param(double): parameter used in the convex combination when ever
            the resulting matrix is not PSD.
        mc_samples(int): number of samples used in the Monte Carlo integration.
        seed (int): seed for the random generator in NORTA
        output_flag (int): 0 no output; 1 debug output
        n_proc (int): num of processor to parallelize norta
    Return:
        NORTA_GEN (NORTA): an object that stores the necessary information to
            generate NORTA random vectors.
    """
    global OUTPUT
    global MC_SAMPLES
    OUTPUT = output_flag
    MC_SAMPLES = mc_samples
    assert n_proc > 0, "Number of processors is positive and integer."
    assert len(data) == n, "Data needs to be a d x n matrix."
    assert len(data[0]) == d, "Data needs to bo a d x n matrix."
    assert 0 < lambda_param < 1, "lambda_para must be between 0 and 1."
    
    if OUTPUT == 1:
        print("Starting NORTA fitting")
        print("Finding %i correlation terms" % (int(d * (d - 1) / 2)))
    C = None  # matrix for NORTA
    
    CovX = np.cov(data, rowvar=False)
    VarX = np.diag(np.diag(CovX))
    if F_invs is None:
        F_invs = [empirical_inverse_cdf(np.sort(data[:, i])) for i in range(d)]
    procedure_done = False
    working_pool = mp.Pool(n_proc)
    while not procedure_done:
        D = np.eye(d)
        EX = np.mean(data, axis=0)
        ij_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        mp_data = product(ij_pairs, [F_invs], [CovX], [EX])
        ij_rho_z = working_pool.map(find_rho_z, mp_data)
        for ix, (i, j) in enumerate(ij_pairs):
            D[i, j] = ij_rho_z[ix]
            D[j, i] = D[i, j]
        
        try:
            C = cholesky(D, lower=True)
            procedure_done = True
        except np.linalg.LinAlgError:
            CovX = (1 - lambda_param) * CovX + lambda_param * VarX
            if OUTPUT == 1:
                print("Cholesky factorization failed, starting over with a modified cov matrix.")
    working_pool.terminate()
    NORTA_GEN = NORTA(F_invs, C, seed)
    return NORTA_GEN


class empirical_inverse_cdf():
    """
    Builds an inverse CDF function given a sorted vector of values defining a 
    marginal distribution.
    
    Args:
        X: Sorted vector of observations of a single random random variable.
    """
    def __init__(self, X):
        self.n = len(X)
        self.X = X
    
    def __call__(self, prob):
        """
        Args:
            prob (ndarray): vector with probabilities to compute the inverse
        """
        # assert 0<=prob<=1, 'Argument of inverse function is a probability >=0 and <= 1.'
        X = self.X
        n = self.n
        return X[np.minimum((n * np.array(prob)).astype(int), n - 1)]


class NORTA:
    """
        Class to create a Normal-to-Anything model
        Attributes:
        Finv (list of func): inverse CDFs (vectorized) of the marginals
        C (ndarray): numpy array with the Cholesky factorization matrix
            that defines the covariance of the variables.
    """
    def __init__(self, Finv, C, seed=0):
        assert len(Finv) == len(C), "Dimension of the marginals and C do not match."
        self.F_inv = Finv
        self.C = C
        self.rnd = np.random.RandomState(seed)  # Random stream
    
    def reset_seed(self, seed):
        self.rnd = np.random.RandomState(seed)  # Random stream
    
    def gen(self, n=1):
        """
        Generates an array of vectors where each component follow the marginal
        distribution and the realizations are correlated by means of the
        covariance matrix CovZ computed in the fitting process.
        
        Args:
            n (int): number of samples (vectors) to generate
        """
        d = len(self.F_inv)
        w = self.rnd.normal(size=(d, n))
        z = self.C.dot(w)
        
        X = np.array([self.F_inv[i](Z.cdf(z[i])) for i in range(d)]).transpose()
        return X
