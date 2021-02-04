import numpy as np
import sys
import os
norta_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(norta_path, os.pardir))
sys.path.append(parent_path)
from norta import fit_NORTA as fit_NORTA
"""
Example of using NORTA
"""
np.random.seed(0)
n_sample = 100
d_sample = 10
cov_sample = np.eye(d_sample) + np.random.rand(d_sample, d_sample)
sim_cov = cov_sample.transpose().dot(cov_sample)
data = np.random.exponential(size=(n_sample, d_sample)) + np.random.multivariate_normal(
    np.zeros(d_sample), sim_cov, size=n_sample)
n = len(data)
d = len(data[0])
norta_data = fit_NORTA(data, n, d, lambda_param=0.05, mc_samples=1E6, output_flag=1, n_proc=4)
NG = norta_data.gen(1000)
print(NG.mean(axis=0), data.mean(axis=0))
print(np.corrcoef(NG, rowvar=0))
print(np.corrcoef(data, rowvar=0))
print(np.cov(NG, rowvar=False))
print(np.cov(data, rowvar=False))
