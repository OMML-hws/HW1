
from scipy.stats import truncnorm
import numpy as np

def get_truncated_normal(dict,n_sample):
    np.random.seed(7)
    low      = dict["low"]
    upp      = dict["upp"]
    mean     = dict["mean"]
    sd       = dict["sd"]

    truncnorm_obj =  truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return truncnorm_obj.rvs(n_sample)