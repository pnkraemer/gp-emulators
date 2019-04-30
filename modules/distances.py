"""
NAME: distances.py

PURPOSE: Distance functions (Hellinger, RMSE, ...) to measure errors
"""
import numpy as np
from pointsets import *

class Distance():
    pass

class RMSE(Distance):
    
    
    """
    RMSE.random takes functions which take (only) pointsets as input and have 1d output
    """
    @staticmethod
    def random(truth, function, num_evals = 999, eval_dim = 1):
        eval_ptset = Random.construct(num_evals, eval_dim)
        return np.linalg.norm(truth(eval_ptset) - approximation(eval_ptset)) / np.sqrt(num_evals)
