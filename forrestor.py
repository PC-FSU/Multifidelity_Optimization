import math
from typing import Optional

import torch
from torch import Tensor
import numpy as np
from botorch.test_functions.synthetic import SyntheticTestFunction
from scipy.stats import multivariate_normal

class AugmentedForrestor(SyntheticTestFunction):
    
    r"""Augmented Forrestor test function for multi-fidelity optimization.

    2-dimensional function with domain `[0, 1] * [0,1]`, where
    the last dimension of is the fidelity parameter:

  
    f(x) = -( (6x-2)*sin(12x-4) + (10-s*10)(x-0.5) + 20(1-s) )
    s is the fidelity.
    optimal_value = 6.02
    optimizer = 0.7587
    
    """

    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _optimal_value = 6.02
    _optimizers = [0.7587]

    def evaluate_true(self, X: Tensor) -> Tensor:     
        x = X[:,0]
        s = X[:,1]
        temp = -1*(torch.square(6*x-2)*torch.sin(12*(x)-4) + (10-s*10)*(x-0.5) + 20*(1-s))
        return temp
    
class AugmentedForrestor_Negate(SyntheticTestFunction):
    
    r"""Augmented Forrestor test function for multi-fidelity optimization.

    2-dimensional function with domain `[0, 1] * [0,1]`, where
    the last dimension of is the fidelity parameter:

  
    f(x) = -( (6x-2)*sin(12x-4) + (10-s*10)(x-0.5) + 20(1-s) )
    s is the fidelity.
    optimal_value = 6.02
    optimizer = 0.7587
    
    """

    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _optimal_value = 6.02
    _optimizers = [0.7587]

    def evaluate_true(self, X: Tensor) -> Tensor:     
        x = X[:,0]
        s = X[:,1]
        s = 1 - s
        temp = -1*(torch.square(6*x-2)*torch.sin(12*(x)-4) + (10-s*10)*(x-0.5) + 20*(1-s))
        return temp    
    
    