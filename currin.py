import math
from typing import Optional

import torch
from torch import Tensor
import numpy as np
from botorch.test_functions.synthetic import SyntheticTestFunction
from scipy.stats import multivariate_normal

class AugmentedCurrin(SyntheticTestFunction):
    
    r"""Augmented Currin test function for multi-fidelity optimization.

    3-dimensional function with domain `[0, 1] x [0, 1] * [0,1]`, where
    the last dimension of is the fidelity parameter:

        a(x) = (1-e(-1/2x_2))*( (2300*x_1^3 + 1900*x_1^2 + 2092*x_1 + 60)/(100*x_1^3 + 500*x_1^2 + 4x_1 + 20))
        f(x) = PDF( a(x), N(x3,[[1.5,0],[0,1.5]]) )
        
    optimal_value = 119.901
    optimizer = [0.216,0]
    
    
    """

    dim = 3
    _bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    _optimal_value = 119.901
    _optimizers = [(0.2160804, 0),]

    def evaluate_true(self, X: Tensor) -> Tensor:
     
        first_term  = (1 - torch.exp(-1/(2*X[:,1])))
        second_term = 2300*torch.pow(X[:,0],3) + 1900*torch.pow(X[:,0],2) + 2092*X[:,0] + 60
        third_term  = 100*torch.pow(X[:,0],3)  + 500*torch.pow(X[:,0],2)  + 4*X[:,0]    + 20

        original_currin = (first_term * second_term) / (third_term)

        domain_corner  = torch.tensor([1,1],   dtype=torch.double)
        global_max     = torch.tensor([0.21,0],dtype=torch.double)

        mean           = 1 - (domain_corner -  global_max)*X[:,2].unsqueeze(-1)
        mean           = mean.numpy()
        cov            = [[1.5,0],[0,1.5]]
        points         = X[:,0:2].numpy()
        gaussian_weights = torch.tensor([0]*X.size()[0], dtype=torch.double)

        for idx,val in enumerate(points):
            point   = points[idx]
            mean_pt = mean[idx]
            gaussian_weights[idx] = multivariate_normal.pdf(point,mean_pt,cov)

        modified_currin = original_currin + 1000. * gaussian_weights
        return modified_currin/100
