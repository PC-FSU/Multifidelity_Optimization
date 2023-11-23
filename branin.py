from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor
from scipy.stats import multivariate_normal

class AugmentedBranin_modified(SyntheticTestFunction):
    r"""Augmented Branin test function for multi-fidelity optimization.

    3-dimensional function with domain `[-5, 10] x [0, 15] * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = (x_2 - (b - 0.1 * (1 - x_3))x_1^2 + c x_1 - r)^2 +
            10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`.
    B has infinitely many minimizers with `x_1 = -pi, pi, 3pi`
    and `B_min = 0.397887`
    """

    dim = 3
    _bounds = [(-5.0, 10.0), (0.0, 15.0), (0.0, 1.0)]
    _optimal_value = 0.397887
    _optimizers = [  # this is a subset, ther are infinitely many optimizers
        (-math.pi, 12.275, 1),
        (math.pi, 1.3867356039019576, 0.1),
        (math.pi, 1.781519779945532, 0.5),
        (math.pi, 2.1763039559891064, 0.9),
    ]
    
    SD = 1.5
    Scale = 10
    
    def set_SD(self,SD):
        self.SD = SD
        print("\nSetting the SD to : %f\n"%self.SD)
    
    def set_Scale(self,Scale):
        self.Scale = Scale
        print("\nSetting the Scale to : %f\n"%self.Scale)
        
    def evaluate_true(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - (5.1 / (4 * math.pi ** 2) - 0.1 * (1 - X[:, 2])) * X[:, 0] ** 2
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        
        original_branin = t1 ** 2 + t2 + 10
        
        domain_corner  = torch.tensor([2.5,7.5],   dtype=torch.double)
        global_max     = torch.tensor([9.425,2.475],dtype=torch.double)

        mean           = domain_corner - (domain_corner -  global_max)*X[:,2].unsqueeze(-1)
        mean           = mean.numpy()
        cov            = [[self.SD,0],[0,self.SD]]
        points         = X[:,0:2].numpy()
        gaussian_weights = torch.tensor([0]*X.size()[0], dtype=torch.double)

        for idx,val in enumerate(points):
            point   = points[idx]
            mean_pt = mean[idx]
            gaussian_weights[idx] = multivariate_normal.pdf(point,mean_pt,cov)
            
        #we are working with inverted branin, so negate the original branin and add weigths    
        modified_branin = -1*original_branin + self.Scale * gaussian_weights
        
        return modified_branin