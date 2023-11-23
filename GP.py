from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP,FixedNoiseMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
import torch 
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    _GaussianLikelihoodBase,
)
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.constraints.constraints import GreaterThan


def initialize_model(train_x, train_obj, Test_Prob):
    """
    Initialize the Gaussian Process model for multi-fidelity optimization.

    Args:
    train_x (tensor): Training input data.
    train_obj (tensor): Training objective data.
    Test_Prob: Test problem for which the model is being initialized.

    Returns:
    ExactMarginalLogLikelihood: Marginal log likelihood of the model.
    SingleTaskMultiFidelityGP: Multi-fidelity Gaussian Process model.
    """

    # Extract information from the test problem
    data_fidelity = Test_Prob['fx'].dim - 1
    bounds = torch.tensor(Test_Prob['fx']._bounds).T

    # Specify the likelihood
    noise_prior = GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    MIN_INFERRED_NOISE_LEVEL = 1e-3  # Increase from default due to cholesky issue.
    
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
    
    # Create the SingleTaskMultiFidelityGP model
    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        outcome_transform=Standardize(m=1),
        data_fidelity=data_fidelity,
        linear_truncated=False,
        likelihood=likelihood
    )
    
    # Compute the Exact Marginal Log Likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    return mll, model




# def initialize_model(train_x, train_obj, Test_Prob):

#     data_fidelity = Test_Prob['fx'].dim - 1
#     bounds = Test_Prob['fx']._bounds
#     bounds    = torch.tensor(bounds).T
    
    
#     #specify the likelihood
#     noise_prior = GammaPrior(1.1, 0.05)
#     noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
#     MIN_INFERRED_NOISE_LEVEL = 1e-3 # increase from default due to cholesky issue.
#     likelihood = GaussianLikelihood(
#         noise_prior=noise_prior,
#         noise_constraint=GreaterThan(
#             MIN_INFERRED_NOISE_LEVEL,
#             transform=None,
#             initial_value=noise_prior_mode,
#         ),
#     )
    
#     model = SingleTaskMultiFidelityGP(
#         train_x, 
#         train_obj, 
#         outcome_transform=Standardize(m=1),
#         data_fidelity=data_fidelity,
#         linear_truncated = False,
#         likelihood = likelihood
#     )
    
#     mll = ExactMarginalLogLikelihood(model.likelihood, model)
#     return mll, model



    '''
    model = FixedNoiseMultiFidelityGP(
        train_X = train_x, 
        train_Y = train_obj, 
        train_Yvar = torch.full_like(train_obj,0.001),
        outcome_transform=Standardize(m=1),
        #An outcome transform that is applied to the training data during instantiation and to
        #the posterior during inference (that is, the Posterior obtained by calling .posterior 
        #                                on the model will be on the original scale).
        data_fidelity=data_fidelity  #The column index for the downsampling fidelity parameter (optional)    
    )
    '''