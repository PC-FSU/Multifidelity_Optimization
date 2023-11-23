from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.optim import optimize_acqf_cyclic
from utilis import *
torch.set_printoptions(precision=3, sci_mode=False)
from config import * 



def get_mfMes(model, bounds, cost_aware_utility, test_prob):
    """
    Get the Max Value Entropy Acquisition Function.
    
    Args:
    model: The Gaussian Process model.
    bounds: Bounds for optimization.
    cost_aware_utility: InverseCostWeight utility for BO.
    test_prob (dict): Dictionary containing information about the test problem.
    
    Returns:
    qMultiFidelityMaxValueEntropy: Max Value Entropy Acquisition Function instance.
    """
    
    candidate_set = torch.rand(1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
    
    target_fidelities = get_target_fidelities(test_prob)
    project = lambda X: project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
    
    qMES = qMultiFidelityMaxValueEntropy(
        model=model, 
        candidate_set=candidate_set,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )
    return qMES.double()


def optimize_mfmes_and_get_observation(Mfmes_acqf, bounds, q, cost_model, test_prob, single_fidelity):
    """
    Optimize the Max Value Entropy Acquisition Function (MFMES) and return a new candidate, observation, and cost.

    Args:
    Mfmes_acqf: Max Value Entropy Acquisition Function instance.
    bounds: Bounds for optimization.
    q (int): Number of points selected in the optimization.
    cost_model: Cost model for optimization.
    test_prob (dict): Dictionary containing information about the test problem.
    single_fidelity (bool): If True, consider a single fidelity.

    Returns:
    new_x: New candidate points after optimization.
    new_obj: New observations for the candidate points.
    cost: Total cost incurred during optimization.
    """

    if not single_fidelity:
        candidates_q2_cyclic, acq_value_q2_cyclic = optimize_acqf_cyclic(
            acq_function=Mfmes_acqf.double(),
            bounds=bounds,
            q=q,
            num_restarts=10,
            raw_samples=512,
            cyclic_options={"maxiter": 20},
        )
    else:
        candidates_q2_cyclic, acq_value_q2_cyclic = optimize_acqf_cyclic(
            acq_function=Mfmes_acqf.double(),
            bounds=bounds,
            q=q,
            num_restarts=10,
            raw_samples=512,
            cyclic_options={"maxiter": 20},
            fixed_features={test_prob['fx'].dim - 1: 1.0},
        )

    # Observe new values
    cost = cost_model(candidates_q2_cyclic).sum()
    new_x = candidates_q2_cyclic.detach()
    new_obj = test_prob['fx'](new_x).unsqueeze(-1)

    print(f"Candidates:\n{new_x}\n", flush=True)
    print(f"Observations:\n{new_obj}\n\n", flush=True)
    return new_x.double(), new_obj.double(), cost.double()