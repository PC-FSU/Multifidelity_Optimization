from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from utilis import *
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions


def get_mfkg(model, bounds, cost_aware_utility, test_prob):
    """
    Get the Multi-Fidelity Knowledge Gradient (MFKG) acquisition function.

    Args:
    model: Model to be used in acquisition.
    bounds: Bounds for optimization.
    cost_aware_utility: InverseCostWeight utility for MFKG.
    test_prob (dict): Dictionary containing information about the test problem.

    Returns:
    qMultiFidelityKnowledgeGradient: MFKG acquisition function instance.
    """

    dim = test_prob['fx'].dim
    columns = [test_prob['fx'].dim - 1]

    target_fidelities = get_target_fidelities(test_prob)
    project = lambda X: project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dim,
        columns=columns,
        values=[1],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 10, "maxiter": 200},
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=32,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_mfkg_and_get_observation(mfkg_acqf, bounds, q, cost_model, test_prob, single_fidelity):
    """
    Optimize the Multi-Fidelity Knowledge Gradient (MFKG) and return a new candidate, observation, and cost.

    Args:
    mfkg_acqf: MFKG acquisition function instance.
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

    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=mfkg_acqf,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    if not single_fidelity:
        candidates, _ = optimize_acqf(
            acq_function=mfkg_acqf,
            bounds=bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
        )
    else:
        candidates, _ = optimize_acqf(
            acq_function=mfkg_acqf,
            bounds=bounds,
            q=q,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
            fixed_features={test_prob['fx'].dim - 1: 1.0},
        )

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = test_prob['fx'](new_x).unsqueeze(-1)

    print(f"Candidates:\n{new_x}\n", flush=True)
    print(f"Observations:\n{new_obj}\n\n", flush=True)
    return new_x, new_obj, cost
