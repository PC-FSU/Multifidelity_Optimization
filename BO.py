import numpy as np
import sys
import torch
from botorch import fit_gpytorch_model
from botorch.utils.transforms import standardize
from utilis import *
from GP import *
from qMES import *
from qMFKG import *
torch.set_printoptions(precision=3, sci_mode=False)
from config import *


def run_BO(path=None, n_iter=None, save_result=None, aqf_selected=None, load_previous=None, test_prob=None, cost_model=None, cost_aware_utility=None, q=None, single_fidelity=None):
    """
    Run Bayesian Optimization routine.

    Args:
    path (str): Path to the folder where the data will be stored.
    n_iter (int): Number of iterations for Bayesian Optimization.
    save_result (bool): If True, save the result of the experiment.
    aqf_selected (str): Acquisition function used for Bayesian Optimization.
    load_previous (bool): If True, restart the last experiment from where it was stopped.
    test_prob: Problem for which BO is going to run.
    cost_model: Cost model for BO.
    cost_aware_utility: InverseCostWeight utility for BO.
    q (int): Number of points selected in one iteration.
    single_fidelity (bool): If True, run for a Fidelity=1 only.

    Returns:
    None
    """

    # Generate the training data.
    train_x, train_obj, bounds = generate_initial_data(test_prob, n=4)
    cumulative_cost = 0.0
    start_iter = 0
    end_iter = n_iter

    # If not Load_Previos, create a file for a fresh experiment.
    if save_result and not load_previous:
        fPath = os.path.join(path, str(0))
        print(f"fpath {fPath}\n\n", flush=True)
        os.makedirs(fPath)
        os.makedirs(os.path.join(fPath, "model"))
        np.savetxt(os.path.join(fPath, "train_x_inital.txt"), train_x)
        np.savetxt(os.path.join(fPath, "train_obj_inital.txt"), train_obj)


    # if Load_Previus load the path, and training data, and find start_iter, end_iter and cumulative_cost
    if save_result and load_previous:
        fPath = os.path.join(path, str(0))

        # These points serve as the starting inputs for the beginning of the Bayesian Optimization (BO) process
        train_x_inital = np.loadtxt(os.path.join(fPath, "train_x_inital.txt"))
        # Point sampled during the Bayesian Optimization (BO) process.
        train_x_saved = np.loadtxt(os.path.join(fPath, "train_X.txt")).reshape(-1, train_x_inital.shape[1])
        # Make the complete training set
        train_x = np.concatenate((train_x_inital, train_x_saved), axis=0)
        train_x = torch.tensor(train_x, **tkwargs)

        # Get the initial objective values, and objective value on points sampled during BO.
        train_obj_inital = np.loadtxt(os.path.join(fPath, "train_obj_inital.txt")).flatten()
        train_obj_saved = np.loadtxt(os.path.join(fPath, "train_obj.txt")).flatten()

        # Make the complete objective
        train_obj = np.concatenate((train_obj_inital, train_obj_saved), axis=0)
        train_obj = torch.tensor(train_obj, **tkwargs)
        train_obj = train_obj.unsqueeze(-1)  # Add output dimension

        # Get the iter # where the experiment stopped last
        start_iter = len(train_obj) - len(train_obj_inital)
        end_iter = n_iter

        cumulative_cost = np.sum(np.loadtxt(os.path.join(fPath, "cost.txt")))
        print("\nStarting at iteration: %d\n" % start_iter, "\n", flush=True)

    # ********************************* Main BO routine ************************************************************
    for i in range(start_iter, end_iter):

        # Get and fit a new GP model after every iteration.
        print(f"Routine Running for Iteration: {i}\n", flush=True)
        mll, model = initialize_model(train_x, standardize(train_obj), test_prob)
        model.to(dtype=torch.double)
        fit_gpytorch_model(mll)

        # Get the new point to inculde in training set, along with cost required to sample the point, and corresponding objective value.
        if aqf_selected == 'qMultiFidelityMaxValueEntropy':
            mfmes_acqf = get_mfMes(model, bounds, cost_aware_utility, test_prob)
            new_x, new_obj, cost = optimize_mfmes_and_get_observation(mfmes_acqf, bounds, q, cost_model, test_prob, single_fidelity)
        else:
            mfkg_acqf = get_mfkg(model, bounds, cost_aware_utility, test_prob)
            new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf, bounds, q, cost_model, test_prob, single_fidelity)
        
        # Update the training set, and objective.
        train_x = torch.cat([train_x, normalize(new_x, bounds=bounds)])
        train_obj = torch.cat([train_obj, new_obj])
        cumulative_cost += cost

        print(f"cost : {cost}\n", flush=True)

        # The objective evaluated at highest fidelity for the point sampled during the current iteration.
        eval_high = eval_highest(test_prob=test_prob, x=new_x)
        # The maximizer and maximum value of the GP posterior
        final_rec, final_obj = get_recommendation(model, bounds, test_prob)

        print(eval_high, final_rec, final_obj)

        if save_result:
            # Save new candidate
            with open(os.path.join(fPath, "train_X.txt"), "ab") as f:
                np.savetxt(f, new_x.numpy())

            # Save new objective
            with open(os.path.join(fPath, "train_obj.txt"), "ab") as f:
                np.savetxt(f, new_obj.numpy())

            # Save new cost
            with open(os.path.join(fPath, "cost.txt"), "ab") as f:
                np.savetxt(f, [cost.numpy()])

            # Save the value evaluated at highest fidelity
            with open(os.path.join(fPath, "metric_1.txt"), "ab") as f:
                np.savetxt(f, eval_high.numpy())

            with open(os.path.join(fPath, "metric_2.txt"), "ab") as f:
                np.savetxt(f, final_rec.numpy())
                np.savetxt(f, final_obj.numpy())

            # if i % 5 == 0:
            print(f"Saving the model at iteration: {i}\n", flush=True)

            torch.save(model.state_dict(), os.path.join(fPath, "model", 'model_state' + '_' + str(i) + '.pth'))

    print(f"\ntotal cost: {cumulative_cost}\n", flush=True)

   
