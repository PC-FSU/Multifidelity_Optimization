from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition import PosteriorMean
import torch
import os
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import normalize
import numpy as np
import matplotlib.pyplot as plt
from config import *

torch.set_printoptions(precision=3, sci_mode=False)
SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
#torch.manual_seed(0)
print(NUM_RESTARTS, RAW_SAMPLES)



def generate_initial_data(test_prob, n=16):
    '''
    Generate the training data.

    Args:
    test_prob (dict): Dictionary containing information about the test problem.
    n (int, optional): Number of training samples to generate. Default is 16.

    Returns:
    torch.Tensor, torch.Tensor, torch.Tensor: Generated training data, objective values, and bounds.
    '''

    dim = test_prob['fx'].dim
    
    bounds = test_prob['fx']._bounds
    bounds = torch.tensor(bounds, **tkwargs).T
    
    # Generate training data
    train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, dim, **tkwargs)
    train_obj = test_prob['fx'](train_x).unsqueeze(-1)  # Add output dimension
    train_x = normalize(train_x, bounds=bounds)

    return train_x, train_obj, bounds


def get_target_fidelities(test_prob):
    '''
    Get target fidelity of 1.

    Args:
    test_prob (dict): Dictionary containing information about the test problem.

    Returns:
    dict: Target fidelities.
    '''
    target_fidelities = {test_prob['fx'].dim - 1: 1.0}
    return target_fidelities


def get_recommendation(model, bounds, test_prob):
    '''
    Obtain the maximizer and maximum value of the GP posterior.

    Args:
    model: The Gaussian Process model.
    bounds: Bounds for optimization.
    test_prob (dict): Dictionary containing information about the test problem.

    Returns:
    torch.Tensor, torch.Tensor: Maximizing point and its objective value.
    '''

    # For curr_val_acqf
    dim = test_prob['fx'].dim
    columns = [test_prob['fx'].dim - 1]

    # Create FixedFeatureAcquisitionFunction
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dim,
        columns=columns,
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )

    final_rec = rec_acqf._construct_X_full(final_rec)
    objective_value = test_prob['fx'](final_rec)
    
    return final_rec, objective_value


def eval_highest(test_prob=None, x=None):
    '''
    Evaluate the objective at the highest fidelity for the point sampled during the current iteration.

    Args:
    test_prob (dict): Dictionary containing information about the test problem.
    x (torch.Tensor): Sampled point.

    Returns:
    torch.Tensor: Objective value.
    '''
    temp = x.detach().clone()
    temp[:, -1] = 1
    return test_prob['fx'](temp).unsqueeze(-1)




def plot_signal(time, signal, title='', xlab='', ylab='',
                line_width=1, alpha=1, color='k',
                subplots=False, show_grid=True, fig=None):
    """
    Plot a signal over time.

    Args:
    time (array-like): Time values for the signal.
    signal (array-like): Signal values corresponding to each time point.
    title (str): Title for the plot.
    xlab (str): Label for the x-axis.
    ylab (str): Label for the y-axis.
    line_width (int): Width of the plotted line.
    alpha (float): Alpha value for transparency.
    color (str): Color of the plotted line.
    subplots (bool): Whether to create subplots.
    show_grid (bool): Whether to show grid lines on the plot.
    fig (matplotlib.figure.Figure): Figure object to plot on.

    Returns:
    matplotlib.figure.Figure: The figure object.
    """

    if subplots and fig is None:
        fig = plt.figure()

    if subplots:
        axarr = fig.add_subplot(1, 1, 1)  # Adding subplot to the figure
    else:
        axarr = plt.gca()

    axarr.plot(time, signal, linewidth=line_width,
               alpha=alpha, color=color)
    axarr.set_xlim(min(time), max(time))
    axarr.set_xlabel(xlab)
    axarr.set_ylabel(ylab)
    axarr.grid(show_grid)
    axarr.set_title(title, size=16)

    return fig


'''
def project(X,test_prob):
    target_fidelities = get_target_fidelities(test_prob)
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
'''