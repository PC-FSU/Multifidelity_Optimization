import numpy as np
import sys
import os
import argparse
import torch
from distutils.util import strtobool
import datetime

from botorch.test_functions.multi_fidelity import AugmentedHartmann,AugmentedBranin,AugmentedRosenbrock
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
 
from botorch.utils.sampling import draw_sobol_samples

from botorch import fit_gpytorch_model
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim import optimize_acqf_cyclic
from botorch.utils.transforms import unnormalize,standardize

from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
torch.set_printoptions(precision=3, sci_mode=False)

from currin import *
from forrestor import *

tkwargs = {
    "dtype":  torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 32 if not SMOKE_TEST else 4
torch.manual_seed(0)

def generate_initial_data(n=16):
    dim = Test_Prob['fx'].dim
    
    bounds = Test_Prob['fx']._bounds
    bounds    = torch.tensor(bounds).T
    # generate training data
    
    train_x   = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, dim, **tkwargs)
    train_obj = Test_Prob['fx'](train_x).unsqueeze(-1) # add output dimension

    print(bounds)
    
    return train_x, train_obj, bounds


def initialize_model(train_x, train_obj):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension 6, as in [2]
    data_fidelity = Test_Prob['fx'].dim - 1
    
    model = SingleTaskMultiFidelityGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=1),  
        #An outcome transform that is applied to the training data during instantiation and to
        #the posterior during inference (that is, the Posterior obtained by calling .posterior 
        #                                on the model will be on the original scale).
        data_fidelity=data_fidelity  #The column index for the downsampling fidelity parameter (optional)
    )       
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def get_target_fidelities():
    target_fidelities = {Test_Prob['fx'].dim - 1 : 1.0}
    return target_fidelities


'''
def project(X):
    target_fidelities = get_target_fidelities()
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
'''


# Define the Acquisition function for qMultiFidelityKnowlegeGradient
def get_mfkg(model,bounds,cost_aware_utility):
    
    #for curr_val_acqf
    dim     = Test_Prob['fx'].dim
    columns = [Test_Prob['fx'].dim - 1]
        
    target_fidelities = get_target_fidelities()
    project = lambda X: project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
    
    #The first two function calculate the current optimzation value and feed it to the optimize routine
    #defined below.
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dim,
        columns=columns,
        values=[1],  #Assign a value 1 to the columns value passed above 
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        #bounds = bounds,
        q=1,
        num_restarts=NUM_RESTARTS if not SMOKE_TEST else 2,
        raw_samples =RAW_SAMPLES if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
    
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=8 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_mfkg_and_get_observation(mfkg_acqf,bounds,q,cost_model):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""
    
    #generate initial condition
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function = mfkg_acqf,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    #use the initial condition defined above and perform optimization
    candidates, _ = optimize_acqf(
        acq_function=mfkg_acqf,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = Test_Prob['fx'](new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost



# Define the Acquisition function for qMultiFidelityMaxValueEntropy
def get_MfMes(model,bounds,cost_aware_utility):
    
    target_fidelities = get_target_fidelities()
    project = lambda X: project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
    
    candidate_set = torch.rand(1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype)
    candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
    
    qMES = qMultiFidelityMaxValueEntropy(
        model = model, 
        candidate_set = candidate_set,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )
    return qMES


def optimize_Mfmes_and_get_observation(Mfmes_acqf,bounds,q,cost_model):
    """Optimizes MFMES and returns a new candidate, observation, and cost."""

    candidates_q2_cyclic, acq_value_q2_cyclic = optimize_acqf_cyclic(
        acq_function=Mfmes_acqf, 
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        cyclic_options={"maxiter": 20},
    )
    
    # observe new values
    cost = cost_model(candidates_q2_cyclic).sum()
    new_x = candidates_q2_cyclic.detach()
    new_obj = Test_Prob['fx'](new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost


def get_recommendation(model,bounds):
    
    #for curr_val_acqf
    dim = Test_Prob['fx'].dim
    columns = [Test_Prob['fx'].dim - 1]
    
    #
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dim,
        columns=columns,
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
    )
    
    final_rec = rec_acqf._construct_X_full(final_rec) 
    #what's construct_x_full
    
    objective_value = Test_Prob['fx'](final_rec)
    print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec,objective_value


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Multifidelity Experiment")
    
    parser.add_argument("--save_result", type=lambda x:bool(strtobool(x)),
    nargs='?', const=True, default=False, help='If True, Save the data for the experiment')
    
    parser.add_argument("--Choose_Prob",type=int, default=0,
                   help='if value passed == 0, Hartmann will be picked, else, Branin will be picked')
    
    parser.add_argument("--q",type=int, default=1,
                   help='Number of point to pick in one iteration')
    
    parser.add_argument("--N_ITER",type=int, default=10,
                   help='Number of Bayesian optimization iteration')  
    
    parser.add_argument("--Choose_AF",type=int, default=0,
                   help='if value passed == 0, qMultiFidelityKnowledgeGradient will be picked, else, qMultiFidelityMaxValueEntropy will be picked')
    
    parser.add_argument("--ReRun",type=int, default=1,
                   help='Random Initialization of a single experiment')
    
    
    argspar = parser.parse_args()

    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n\n')
    
    Choose_Prob      = argspar.Choose_Prob
    Choose_AF        = argspar.Choose_AF
    save_result      = argspar.save_result
    N_ITER           = argspar.N_ITER
    q                = argspar.q
    ReRun            = argspar.ReRun
    
    
    problem = {"Hartmann"   : {'fx':AugmentedHartmann(negate=True).to(**tkwargs),
                               'Fidelity_dim':1},
               "Branin"     : {'fx':AugmentedBranin(negate=True).to(**tkwargs),
                              'Fidelity_dim':1},
               "Rose"       : {'fx':AugmentedRosenbrock(negate=True,dim=5).to(**tkwargs),
                              'Fidelity_dim':2},
               "Currin"     : {'fx':AugmentedCurrin().to(**tkwargs),
                              'Fidelity_dim':1},
               "Forrestor"  : {'fx':AugmentedForrestor().to(**tkwargs),
                              'Fidelity_dim':1},
              }
    
    
    
    AQF = {"qMFKG"  :  qMultiFidelityKnowledgeGradient,
           "qMFMES" :  qMultiFidelityMaxValueEntropy,
          }
    
    
    global Test_Prob
    global Test_AF
    
    if Choose_Prob == 0:
        Test_Prob = problem['Hartmann']
    else:
        Test_Prob = problem['Branin']
    
    Test_Prob = problem['Forrestor']
    
        
    if Choose_AF == 0:
        Test_AF = AQF['qMFKG']
    else:
        Test_AF = AQF['qMFMES']
        
    
    
    AQF_selected = str(Test_AF).rpartition('.')[-1][:-2]
    if AQF_selected == 'qMultiFidelityMaxValueEntropy':
        Parent_dic = os.path.join(str(Test_Prob['fx'])[:-2],"qMFMES")
    else:
        Parent_dic = os.path.join(str(Test_Prob['fx'])[:-2],"qMFKG")

    print("The Choosen AQF is : %s \nThe Choosen Test Problem is : %s\n "%(AQF_selected,Test_Prob['fx']))
    if not os.path.exists(Parent_dic):
        os.makedirs(Parent_dic)
        
        
    target_fidelities  = get_target_fidelities()
    cost_model         = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=5.0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    
    if save_result:
        now  = datetime.datetime.now()
        file = now.strftime("%m_%d_%H_%M")
        path = os.path.join(Parent_dic,file)
        os.makedirs(path)
        #os.makedirs(os.path.join(path,"model"))
        

     
    if save_result:
        file = open(os.path.join(path,"readme.txt"),"a")

        file.write("\nThe routine is running for %s, and on %s test problem\n"%(AQF_selected,str(Test_Prob['fx'])[:-2]))

        file.write("\nThe Routine is running for %d iteration, and at each iteration %d points are selected \n" %(N_ITER,q))     

        file.write("\nFile train_x_inital.txt contain the intial input sample point\n")
        #np.savetxt(os.path.join(path,"train_x_inital.txt"),train_x)

        file.write("\nFile train_obj_inital.txt contain the objective value of initial inpit sample points\n")
        #np.savetxt(os.path.join(path,"train_obj_inital.txt"),train_obj)

        file.write("\nFolder Model contain the GP model and state dict after %d iterations\n"%(N_ITER))
        file.write("\nFile train_X.txt contain the final sample point after %d iterations\n"%(N_ITER))
        file.write("\nFile train_obj.txt contain the final sample point after %d iterations\n"%(N_ITER))
        file.write("\nFile cost.txt contain the value of cost at each iterations\n")
        file.write("\nFile cost.txt contain the value of cost at each iterations\n")
        file.write("\nFile final_rec contain the location of gloabl optimizer, optimize objective value and total cumulative cost\n")
        file.close()


    
    for run in range(ReRun):
        
        train_x, train_obj, bounds = generate_initial_data(n=16)
        
        if save_result:
            fPath = os.path.join(path,str(run))
            os.makedirs(fPath)
            os.makedirs(os.path.join(fPath,"model"))
            np.savetxt(os.path.join(fPath,"train_x_inital.txt"),train_x)
            np.savetxt(os.path.join(fPath,"train_obj_inital.txt"),train_obj)
                                 
        cumulative_cost = 0.0

        for i in range(N_ITER):

            print("Routine Running for Run, Iteration : %d, %d\n"%(run,i))
                                 
            mll, model = initialize_model(train_x, standardize(train_obj))
            fit_gpytorch_model(mll)

            if AQF_selected == 'qMultiFidelityMaxValueEntropy':
                mfmes_acqf = get_MfMes(model,bounds,cost_aware_utility)
                new_x, new_obj, cost = optimize_Mfmes_and_get_observation(mfmes_acqf,bounds,q,cost_model)
            else:
                mfkg_acqf = get_mfkg(model, bounds,cost_aware_utility)
                new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf,bounds,q,cost_model)

            train_x   = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            cumulative_cost += cost

            if save_result:
                #save new candidate
                with open(os.path.join(fPath,"train_X.txt"), "ab") as f:
                    np.savetxt(f, new_x)
                # save new objective
                with open(os.path.join(fPath,"train_obj.txt"), "ab") as f:
                    np.savetxt(f, new_obj)   
                # save new cost    
                with open(os.path.join(fPath,"cost.txt"), "ab") as f:
                    np.savetxt(f, [cost])  

                if i % 5 == 0:
                    print("Saving the model at iteration: %d\n"%i)
                    torch.save(model.state_dict(), os.path.join(fPath,"model",'model_state.pth'))

        final_rec,final_obj = get_recommendation(model,bounds)
        print(f"\ntotal cost: {cumulative_cost}\n")

        if save_result:
            with open(os.path.join(fPath,"final_rec.txt"), "ab") as f:
                np.savetxt(f, final_rec.numpy())
                np.savetxt(f, [cumulative_cost])
                np.savetxt(f, [final_obj.numpy()])