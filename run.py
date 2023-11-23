import argparse
from distutils.util import strtobool
import datetime
import time
import torch
import sys

from botorch.test_functions.multi_fidelity import AugmentedHartmann
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

#import custom test function 
from currin import *
from forrestor import *
from branin import *
from cost_model import *
from utilis import *
from BO import *
from config import *


if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Multifidelity Experiment")

    # Argument for saving results
    parser.add_argument("--save_result", type=lambda x: bool(strtobool(x)),
                        nargs='?', const=True, default=False,
                        help='If True, Save the data for the experiment')

    # Argument for choosing problem
    parser.add_argument("--choose_prob", type=int, default=0,
                        help='if Choose_Prob == 0, pick Hartmann, Choose_Prob==1 pick branin, Choose_Prob==2 pick Currin, else, Forrestor will be picked')

    # Argument for selecting number of points to pick in one iteration
    parser.add_argument("--q", type=int, default=1,
                        help='Number of points to pick in one iteration')

    # Argument for number of Bayesian optimization iterations
    parser.add_argument("--n_iter", type=int, default=10,
                        help='Number of Bayesian optimization iterations')

    # Argument for choosing Acquisition Function
    parser.add_argument("--choose_af", type=int, default=0,
                        help='if value passed == 0, qMultiFidelityKnowledgeGradient will be picked, else, qMultiFidelityMaxValueEntropy will be picked')

    # Argument for loading previous experiment
    parser.add_argument("--load_previous", type=lambda x: bool(strtobool(x)),
                        nargs='?', const=True, default=False,
                        help='If True, Restart the last experiment where from where it was stopped')

    # Argument for path if loading previous experiment
    parser.add_argument("--path", type=str, default=None,
                        help="If Load_Previous == True, pass the path of parent dir e.g.: HOME/AugmentedHartmann/qMFKG/06_22_13_13, also pick the right test prob and AF")

    # Argument for special experiment remarks
    parser.add_argument("--notes", type=str, default="No special Remarks",
                        help="Special Remarks for experiment")

    # Argument for seed
    parser.add_argument("--seed", type=int, default=0,
                        help='Seed for torch.')

    # Argument for cost options
    parser.add_argument("--cost_option", type=int, default=0,
                        help='if 0-->cost=5+s, if 1 --> cost = 5 + exp(4s), if 2-->cost = 5 + exp(8s)')

    # Argument for running for a single fidelity
    parser.add_argument("--single_fidelity", type=lambda x: bool(strtobool(x)),
                        nargs='?', const=True, default=False,
                        help='If True, Run for a Fidelity=1 only.')

    argspar = parser.parse_args()


    # Printing parsed arguments
    for p, v in vars(argspar).items():
        print("\t{}: {}".format(p, v), flush=True)
    print('\n\n')

    # Assigning parsed arguments
    choose_prob = argspar.choose_prob
    choose_af = argspar.choose_af
    save_result = argspar.save_result
    n_iter = argspar.n_iter
    q = argspar.q
    load_previous = argspar.load_previous
    previous_path = argspar.path
    notes = argspar.notes
    seed = argspar.seed
    cost_option = argspar.cost_option
    single_fidelity = argspar.single_fidelity
    
    #********************************* Define dict for test problem, acquisation function, and cost function**************************

    TEST_PROBLEMS = {
        "Hartmann": {'fx': AugmentedHartmann(negate=True).to(**tkwargs),
                    'Fidelity_dim': 1},
        "Branin": {'fx': AugmentedBranin_modified().to(**tkwargs),
                'Fidelity_dim': 1},
        "Currin": {'fx': AugmentedCurrin().to(**tkwargs),
                'Fidelity_dim': 1},
        "Forrestor": {'fx': AugmentedForrestor().to(**tkwargs),
                    'Fidelity_dim': 1},
    }

    AQF = {
        "qMFKG": qMultiFidelityKnowledgeGradient,
        "qMFMES": qMultiFidelityMaxValueEntropy,
    }

    COST = {
        0: "Linear",
        1: "exp(4s)",
        2: "exp(8s)",
    }

    
    #********************************* Select problem and acquisation function based on cmd arguments ******************************
    #select test problem
    global test_prob
    if choose_prob == 0:
        test_prob = TEST_PROBLEMS['Hartmann']
    elif choose_prob == 1:
        test_prob = TEST_PROBLEMS['Branin']
    elif choose_prob == 2:
        test_prob = TEST_PROBLEMS['Currin']
    else:
        test_prob = TEST_PROBLEMS['Forrestor']
                    
            
    #select acquisation function        
    global test_af    
    if choose_af == 0:
        test_af = AQF['qMFKG']
    else:
        test_af = AQF['qMFMES']
        
    
    #Select cost model
    if cost_option not in {0,1,2}:
        print("\npassed valid argument for cost option\n",flush= True)
        sys.exit()
    target_fidelities = get_target_fidelities(test_prob)
    

    if cost_option == 0:
        cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=5.0)
    elif cost_option == 1:
        cost_model = AffineFidelityCostModel_exp(fidelity_weights=target_fidelities, fixed_cost=5.0, exp_coff = 4)
    else:
        cost_model = AffineFidelityCostModel_exp(fidelity_weights=target_fidelities, fixed_cost=5.0, exp_coff = 8)
    
    
    #Based on Selected test problem, acquisation Function, and cost function, create the partial path to where the data will stored.
    aqf_selected = str(test_af).rpartition('.')[-1][:-2]
    base = os.getcwd()

   
    if aqf_selected == 'qMultiFidelityMaxValueEntropy':
        if single_fidelity == False:
            parent_dic = os.path.join(base, str(test_prob['fx'])[:-2], "qMFMES", COST[cost_option])
        else:
            parent_dic = os.path.join(base, str(test_prob['fx'])[:-2], "qMFMES", COST[cost_option], "Single_Fidelity")
    else:
        if single_fidelity ==False:
            parent_dic = os.path.join(base ,str(test_prob['fx'])[:-2], "qMFKG", COST[cost_option])
        else:
            parent_dic = os.path.join(base, str(test_prob['fx'])[:-2], "qMFKG", COST[cost_option], "Single_Fidelity")
    
    
    #***********************************utilis used for BO *******************************************************
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    
    
    #***********************************make dict and text file if save_result is True ***************************    
    path = None # A hacky way to make thing work if you don't want to save result.
    # if not Load_previous create files and dict for a fresh experiment
    if not load_previous:
        print(f"Parent Dic: {parent_dic}\n", flush=True)
        print(f"The Choosen AQF is: {aqf_selected}\n"
            f"The Choosen Test Problem is: {test_prob['fx']}\n"
            f"The Choosen Cost function is: {COST[cost_option]}\n", flush=True)


        #if parent dic doesn't exist, create one.
        if not os.path.exists(parent_dic):
            os.makedirs(parent_dic)
                
        if save_result:
            now  = datetime.datetime.now()
            file = now.strftime("%m_%d_%H_%M")
            #final path where data will be stored (by appending data and time when exiperment is ran) 
            path = os.path.join(parent_dic, file)
            #if previous experiment is finished within a minute, and a new script is launched, it will try to create the directory
            #which already exists. So wait a minute if directory already exist.
            if os.path.exists(path):
                import time
                time.sleep(65)
                now  = datetime.datetime.now()
                file = now.strftime("%m_%d_%H_%M")
                path = os.path.join(parent_dic, file)
            os.makedirs(path)

            
            #write the readme file.
            readme_content = f"""
            The routine is running for {aqf_selected} on the {str(test_prob['fx'])[:-2]} test problem.
            The routine runs for {n_iter} iteration(s), and at each iteration, {q} point(s) are selected.

            File 'train_x_initial.txt' comprises the initial sample points. These points serve as the starting inputs for the beginning of the Bayesian Optimization (BO) process
            File 'train_obj_initial.txt' contains the objective value evaluated at initially sampled points.
            File 'metric_1.txt' records the objective function values evaluated at highest fidelity for points selected during the (BO) iterations.
            File 'metric_2.txt' contains the maximizer and maximum value of the GP posterior.
            Folder 'Model' contains the GP model and state dict after {n_iter} iteration(s).
            File 'train_X.txt' contains the sample points obtained after {n_iter} iteration(s) during the Bayesian Optimization (BO) process.
            File 'train_obj.txt' contains the final sample point after {n_iter} iteration(s). These are objective value evaluated at points that are sampled during BO.
            File 'cost.txt' contains the value of the cost at each iteration.
            File 'final_rec' contains the location of the global optimizer, optimized objective value, and total cumulative cost.

            {notes}

            Cost_MODEL = {COST[cost_option]}
            """

            readme_path = os.path.join(path, "readme.txt")

            with open(readme_path, "a") as file:
                file.write(readme_content)

    # Using the existing path passed via cmd if Load_Previous is True.
    if load_previous:
        path = previous_path

    #Run the BO routine.
    start = time.time()
    torch.manual_seed(seed)
    run_BO(path=path,
           n_iter=n_iter,
           save_result=save_result,
           aqf_selected=aqf_selected,
           load_previous=load_previous,
           test_prob=test_prob,
           cost_model=cost_model,
           cost_aware_utility=cost_aware_utility,
           q=q,
           single_fidelity=single_fidelity)
    
    print(f"\nTime : {str(time.time()-start)}\n")