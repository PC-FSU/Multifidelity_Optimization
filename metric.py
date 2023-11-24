from utilis import *
import matplotlib.pyplot as plt


def metric_1(parent_dir = None):
    
    #load cost
    Cost         = np.loadtxt(os.path.join(parent_dic,"0","cost.txt"))
    cumsum       = np.cumsum(a)
    #Load Multi-Fidelity obective value picked
    MF_points    = np.loadtxt(os.path.join(parent_dic,"0","train_obj.txt"))
    #Load MF objective value projected to s=1
    MF_projected = np.loadtxt(os.path.join(parent_dic,"0","metric_1.txt"))
    
   
    f, ax = plt.subplots(1, 2, figsize=(10, 5),constrained_layout=True)
    
    
    ax[0][0].scatter(cumsum,MF_points,linestyle='None',label="MF Objective Value")
    ax[0][0].scatter(cumsum,MF_projected,label = "Projected Objective Value")
    ax[0][0].axhline(y=3.298, color='r', linestyle='-',label="Global Maxima")
    #plt.ylim(0,-50)
    ax[0][0].legend()
    ax[0][0].xlabel("Cumulative Cost")
    ax[0][0].ylabel("Obj Value")

    
    
    MF_points_best_observed    = [max(MF_points[:i+1]) for i in range(len(MF_points))]
    MF_projected_best_observed = [max(MF_projected[:i+1]) for i in range(len(MF_projected))]
    ax[0][1].scatter(cumsum,MF_points_best_observed,linestyle='None',label="Experimentaly best observed")
    ax[0][1].scatter(cumsum,MF_projected_best_observed,label = "Projected best observed")
    ax[0][1].axhline(y=3.298, color='r', linestyle='-',label="Global Maxima")
    #plt.ylim(0,-50)
    ax[0][1].legend()
    ax[0][1].xlabel("Cumulative Cost")
    ax[0][1].ylabel("Best Observed Obj Value")
    
    
    plt.savefig("metric_1.jpg")
    return None
    

def metric_2(parent_dir = None):
    
    #load cost
    Cost         = np.loadtxt(os.path.join(parent_dic,"0","cost.txt"))
    cumsum       = np.cumsum(a)
    #Load recommendation points at each iteration
    recommendate_objective  = np.loadtxt(os.path.join(parent_dic,"0","metric_2.txt"))[1::2]

    f, ax = plt.subplots(1, 2, figsize=(10, 5),constrained_layout=True)
    
    
    ax[0][0].plot(cumsum,recommendate_objective,label="Maximum Posterior")
    ax[0][0].axhline(y=3.298, color='r', linestyle='-',label="Global Maxima")
    ax[0][0].legend()
    ax[0][0].xlabel("Cumulative Cost")
    ax[0][0].ylabel("Obj Value")

    
    recommendate_objective_best_observed    = [max(recommendate_objective[:i+1]) for i in range(len(recommendate_objective))]
    ax[0][0].plot(cumsum,recommendate_objective_best_observed,label="Best Observed Posterior")
    ax[0][0].axhline(y=3.298, color='r', linestyle='-',label="Global Maxima")
    ax[0][1].legend()
    ax[0][1].xlabel("Cumulative Cost")
    ax[0][1].ylabel("Best Observed Posterior")
    
    
    plt.savefig("metric_2.jpg")
    return None


if __name__ == "__main__":
    
    #************************* parse cmd argument *****************************************
    parser = argparse.ArgumentParser(description="Run Multifidelity Experiment")
    
    parser.add_argument("--Path", type=str, default = None, help="Path of folder for which you want results")
    
    argspar = parser.parse_args()

    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print("\t{}: {}".format(p, v), flush=True)
    print('\n\n')

    Path = argspar.Path


    metric_1(parent_dir=Path)
    metric_2(parent_dir=Path)

    