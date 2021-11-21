import numpy as np
from code_py.ex22 import RBF_Extreme_Opt
from code_py.get_dataset import load_split_dataset

# seed 
np.random.seed(7)

# load the data
X_train,Y_train,X_test,Y_test=load_split_dataset(name="data/DATA.csv",fraction=0.744,seed=1942297)

# parameters
N = 60
rho =  1e-5
sigma = 1
simulations = 100
Kmeans = False

if Kmeans: # use the kmean + train
    ext_RBF_net = RBF_Extreme_Opt(x=X_train,y=Y_train,x_test=X_test,y_test=Y_test,rho=rho,sigma=sigma,N=N)
    best_c = ext_RBF_net.c
    loss,v_ext_opt,problem   = ext_RBF_net.convex_training_error_opt(c=best_c, verbose=False)
else: # use the "random sampling method" + train
    ext_RBF_net = RBF_Extreme_Opt(x=X_train,y=Y_train,x_test=X_test,y_test=Y_test,rho=rho,sigma=sigma,N=N,c_flag=True) 
    best_c = ext_RBF_net.Random_Sampling(simulations)
    loss,v_ext_opt,problem   = ext_RBF_net.convex_training_error_opt(c=best_c, verbose=False)


# print of the run file
print("===========================================")
print("Number of neurons: ", ext_RBF_net.N)
print("Value of Sigma:",ext_RBF_net.sigma)
print("Value of Rho: ",ext_RBF_net.rho)
if not Kmeans:
    print("Number of simulation for \"Random_Sampling\": ", simulations)
print(f"Optimization Solvers used: {problem.solver_stats.solver_name}")
#print(f"Total number of iteration: {problem.solver_stats.num_iters}")
print(f"Time needed to optimise the function: {problem.solver_stats.solve_time}s")
print("Train Error: ",ext_RBF_net.train_loss(v=v_ext_opt, c=best_c))
print("Test Error: ", ext_RBF_net.validation_loss(v=v_ext_opt, c=best_c))
print("==========================================")