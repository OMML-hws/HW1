from cvxpy.expressions.cvxtypes import problem
import numpy as np
from code_py.ex21 import MLP_Extreme_Opt
from code_py.get_dataset import load_split_dataset
import torch

# seed 
torch.manual_seed(7)

# load the data
X_train,Y_train,X_test,Y_test=load_split_dataset(name="data/DATA.csv",fraction=0.744,seed=1942297)

# parameters
N = 40 
rho =  1e-5
sigma = 1.05
simulations = 100

# use the "random sampling method" + train the model
ext_MLP_net = MLP_Extreme_Opt(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test,rho=rho,sigma=sigma,N=N) 
best_b, best_w = ext_MLP_net.Random_Sampling(simulations)
ext_MLP_net = MLP_Extreme_Opt(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test,rho=rho,sigma=sigma,N=N)
loss,v_ext_opt,problem   = ext_MLP_net.convex_training_error_opt(b=best_b,w=best_w,verbose=False)

# print of the run file
print("===========================================")
print("Number of neurons: ", ext_MLP_net.N)
print("Value of Sigma:",ext_MLP_net.sigma)
print("Value of Rho: ",ext_MLP_net.rho)
print("Number of simulation for \"Random_Sampling\": ", simulations)
print(f"Optimization Solvers used: {problem.solver_stats.solver_name}")
#print(f"Total number of iteration: {problem.solver_stats.num_iters}")
print(f"Time needed to optimise the function: {problem.solver_stats.solve_time}s")
print("Train Error: ",ext_MLP_net.train_loss(b=best_b,w=best_w,v=v_ext_opt))
print("Test Error: ", ext_MLP_net.validation_loss(b=best_b,w=best_w,v=v_ext_opt))
print("==========================================")