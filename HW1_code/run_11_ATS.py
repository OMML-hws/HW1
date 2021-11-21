from code_py.get_dataset import *
from code_py.ex11 import MLP_Network
import pandas as pd
from sklearn.model_selection import KFold
from code_py.KFold_tuning import *
import operator 
from code_py.get_dataset import load_split_dataset

# Data splitting for x_train, y_train, x_test and y_test by setting a seed for reproducibility
x_train,y_train,x_test,y_test=load_split_dataset(name="data/DATA.csv",fraction=0.744,seed=1942297)

# fine_tune is a flag for hyper-parameter tuning, if False the best hyper-parameters will be used then
# save is a flag to observe average validation accuracy, It can help to observe over/under-fitting behaviors
fine_tune = False
save = True

if fine_tune:

    hyp_params={
        "regularization": [1e-5, 1e-4, 1e-3],
        "N_el":           [5, 10, 20, 30, 40, 50, 60, 70, 80],
        "sigma":          [.95, 1, 1.05]}

    MLP_combination = cartesian_combination(hyp_params)

    if save:

        HyperParameter_MLP_dict = {}
        HyperParameter_MLP_list= tuning_MLP_Network(MLP_combination,x_train,y_train,num_fold=5,n_jobs=-1)
        HyperParameter_MLP_dict= HyperParameter_MLP_dict.fromkeys(MLP_combination, 0)
        HyperParameter_MLP_dict.update(zip(HyperParameter_MLP_dict, HyperParameter_MLP_list))

        save_dict("HyperParameter_MLP",HyperParameter_MLP_dict)

    else:

        HyperParameter_MLP_dict = read_dict("HyperParameter_MLP")


    HyperParameter_MLP_dict = sorted(HyperParameter_MLP_dict.items(), key=operator.itemgetter(1))

    rho   = HyperParameter_MLP_dict[0][0][0]
    N     = HyperParameter_MLP_dict[0][0][1]
    sigma = HyperParameter_MLP_dict[0][0][2]
else:
    rho   = 1e-5
    N     = 40
    sigma = 1.05

test_loss, train_loss, method, message, omega, nfev, njev, nit, opt_time = loss_eval(X_train = x_train, Y_train = y_train, X_test = x_test, Y_test=y_test, sigma = sigma, rho = rho, N = N, method = "MLP")
print("Number of neurons N chosen: ", N)
print("Value of σ chosen: ", sigma)
print("Value of ρ chosen: ", rho)
print("Value of additional hyper-parameter k (k-fold CV): ", str(5))
print("Optimization solver chosen: CG")
print("Number of function evaluations: ", nfev)
print("Number of gradient evaluations: ", njev)
print("Time for optimizing the network: ", round(opt_time, 3), "sec")
print("Training Error: ", "{:.3e}".format(train_loss))
print("Test Error: ", "{:.3e}".format(test_loss))

'''
print("This part can be used to answer some of questions:")
print("--------------------------------------------------")
MLP_net = MLP_Network(x=x_train,y=y_train,x_test=x_test,y_test=y_test,rho=rho,sigma=sigma,N=N)
initial_grad_norm, final_grad_norm = MLP_net.gradient_norm(omega)
initial_obj, final_obj = MLP_net.objective_func_value(omega)
initial_train_error = MLP_net.initial_train_error()
print("Message: " + str(message))
print("Number of iterations: " + str(nit))
print("Initial norm of the gradient at the starting: ", "{:.3e}".format(initial_grad_norm),
      "| final point: ", "{:.3e}".format(final_grad_norm))
print("Starting value of the objective function: ", "{:.3e}".format(initial_obj), "| Final value : ", "{:.3e}".format(final_obj))
print("Initial train error: ", "{:.3e}".format(initial_train_error))
MLP_net.Plot3D(omega)
'''
