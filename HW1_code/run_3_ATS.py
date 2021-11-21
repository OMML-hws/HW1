from code_py.ex3 import Two_Blocks_Optimization
from code_py.get_dataset import load_split_dataset
import numpy as np

"""
The function load_split_dataset return the dataset splitted 
into train in test for both x and y variables

:param name: address path + file name
:params fraction: the fraction by which the dataset is divided
:params seed: seed according to the matricula number
:return output -> splitted dataset  
"""
X_train,Y_train,X_test,Y_test=load_split_dataset(name="data/DATA.csv",fraction=0.744,seed=1942297)

"""
setting the parameter that we selcted from the ex11 according to 10-KFold
"""
np.random.seed(7)

N  =40
rho=1e-5
sigma=1.05

"""
Defining two_blocks_opt class according to the splitted dataset and the selected values of rho,sigma,N.
"""
two_blocks_opt=Two_Blocks_Optimization(X_train, Y_train, X_test, Y_test, rho, sigma, N)

"""
Calling the Early stopping method
:params min_delta: threshold below which Early Stoppign is activated, monitoring validation loss
:params patiente: number of epochs to wait during min_delta condition is verified
:params max_iter: maximum number of iterations to wait before forcing 
                  Early Stopping==True (rarely activate, only in case of non-convergence ).
:return output -> class
"""
_,summary_dict = two_blocks_opt.Early_Stopping(two_blocks_opt, min_delta=1e-4, patiente=10, max_iter=500)

"""
Summary dictionary containing statistics of the optimisation process

:params omega: flatten vector with v,b,w concatenated 
:params n_fun_eval: n of function evaluations 
:params n_grad_eval: n of gradient evaluations
:params execution_time: elapsed time from the calling of the solver (seconds)
:params val_loss: Validations Loss
:params train_loss: Number of train
:params outer_success: Numbers of outer success of the non convex optimization block
:params n_iterations: Number of iterations of the non-convex optimization block
:params n_fun_eval_cvx: Number of iterations of the convex block optimization

:return output -> dictionary
"""

print("====================================================================================")
print("Number of neurons N:", two_blocks_opt.N)
print("Value of Sigma:",two_blocks_opt.sigma)
print("Value of Rho",two_blocks_opt.rho)
print("Optimization Solvers used: CG for block 1 (non convex) -- ECOS for block (convex)")
print("Number of function evaluations: ",summary_dict["n_fun_eval"]+summary_dict["n_fun_eval_cvx"])
print("Number of Gradient evaluations: ",summary_dict["n_grad_eval"])
print("Number of successes: ",summary_dict["outer_success"])
print("Time to optimise the function: "+ str(round(summary_dict["execution_time"],2))+"s")
print("Train Error: ", summary_dict["train_loss"][-1])
print("Test Error:  ", summary_dict["val_loss"][-1])
print("====================================================================================")

"""  
3D Plot of the example function
"""
#two_blocks_opt.Plot3D(summary_dict["omega"])

"""
Uncomment if you want the plot of the train, validation loss per epochs.
"""
#two_blocks_opt.train_loss_Plot(summary_dict["val_loss"],summary_dict["train_loss"])
