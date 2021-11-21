import numpy as np
from code_py.ex3 import Two_Blocks_Optimization
from code_py.get_dataset import load_split_dataset



def ICanGeneralize(N=40,rho=1e-5,sigma=1.05,new_matrix=None):

    """
    :params new_matrix: Matrix of shape (N,2) to test the MLP network according to N=40,rho=1e-5,sigma=1
    :output -> predictions matrix of shape (1,N) 
    """

    X_train,Y_train,X_test,Y_test=load_split_dataset(name="data/DATA.csv",fraction=0.744,seed=1942297)
    np.random.seed(7)
    two_blocks_opt=Two_Blocks_Optimization(X_train, Y_train, X_test, Y_test, rho, sigma, N)
    _,summary_dict = two_blocks_opt.Early_Stopping(two_blocks_opt, min_delta=1e-4, patiente=10, max_iter=5000)
    omega_ = summary_dict["omega"]
    v = omega_[:two_blocks_opt.N].reshape(1, two_blocks_opt.N)
    b = omega_[two_blocks_opt.N:2 * two_blocks_opt.N].reshape(1, two_blocks_opt.N)
    w = omega_[2 * two_blocks_opt.N:].reshape(new_matrix.shape[1], two_blocks_opt.N)
    predictions = np.dot(two_blocks_opt.tanh(np.dot(new_matrix, w) + b), v.T)
    return predictions

"""
z is a just a toy example to run the code, 
so please substitute z with your own matrix and run the function ICanGeneralize
to have the prediction array od shape (N,2)
"""
z = np.random.rand(113,2)
y_hat = ICanGeneralize(new_matrix=z)
print(y_hat)
