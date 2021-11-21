from code_py.get_dataset import *
from code_py.ex11 import MLP_Network
import pandas as pd
import operator
from code_py.get_dataset import load_split_dataset


def ICanGeneralize(X_test , rho = 1e-05, sigma = 1.05, N = 40):
    '''
    Predict the labels of the dataset based on the optimal values found for the hyper-parameters rho, sigma and N
    in MLP Network

    :param X_test: The dataset consists of the data features only
    :param rho: hyper-parameter, optimal value is given as default
    :param sigma: hyper-parameter, optimal value is given as default
    :param N: hyper-parameter, optimal value is given as default
    :return: prediction for the output value vector -> len(X_test) x 1
    '''
    x_train, y_train, x_test, y_test = load_split_dataset(name="data/DATA.csv", fraction=0.744, seed=1942297)
    MLP_net = MLP_Network(x=x_train, y=y_train, x_test=x_test, y_test=y_test, rho=rho, sigma=sigma, N=N)
    _, _, _, _, omega, _, _, _, _ = MLP_net.optim()
    y_predict = MLP_net.forward_pass(X_test, omega)
    return y_predict

"""
z is a just a toy example to run the code, 
so please substitute z with your own matrix and run the function ICanGeneralize
to have the prediction array od shape (N,2)
"""
z = np.random.rand(113,2)
y_hat = ICanGeneralize(X_test=z)
print(y_hat)