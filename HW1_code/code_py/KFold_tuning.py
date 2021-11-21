import itertools
from code_py.ex11 import MLP_Network
from code_py.ex12 import RBF_Network
from sklearn.model_selection import train_test_split
import itertools
import numpy as np
import json
from joblib import Parallel, delayed
import tqdm
import pandas as pd
from sklearn.model_selection import KFold



def cartesian_combination(dict):
    reg = dict["regularization"]
    N_el = dict["N_el"]
    sigma = dict["sigma"]

    combination = [reg, N_el, sigma]

    cartesian_stack = []
    for element in itertools.product(*combination):
        cartesian_stack.append(element)
    return cartesian_stack




def tuning_MLP_Network(list,X,Y,num_fold=5,n_jobs=-1):
    '''
    Tune the hyper-parameters by applying 5-folds cross-validation,
    Run CV 10 times to produce different samples in the folds

    :param list: a list that contains rho, N, and sigma hyper-parameters' sample space
    :param X: The features of the dataset points
    :param Y: The label of the dataset points
    :param num_fold: number of folds to be used in k-fold cross validation
    :param n_jobs: The number of jobs to run in parallel
    :return: flatten list that contains average validation loss and success flag from optimisation for every combination
    of grid search
    '''
    flatten_list = []
    KFold_val_loss = []

    def grid_search(i):
        reg_element = list[i][0]
        N_element = list[i][1]
        sigma_element = list[i][2]
        for i in range(10):
            kf = KFold(n_splits=num_fold, shuffle=True)
            for train_index, val_index in kf.split(X):
                X_train, X_test = X[train_index], X[val_index]
                y_train, y_test = Y[train_index], Y[val_index]
                init_MLP = MLP_Network(x=X_train, y=y_train,x_test=X_test,y_test=y_test, rho=reg_element, sigma=sigma_element, N=N_element)
                MLP_opt = init_MLP.optim(verbose=False)
                opt_pass, _, _, _, _, _, val_loss, _, opt_time = MLP_opt
                KFold_val_loss.append(val_loss)
        KFold_val_loss_value = sum(KFold_val_loss)/len(KFold_val_loss)
        
        return (KFold_val_loss_value,opt_pass.success)

    flatten_list = Parallel(n_jobs=n_jobs)(delayed(grid_search)(i) for i in range(len(list)))
    return flatten_list



def tuning_RBF_Network(list,X,Y,num_fold=5,n_jobs=-1):
    '''
    Tune the hyper-parameters by applying 5-folds cross-validation,
    Run CV 10 times to produce different samples in the folds

    :param list: a list that contains rho, N, and sigma hyper-parameters' sample space
    :param X: The features of the dataset points
    :param Y: The label of the dataset points
    :param num_fold: number of folds to be used in k-fold cross validation
    :param n_jobs: The number of jobs to run in parallel
    :return: flatten list that contains average validation loss and success flag from optimisation for every combination
    of grid search
    '''
    flatten_list = []
    KFold_val_loss = []

    def grid_search(i):
        reg_element = list[i][0]
        N_element = list[i][1]
        sigma_element = list[i][2]
        for i in range(10):
            kf = KFold(n_splits=num_fold, shuffle=True)
            for train_index, val_index in kf.split(X):
                X_train, X_test = X[train_index], X[val_index]
                y_train, y_test = Y[train_index], Y[val_index]
                init_RBF = RBF_Network(x=X_train, y=y_train,x_test=X_test,y_test=y_test, rho=reg_element, sigma=sigma_element, N=N_element)
                RBF_opt = init_RBF.optim(verbose=False)
                opt_pass, _, _, _, _, val_loss, train_loss, opt_time = RBF_opt
                KFold_val_loss.append(val_loss)

        KFold_val_loss_value = sum(KFold_val_loss)/len(KFold_val_loss)

        return (KFold_val_loss_value,opt_pass.success)

    flatten_list = Parallel(n_jobs=n_jobs)(delayed(grid_search)(i) for i in range(len(list)))
    return flatten_list

def save_dict(name,dict):
    with open(str(name)+".txt",'w+') as f:
        f.write(str(dict))

def read_dict(name):
    from numpy import nan
    dic = ''
    with open(str(name)+'.txt','r') as f:
             for i in f.readlines():
                dic=i #string
    dic = eval(dic)
    return dic

def loss_eval(X_train,Y_train, X_test, Y_test, sigma, rho, N, method):
    '''
    Compute the loss value on train and test datasets with given hyper-parameters and method(MLP/RBF)

    :param X_train: Data features of training set
    :param Y_train: Labels of training set
    :param X_test: Data features of test set
    :param Y_test: Labels of test set
    :param sigma: hyper-parameter
    :param rho: hyper-parameter
    :param N: number of neurons, hyper-parameter
    :param method: MLP(MultiLayerPerceptron)/ RBF(RadialBasisFunction)

    :return: test error, train error, selected method, message received after optimisation(successful or not),
    number of function evaluations, number of gradient evaluations, number of iterations performed, optimization time
    '''
    if method == 'MLP':
        
        MLP_net = MLP_Network(x=X_train,y=Y_train,x_test=X_test,y_test=Y_test,rho=rho,sigma=sigma,N=N)
        opt_pass, opt_v, opt_b, opt_w, opt_pass.x, opt_pass.fun, test_loss, train_loss, opt_time = MLP_net.optim()
        return test_loss, train_loss, method, opt_pass.message, opt_pass.x, opt_pass.nfev, opt_pass.njev, opt_pass.nit, opt_time
    
    elif method == 'RBF':
        
        RBF_net = RBF_Network(x=X_train,y=Y_train,x_test=X_test,y_test=Y_test,rho=rho,sigma=sigma,N=N)
        opt_pass, opt_v, opt_c, opt_pass.x, opt_pass.fun, test_loss, train_loss, opt_time = RBF_net.optim()
        
        return test_loss, train_loss, method, opt_pass.message, opt_pass.x, opt_pass.nfev, opt_pass.njev, opt_pass.nit, opt_time
