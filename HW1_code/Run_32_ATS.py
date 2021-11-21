from code_py.get_dataset import *
from code_py.ex11 import MLP_Network
import pandas as pd
from sklearn.model_selection import KFold
from code_py.KFold_tuning import *
import operator
from code_py.get_dataset import load_split_dataset

# The complete dataset will be used to train the model
x_train, y_train = load_full_dataset(name="data/DATA.csv")

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

        save_dict("HyperParameter_MLP_bonus",HyperParameter_MLP_dict)

    else:

        HyperParameter_MLP_dict = read_dict("HyperParameter_MLP_bonus")

    HyperParameter_MLP_dict = sorted(HyperParameter_MLP_dict.items(), key=operator.itemgetter(1))
    rho   = HyperParameter_MLP_dict[0][0][0]
    N     = HyperParameter_MLP_dict[0][0][1]
    sigma = HyperParameter_MLP_dict[0][0][2]

else:
    rho   = 1e-5
    N     = 60
    sigma = 1

# Since we use the full dataset for training, we can compute the train loss in the following
_, train_loss, _, _, _, _, _, _, _ = loss_eval(X_train = x_train, Y_train = y_train, X_test = x_train, Y_test = y_train, sigma =sigma, rho = rho, N = N, method = "MLP")

print("Final train error: ", "{:.3e}".format(train_loss))
