import pandas as pd
import numpy as np


def load_split_dataset(name,fraction,seed):
    data = pd.read_csv(name)
    train= data.sample(frac = fraction, random_state=seed)
    test = data.drop(train.index)
    X_train, Y_train = train[['x1', 'x2']].to_numpy(), train[['y']].to_numpy()
    X_test, Y_test = test[['x1', 'x2']].to_numpy(), test[['y']].to_numpy()
    return X_train,Y_train,X_test, Y_test


def load_full_dataset(name):
    data = pd.read_csv(name)
    X_train, Y_train = data[['x1', 'x2']].to_numpy(), data[['y']].to_numpy()
    return X_train,Y_train