import pandas as pd
import numpy as np


# radius=-0.15 col_name=x1
def dropElement(X, col_name,  radius):
    for index, row in X.iterrows():
        if (row[col_name] < radius and row[col_name] > (-1)*radius):
            X.drop(index, inplace=True)


def getRandomX1X2():
    rng = np.random.RandomState(0)
    x1 = rng.randn(2100)
    x2 = rng.randn(2100)
    return x1, x2

def generate_vertical_no_lowdense():
    x1, x2 = getRandomX1X2()
    col_name_1, col_name_2 = 'x1', 'x2'

    X = pd.DataFrame({col_name_1: x1, col_name_1: x2})
    dropElement(X, col_name_1, 0.15)

    y = np.logical_xor(X[col_name_1] > 0, X[col_name_2] > 0)

    return X, y


def generate_horizontal_no_lowdense():
    x1, x2 = getRandomX1X2()
    col_name_1, col_name_2 = 'x1', 'x2'

    X = pd.DataFrame({col_name_1: x1, col_name_1: x2})
    dropElement(X, col_name_2, 0.25)

    y = np.logical_xor(X[col_name_1] > 0, X[col_name_2] > 0)

    return X, y

def generatelowdense():
    x1, x2 = getRandomX1X2()
    col_name_1, col_name_2 = 'x1', 'x2'

    X = pd.DataFrame({col_name_1: x1, col_name_1: x2})
    dropElement(X, col_name_1, 0.1)

    y = np.logical_or(np.logical_and(X[col_name_1] > 0, X[col_name_2] > 0), np.logical_and(X[col_name_1] > 0, X[col_name_2] < 0))

    return X, y


