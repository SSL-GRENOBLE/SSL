from ssl_random_forest import SemiSupervisedRandomForest

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np 
import random
import math
import time

def read_binary_pendigits(path_to_data):
    df = load_svmlight_file(path_to_data)
    features = df[0].todense().view(type=np.ndarray)
    target = df[1].astype(np.int)
    # classification task is to distinguish between 4 and 9
    condition = np.logical_or((target==9),(target==4))
    x = features[condition,:]
    y = target[condition]
    # label is 0, when the image depicts 4, label is 1 otherwise
    y[y == 4] = 0
    y[y == 9] = 1
    return x, y

def split_data(x, y, random_state, train_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=random_state)
    x_l_train, x_u_train, y_l_train, y_u_train = train_test_split(x_train, y_train, train_size=train_size, random_state=random_state)
    return x_l_train, y_l_train, x_u_train, x_test, y_test

def prepare_datasets():
    data = dict()
    
    x, y = read_binary_pendigits("C:\\Users\\sonya\\Desktop\\SSL\\datasets\\pendigits")
    data["pendigits"] = ([x, y])
    return data

#main
partitions = [10]
data = prepare_datasets()
params = [0.2, 0.1, 1.1] # T0, alpha, c0

start_time = time.time()
for k,v in data.items():
    print("\nFor dataset '"+k+"':\n")
    for p in partitions:
        acc = 0
        for random_state in range(0, 100, 10):
           x_l_train, y_l_train, x_u_train, x_test, y_test = split_data(v[0], v[1], random_state, p)
           rf = SemiSupervisedRandomForest()
           rf.train(x_l_train, y_l_train, x_u_train, random_state, params[0], params[1], params[2])
           y_predicted = rf.predict(x_test)
           acc += accuracy_score(y_test, y_predicted)
        acc /= 10
        print("Accuracy(with labeled_size = " + str(p) + "): " + str(acc))
print("--- %s seconds ---" % (time.time() - start_time))