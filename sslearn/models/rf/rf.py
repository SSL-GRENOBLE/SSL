from .rf_tree import RFTree

import numpy as np
import random

class RandomForest:
    def __init__(self, random_state, N = 100):
        self.N = N
        self.trees = []
        self.oobe = 0
        self.random_state = random_state
        
    def _prepare_train_data(self, x, y):
        n_samples = len(y)
        steps = 10
        st = 0
        x_i = []
        y_i = []
        choices = []
        while st < steps:
            choices = random.choices(range(0, len(y)), k=n_samples) # indices of samples
            y_i = [y[i] for i in choices]
            st += 1
            if y_i.count(y_i[0]) != len(y_i):
                break
        if st == steps:
            sys.exit("We couldn't generate good data for RF")
        x_i = [x[i] for i in choices]
        return x_i, y_i, choices
    
    def _prepare_oob_data(self, x, y, indices):
        x_i = [x[i] for i in range(0, len(x)) if i not in indices]
        y_i = [y[i] for i in range(0, len(x)) if i not in indices]
        return x_i, y_i
    
    def fit(self, x, y):
        random.seed(self.random_state)
        self.trees = []
        self.oobe = 0
        for i in range(0, self.N):
            rfTree = RFTree()
            x_i, y_i, idx = self._prepare_train_data(x, y)
            #print("Train tree with features of size: ", len(x_i[0]))
            rfTree.train(x_i, y_i, self.random_state)
            x_oob, y_oob = self._prepare_oob_data(x, y, idx)
            self.trees.append(rfTree)
            self.oobe += rfTree.count_oobe(x_oob, y_oob)
        
        self.oobe /= self.N
    
    def get_oobe(self):
        return self.oobe
    
    def predict_proba(self, x):
        target = self.trees[0].predict_proba(x)
        for i in range(1, self.N):
            target = np.add(self.trees[i].predict_proba(x), target)        
        for i in range(0, len(target)):
            target[i] = [x / self.N for x in target[i]]
        
        return target.tolist()
    
    def predict(self, x):
        y = self.trees[0].predict(x)
        for i in range(1, self.N):
            y = np.add(self.trees[i].predict(x), y)
        res = []
        
        for i in range(0, len(x)):
            if y[i] > self.N/2:
                res.append(1)
            else:
                res.append(0)
        return res