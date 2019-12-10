from .rf import RandomForest

import random
from scipy.stats import bernoulli
import numpy as np


class SemiSupervisedRandomForest(RandomForest):
    def __init__(self, random_state, T0=0.2, alpha=0.1, c0=1.1):
        super().__init__(random_state)
        self.T0 = T0
        self.c0 = c0
        self.alpha = alpha

    def __change_distribution(self, probs, alpha, T):
        newprobs = []
        for i in range(0, len(probs)):
            Z = 0
            newprobs_i = []
            for p in probs[i]:
                # newp = p**(-alpha/T)/np.exp(1) if p > 0 else 0
                newp = np.exp(-(alpha * np.log2(p) + T) / T) if p > 0 else 0
                newprobs_i.append(newp)
                Z += newp

            if np.isinf(Z) or np.any(np.isinf(newprobs_i)):
                return [], True
            newprobs_i = [p / Z for p in newprobs_i]
            newprobs.append(newprobs_i)
        return newprobs, False

    def __retrain_random_forest(self, x_l, y_l, x_u, p):
        oobe = 0
        np.random.seed(self.random_state)
        for i in range(0, self.N):
            y_hat = np.random.binomial(1, p)
            x_train = np.concatenate((x_l, x_u))
            y_train = np.concatenate((y_l, y_hat))
            rfTree = self.trees[i]
            x_i, y_i, x_oob, y_oob = self._generate_tree_data(x_train, y_train, x_l, y_l)
            rfTree.train(x_i, y_i, self.random_state)
            oobe += rfTree.count_oobe(x_oob, y_oob)
        oobe /= self.N
        return oobe

    def fit(self, x_l, y_l, x_u):
        random.seed(self.random_state)
        steps = 20
        # T0 = 0.2 # T_m ~ T0*exp^(-m) - cooling function
        # alpha = 0.1
        super().fit(x_l, y_l)  # train RF with labeled data
        Told = self.T0
        m = 0  # set epoch
        oobe = 0
        while True:
            Tnew = Told / self.c0
            m = m + 1
            target = self.predict_proba(x_u)
            newtarget, is_overflow = self.__change_distribution(
                target, self.alpha, Tnew
            )
            if is_overflow:
                # print("Overflow happened, stopped after ", m, " steps")
                break
            p = [x[1] for x in newtarget]

            oobe = self.__retrain_random_forest(x_l, y_l, x_u, p)
            Told = Tnew
            if m >= steps:
                break
        if oobe > self.oobe:
            print(
                "[D]Semi-supervised approach was discarded with oobe: "
                + str(oobe)
                + ", oobe for pure RF: "
                + str(self.oobe)
            )
            random.seed(self.random_state)
            super().fit(x_l, y_l)

        else:
            print(
                "[A]Semi-supervised approach was accepted with oobe: "
                + str(oobe)
                + ", oobe for pure RF: "
                + str(self.oobe)
            )
            self.oobe = oobe

    def predict(self, x):
        return super().predict(x)
