from .rf import RandomForest

import random
import numpy as np


class SemiSupervisedRandomForest(RandomForest):
    """SemiSupervisedRandomForest constructor.

        Attributes:
            random_state - random state
            T0 - initial temperature
            alpha - proportional parameter
            c0 - cooling parameter
            max_iter - number of iterations
    """

    def __init__(self, random_state, T0=0.005, alpha=0.1, c0=1.1, max_iter=20):
        super().__init__(random_state)
        self.T0 = T0
        self.c0 = c0
        self.alpha = alpha
        self.max_iter = max_iter

    """Update distribution of X due to the algorith parameters.
       As the overflow is possible during the updates, updates are discarded in such case.
       Returns new distribution and the boolean result of the operation(False for overflow).
        
        Attributes:
            probs - probabilities to update
            T - current temperature
    """

    def __update_distribution(self, probs, T):
        eps = 0.01
        p_new = []
        for i in range(0, len(probs)):
            probs[i] = [eps if p_ <= 0 else p_ for p_ in probs[i]]
            p_upd = [np.exp(-(self.alpha * np.log(p) * p + T) / T) for p in probs[i]]
            Z = np.sum(p_upd)
            if np.isinf(Z) or np.any(np.isinf(p_upd)):
                return [], True
            p_new.append([p / Z for p in p_upd])
        return p_new, False

    """Generate pseudolabels according to the distribution p.
        
        Attributes:
            p - probability distribution on X_u
    """

    def __generate_pseudolabels(self, p, labels):
        y = [random.choices(range(0, len(p_)), p_)[0] for p_ in p]
        y_u = [labels[y_] for y_ in y]
        return y_u

    """Update and retrain semi-supervised random forest with labeled and pseudo-labeled data.

        Attributes:
            X_l - labeled feature matrix
            y_l - labels for X_l
            X_u - unlabeled feature matrix
            p - probability distribution on p
            labels - real labels of the classes
    """

    def __update_random_forest(self, X_l, y_l, X_u, p, labels):
        oobe = 0
        random.seed(self.random_state)
        for i in range(0, self.N):
            y_u = self.__generate_pseudolabels(p, labels)
            X_train = np.concatenate((X_l, X_u))
            y_train = np.concatenate((y_l, y_u))
            rfTree = self.trees[i]
            X_i, y_i, X_oob, y_oob = self._generate_tree_data(
                X_train, y_train, X_l, y_l
            )
            rfTree.fit(X_i, y_i)
            oobe += rfTree.count_oobe(X_oob, y_oob)
        oobe /= self.N
        return oobe

    """Fit semi-supervised random forest with labeled and unlabeled data.

        Attributes:
            X_l - initial labeled feature matrix
            y_l - initial labels for a feature matrix
            X_u - initial unlabled data
    """

    def fit(self, X_l, y_l, X_u):
        random.seed(self.random_state)
        super().fit(X_l, y_l)  # train RF with labeled data
        T = self.T0
        m = 0  # set epoch
        oobe = 0
        while m < self.max_iter:
            T = T / self.c0
            m += 1
            target, labels = super().predict_proba(X_u)
            p_new, is_overflow = self.__update_distribution(target, T)
            if is_overflow:
                break
            oobe = self.__update_random_forest(X_l, y_l, X_u, p_new, labels)
            # print("OOBE =", oobe, " on step m=", m)

        if oobe > self.oobe:
            print(
                "[D]Semi-supervised approach was discarded with oobe: "
                + str(oobe)
                + ", oobe for pure RF: "
                + str(self.oobe)
            )
            random.seed(self.random_state)
            super().fit(X_l, y_l)

        else:
            print(
                "[A]Semi-supervised approach was accepted with oobe: "
                + str(oobe)
                + ", oobe for pure RF: "
                + str(self.oobe)
            )
            self.oobe = oobe

    """Predict labels for X, returns labels y.

        Attributes:
            X - feature matrix to predict labels for it
    """

    def predict(self, X):
        return super().predict(X)
