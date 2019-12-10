# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from .ssclustering import SSClustering


class SSLGBMClassifier(object):
    def __init__(
        self, eps: float, min_points: int, random_state: int, **kwargs
    ) -> None:
        self.clustering = SSClustering(eps, min_points)
        # self.model = LGBMClassifier(random_state=random_state, **kwargs)
        self.model = RandomForestClassifier(random_state=random_state, **kwargs)

    def fit(self, ldata, labels, udata) -> None:
        X_train, y_train = self.clustering.fit(ldata, labels, udata)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
