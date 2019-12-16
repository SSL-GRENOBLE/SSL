from sklearn.ensemble import RandomForestClassifier

from .ssclustering import SSClustering


class SSClusteringClassifier(object):
    def __init__(
        self,
        eps: float,
        min_points: int,
        random_state: int = 42,
        use_border_points: bool = False,
        **kwargs
    ) -> None:
        self.clustering = SSClustering(eps, min_points, use_border_points)
        self.model = RandomForestClassifier(random_state=random_state, **kwargs)

    def fit(self, ldata, labels, udata) -> None:
        X_train, y_train = self.clustering.transform(ldata, labels, udata)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
