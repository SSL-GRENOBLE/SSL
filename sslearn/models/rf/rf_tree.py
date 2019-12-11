from sklearn.metrics import accuracy_score
from sklearn import tree


class RFTree:
<<<<<<< HEAD
    def train(self, x, y, random_state):
        self.clf = None
        self.clf = tree.DecisionTreeClassifier(
            random_state=random_state, max_features="sqrt", splitter="random"
        )  # add random state
        self.clf.fit(x, y)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def predict(self, x):
        return self.clf.predict(x)

    def count_oobe(self, x, y):
        y_predicted = self.clf.predict(x)
        return 1 - accuracy_score(y, y_predicted)
=======
    def __init__(self, random_state):
        self.clf = tree.DecisionTreeClassifier(
            random_state=random_state, max_features="sqrt", splitter="random"
        )

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

    def count_oobe(self, X, y):
        y_predicted = self.clf.predict(X)
        return 1 - accuracy_score(y, y_predicted)

    def get_classes(self):
        return self.clf.classes_
>>>>>>> Refactor random forest
