from sklearn.metrics import accuracy_score
from sklearn import tree

class RFTree:
    def train(self, x, y, random_state):
        self.clf = None
        self.clf = tree.DecisionTreeClassifier(random_state=random_state, max_features="sqrt", splitter="random") # add random state
        self.clf.fit(x, y)
    
    def predict_proba(self, x):
        return self.clf.predict_proba(x)
    
    def predict(self, x):
        return self.clf.predict(x)
    
    def count_oobe(self, x, y):
        y_predicted = self.clf.predict(x)
        return 1-accuracy_score(y, y_predicted)
