import numpy as np
import os
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


RANDOM_STATE_VALUES = np.arange(0, 100, 10)
N_LABELED_EXAMPLES = [15, 30, 50]
DATASET_NAME = 'banknote_authentication'


path = os.path.join('dataset', '01_banknote_authentication.txt')
df = pd.read_csv(path, sep=',', header=None)
X = df.values[:, :-1].astype(float)
y = df.values[:, -1].astype(int)


def split_data_generator(X, y, labeled_size):
    for random_state in RANDOM_STATE_VALUES:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=labeled_size,
                                                            stratify=y, random_state=random_state)
        yield X_train, X_test, y_train, y_test


def test_supervised_model(model):
    print(f'Dataset - {DATASET_NAME}')
    print('Supervised model\n')

    for labeled_size in N_LABELED_EXAMPLES:
        scores = []
        for (X_train, X_test, y_train, y_test) in tqdm(split_data_generator(X, y, labeled_size), total=len(RANDOM_STATE_VALUES)):
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            scores.append(accuracy_score(y_test, prediction))

        print(f'Labeled size = {labeled_size}, accuracy = {np.mean(scores)}')


def test_semisupervised_model(model):
    print(f'Dataset - {DATASET_NAME}')
    print('Semi-supervised model\n')

    for labeled_size in N_LABELED_EXAMPLES:
        scores = []
        for (X_train, X_test, y_train, y_test) in tqdm(split_data_generator(X, y, labeled_size), total=len(RANDOM_STATE_VALUES)):
            model.fit(X_train, y_train, X_test)
            prediction = model.predict(X_test)
            scores.append(accuracy_score(y_test, prediction))

        print(f'Labeled size = {labeled_size}, accuracy = {np.mean(scores)}')
