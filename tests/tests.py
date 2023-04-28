"""
Test for the tree algorithms
"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

from src.algorithms.decision_tree import DecisionTree
from src.algorithms.random_forest import RandomForest

CURRENT_PATH = Path(__file__).parent


def test_gini():
    # data
    X, y = load_iris(return_X_y=True)

    # gini sklearn
    clf_sklearn = DecisionTreeClassifier(random_state=0,
                                         max_leaf_nodes=3)
    clf_sklearn.fit(X, y)
    gini_sklearn = clf_sklearn.tree_.impurity[0]

    # my algorithm
    dec_tree = DecisionTree()
    gini_src = dec_tree._gini(y)

    # test
    assert gini_sklearn == gini_src


def test_gini_gain():
    # data
    left_class = np.array(['C+', 'C+', 'C-', 'C+', 'C-', 'C+'])
    right_class = np.array(['C-', 'C-'])

    # my algorithm
    dec_tree = DecisionTree()
    gini_gain = dec_tree._gini_gain(left_class, right_class)

    # test
    assert gini_gain == 0.5 - 1 / 3


def test_best_split():
    # data
    df = pd.read_csv(CURRENT_PATH / 'test_data.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # my algorithm
    dec_tree = DecisionTree()
    best_gini, best_feature_idx, best_threshold = dec_tree._best_split(X, y,
                                                                       [0, 1,
                                                                        2])
    # test
    assert np.round(best_gini, 3) == 0.5 - 0.2
    assert best_feature_idx == 0
    assert best_threshold == 'blue'


def test_decision_tree():
    random_number = np.random.randint(np.iinfo(np.int32).max)

    # data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=random_number)

    # sklearn
    clf_sklearn = DecisionTreeClassifier(random_state=0, max_leaf_nodes=3)
    clf_sklearn.fit(X_train, y_train)
    acc_test_sklearn = clf_sklearn.score(X_test, y_test)

    # my algorithm
    clf_src = DecisionTree()
    clf_src._best_split(X_train, y_train)
    clf_src.fit(X_train, y_train)
    y_pred_src = clf_src.predict(X_test)
    acc_test_src = accuracy_score(y_test, y_pred_src)

    # assert if the difference is less than 10%
    print(f'sklearn:{acc_test_sklearn}, my_algorithm: {acc_test_src}')
    np.testing.assert_allclose(acc_test_sklearn, acc_test_src, rtol=0.1)


def test_random_forest():
    # data
    random_number = np.random.randint(np.iinfo(np.int32).max)
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=3, n_redundant=0, n_repeated=0,
                               n_classes=3, random_state=0)

    # sklearn
    clf_sklearn = RandomForestClassifier(random_state=0,
                                         max_depth=2,
                                         n_estimators=100)
    clf_sklearn.fit(X, y)
    accuracy_sk = clf_sklearn.score(X, y)

    # my algorithm
    clf_src = RandomForest(random_state=0,
                           max_depth=2,
                           n_trees=100)
    clf_src.fit(X, y)
    pred_clf_src = clf_src.predict(X)
    accuracy_src = accuracy_score(y, pred_clf_src)

    # compare
    print(f'accuracy sklearn: {accuracy_src:.3f}, '
          f'accuracy my algorithm: {accuracy_src:.3f}')
    np.testing.assert_allclose(accuracy_sk, accuracy_src, atol=0.15)
