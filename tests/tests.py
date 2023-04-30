"""
This module tests compare the performance of my implemented algorithms with
the corresponding ones in scikit-learn.

It includes the following tests:

- `test_gini()`: Test the calculation of Gini index using the implementation
    in this code and that provided by `scikit-learn`.
- `test_gini_gain()`: Test the calculation of Gini gain.
- `test_best_split()`: Test the calculation of best split on a test dataset.
- `test_decision_tree_basic_data()`: Test the implementation of the
    DecisionTree on a simple dataset.
- `test_decision_tree()`: Test the implementation of the DecisionTree
    on iris dataset.
- `test_random_forest()`: Test the implementation of RandomForest.
- `test_decision_forest()`: Test the implementation of DecisionForest.

"""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification, load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

from src.decision_tree import DecisionTree
from src.forest_ensemble import DecisionForest, RandomForest
from src.forest_tools import FMethodRF

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


def test_decision_tree_basic_data():
    # data
    df = pd.read_csv(CURRENT_PATH / 'test_data.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # my algorithm
    clf_src = DecisionTree(max_depth=2, random_subspace_node=False)
    clf_src.fit(X, y, cat_features=[0, 1, 2])
    tree = clf_src.tree

    # test
    assert tree.feature == 0
    assert tree.threshold == 'blue'
    assert tree.left.leaf_value == 0
    assert tree.right.feature == 2
    assert tree.right.threshold == 'tall'
    assert tree.right.left.leaf_value == 0
    assert tree.right.right.leaf_value == 1


def test_decision_tree():
    random_number = np.random.randint(np.iinfo(np.int32).max)

    # data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=random_number)

    # sklearn
    clf_sklearn = DecisionTreeClassifier(random_state=0,
                                         max_depth=2)
    clf_sklearn.fit(X_train, y_train)
    acc_test_sklearn = clf_sklearn.score(X_test, y_test)

    # my algorithm
    clf_src = DecisionTree(max_depth=2, random_subspace_node=False)
    clf_src.fit(X_train, y_train)
    y_pred_src = clf_src.predict(X_test)
    acc_test_src = accuracy_score(y_test, y_pred_src)

    # assert if the difference is less than 10%
    print(f'sklearn:{acc_test_sklearn:.3f}, '
          f'my_algorithm: {acc_test_src:.3f}')
    np.testing.assert_allclose(acc_test_sklearn, acc_test_src, rtol=0.12)


def test_random_forest():
    # data
    random_number = np.random.randint(np.iinfo(np.int32).max)
    X, y = make_classification(n_samples=300, n_features=10,
                               n_informative=4, n_redundant=0, n_repeated=0,
                               n_classes=4, random_state=random_number)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=random_number)

    # sklearn
    n_trees = np.random.randint(1, 100)
    clf_sklearn = RandomForestClassifier(random_state=0,
                                         max_features='sqrt',
                                         n_estimators=n_trees)
    clf_sklearn.fit(X_train, y_train)
    accuracy_sk = clf_sklearn.score(X_test, y_test)

    # my algorithm
    clf_src = RandomForest(random_state=0,
                           n_trees=n_trees,
                           max_random_features=FMethodRF.SQRT)

    clf_src.fit(X_train, y_train)
    pred_clf_src = clf_src.predict(X_test)
    accuracy_src = accuracy_score(y_test, pred_clf_src)

    # test
    print(f'accuracy sklearn: {accuracy_sk:.3f}, '
          f'accuracy my algorithm: {accuracy_src:.3f}')
    np.testing.assert_allclose(accuracy_sk, accuracy_src, atol=0.12)


def test_decision_forest():
    # data
    random_number = np.random.randint(np.iinfo(np.int32).max)
    X, y = make_classification(n_samples=300, n_features=11,
                               n_informative=3, n_redundant=0, n_repeated=0,
                               n_classes=3, random_state=random_number)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=random_number)

    # sklearn
    max_random_features = 0.75
    n_trees = np.random.randint(1, 20)
    estimator = DecisionTreeClassifier(random_state=0)

    clf_sklearn = BaggingClassifier(estimator=estimator,
                                    random_state=0,
                                    n_estimators=n_trees,
                                    bootstrap=False,
                                    max_features=max_random_features)
    clf_sklearn.fit(X_train, y_train)
    accuracy_sk = clf_sklearn.score(X_test, y_test)

    # my algorithm
    clf_src = DecisionForest(random_state=0,
                             n_trees=n_trees,
                             max_random_features=max_random_features)
    clf_src.fit(X_train, y_train)
    pred_clf_src = clf_src.predict(X_test)
    accuracy_src = accuracy_score(y_test, pred_clf_src)

    print(f'accuracy sklearn: {accuracy_sk:.3f}, '
          f'accuracy my algorithm: {accuracy_src:.3f}')
    np.testing.assert_allclose(accuracy_sk, accuracy_src, atol=0.12)
