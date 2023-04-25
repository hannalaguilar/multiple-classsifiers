from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pandas as pd

from src.algorithms.decision_tree import DecisionTree
from src.algorithms.random_forest import RandomForest


CURRENT_PATH = Path(__file__).parent


def test_gini():
    dec_tree = DecisionTree()
    left_class = np.array(['C+', 'C+', 'C-', 'C+', 'C-', 'C+'])
    right_class = np.array(['C-', 'C-'])

    assert dec_tree._gini_index(left_class, right_class) == 1/3


def test_best_split():
    # data
    df = pd.read_csv(CURRENT_PATH / 'test_data.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    dec_tree = DecisionTree()
    best_gini, best_feature, best_subset = dec_tree._best_split(X, y)

    assert np.round(best_gini, 3) == 0.2
    assert best_feature == 'eye_color'
    assert best_subset == [['brown', 'green'], ['blue']]


def test_decision_tree():
    # data
    random_number = np.random.randint(np.iinfo(np.int32).max)
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=3, n_redundant=0, n_repeated=0,
                               n_classes=3, random_state=random_number)

    # sklearn
    clf_sklearn = DecisionTreeClassifier(random_state=0, max_depth=2)
    clf_sklearn.fit(X, y)


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
