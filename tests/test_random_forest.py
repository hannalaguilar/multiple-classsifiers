import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from src.algorithms.random_forest import RandomForest


def test_random_forest():
    random_number = np.random.randint(np.iinfo(np.int32).max)
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=3, n_redundant=0, n_repeated=0,
                               n_classes=3, random_state=random_number)

    clf_sklearn = RandomForestClassifier(random_state=0,
                                         max_depth=2,
                                         n_estimators=100)
    clf_sklearn.fit(X, y)
    accuracy_sk = clf_sklearn.score(X, y)

    clf_src = RandomForest(random_state=0,
                           max_depth=2,
                           n_trees=100)
    clf_src.fit(X, y)
    pred_clf_src = clf_src.predict(X)
    accuracy_src = accuracy_score(y, pred_clf_src)

    print(f'accuracy sklearn: {accuracy_src:.3f}, '
          f'accuracy my algorithm: {accuracy_src:.3f}')

    np.testing.assert_allclose(accuracy_sk, accuracy_src, atol=0.15)



def test_gini():
    def _gini(self, left_labels, right_labels):
        n_left = len(left_labels)
        n_right = len(right_labels)
        n_total = n_left + n_right
        gini_left = 1 - np.sum(
            [(np.sum(left_labels == c) / n_left) ** 2 for c in
             np.unique(left_labels)])
        gini_right = 1 - np.sum(
            [(np.sum(right_labels == c) / n_right) ** 2 for c in
             np.unique(right_labels)])
        return (n_left / n_total) * gini_left + (
                n_right / n_total) * gini_right


feature = np.array(
    ['blue', 'blue', 'brown', 'green', 'green', 'brown', 'green', 'blue'])
