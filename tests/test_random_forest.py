import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from src.algorithms.random_forest import RandomForest

X, y = make_classification(n_samples=100, n_features=10,
                           n_informative=3, n_redundant=0, n_repeated=0,
                           n_classes=3, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
clf.fit(X, y)
x = [[0, 0, 0, 0]]
pred1 = clf.predict(X)
print(accuracy_score(y, pred1))

print('---------------------------------------------------')
clf2 = RandomForest(random_state=0)
# tt = clf2.fit(X, y)
# pred2 = clf2.predict(X)
# print(accuracy_score(y, pred2))


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




feature = np.array(['blue', 'blue', 'brown', 'green', 'green', 'brown', 'green', 'blue'])