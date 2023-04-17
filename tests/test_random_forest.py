import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.random_forest import RandomForest

X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=3, n_redundant=0, n_repeated=0,
                           n_classes=3, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
x = [[0, 0, 0, 0]]
pred1 = clf.predict(X)


clf2 = RandomForest(random_state=0)
tt = clf2.fit(X, y)
pred2 = clf2.predict(X)

