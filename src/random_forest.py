"""
Implementation of the Random Forest algorithm.
"""
import copy
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class NumberRandomFeatures(Enum):
    SQUARED = 'squared'
    LOG = 'log'


@dataclass
class RandomForest:
    random_state: int = field(default=0)
    n_trees: int = field(default=100)
    n_random_features: Union[int, NumberRandomFeatures] = field(default=NumberRandomFeatures.SQUARED)
    base_learner: DecisionTreeClassifier = field(default=DecisionTreeClassifier(
                                                     criterion='gini'))
    trained_trees: Optional[list] = field(init=False, default=None)
    n_features: Optional[int] = field(init=False, default=None)

    def _check_type(self) -> None:
        if not isinstance(self.random_state, int):
            raise TypeError('random_state must be an integer')
        if not isinstance(self.n_trees, int):
            raise TypeError('n_trees must be an integer')
        if not isinstance(self.n_random_features, NumberRandomFeatures) or \
                not isinstance(self.n_random_features, int):
            raise TypeError('n_random_features must be a NumberRandomFeatures or integer')
        if not isinstance(self.base_learner, DecisionTreeClassifier):
            raise TypeError('base_learner must be an instance of '
                            'DecisionTreeClassifier')

    def _make_one_tree(self,
                       rs_generator: np.random.RandomState) -> DecisionTreeClassifier:
        # copy the base learner algorithm (Decision Tree CART)
        tree = copy.deepcopy(self.base_learner)
        # get random state
        random_state = rs_generator.randint(np.iinfo(np.int32).max)
        to_set = {'random_state': random_state}
        # set random state
        tree.set_params(**to_set)
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        self.n_features = X.shape[1]

        # set max features
        if self.n_random_features == NumberRandomFeatures.SQUARED:
            max_features = max(1, int(np.sqrt(self.n_features)))
        elif self.n_random_features == NumberRandomFeatures.LOG:
            max_features = max(1, int(np.log2(self.n_features)))
        if isinstance(self.n_random_features, int):
            max_features = self.n_random_features

        # check types
        self._check_type()
        # random state generator with self.random_state
        rs_generator = np.random.RandomState(self.random_state)
        # list of different DecisionTrees instances
        trees = [self._make_one_tree(rs_generator) for i in
                 range(self.n_trees)]
        # train each tree and save it in a list
        self.trained_trees = [tree.fit(X, y) for tree in trees]

        assert len(self.trained_trees) == self.n_trees

    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        if isinstance(X, list):
            X = np.array(X)
        assert X.shape[1] == self.n_features, f'X should have {self.n_features} features, but it has {X.shape[1]}'
        predictions = np.array([tree.predict(X) for tree
                                in self.trained_trees]).astype(
            np.int32).flatten()
        assert predictions.shape[0] == self.n_trees
        prediction = np.argmax(np.bincount(predictions))
        return prediction


from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=3, n_redundant=0, n_repeated=0,
                           n_classes=3, random_state=42)

clf = RandomForestClassifier(max_depth=2, random_state=0, max_features=3)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
DecisionTreeClassifier()

clf2 = RandomForest(random_state=0)
tt = clf2.fit(X, y)
preds = np.array([estimator.predict([[0, 0, 0, 0]]) for estimator in
                  clf.estimators_]).astype(np.int32).flatten()
np.argmax(np.bincount(preds))
clf2.predict([[0, 0]])
