"""
Implementation of the Random Forest algorithm.
"""
import copy
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np
from src.algorithms.decision_tree import DecisionTree


class NumberRandomFeatures(Enum):
    SQUARED = auto()
    LOG = auto()


@dataclass
class RandomForest:
    random_state: int = field(default=0)
    n_trees: int = field(default=100)
    n_random_features: Union[int, NumberRandomFeatures] = field(default=NumberRandomFeatures.SQUARED)
    base_learner: DecisionTree = field(default=DecisionTree())
    max_depth: int = field(default=2)
    trained_trees: Optional[list] = field(init=False, default=None)
    n_features: Optional[int] = field(init=False, default=None)
    n_samples: Optional[int] = field(init=False, default=None)

    def _check_type(self) -> None:
        if not isinstance(self.random_state, int):
            raise TypeError('random_state must be an integer')
        if not isinstance(self.n_trees, int):
            raise TypeError('n_trees must be an integer')
        if not isinstance(self.n_random_features, (NumberRandomFeatures, int)):
            raise TypeError('n_random_features must be a NumberRandomFeatures or integer')
        if not isinstance(self.base_learner, DecisionTree):
            raise TypeError('base_learner must be an instance of DecisionTree')

    def _make_one_tree(self,
                       rs_generator: np.random.RandomState) -> DecisionTree:
        # copy the base learner algorithm (Decision Tree CART)
        tree = copy.deepcopy(self.base_learner)
        # get random state
        random_state = rs_generator.randint(np.iinfo(np.int32).max)
        to_set = {'random_state': random_state}
        print(to_set)
        # set random state
        tree.set_params(**to_set)
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        # set max features
        if self.n_random_features is NumberRandomFeatures.SQUARED:
            max_features = int(np.sqrt(self.n_features))
        elif self.n_random_features is NumberRandomFeatures.LOG:
            max_features = int(np.log2(self.n_features))
        else:
            if isinstance(self.n_random_features, int):
                max_features = self.n_random_features

        # check types
        self._check_type()

        # random state generator with self.random_state
        rs_generator = np.random.RandomState(self.random_state)

        # base_learner
        params = {'max_depth': self.max_depth,
                  'max_features': 'sqrt',
                  'random_state': self.random_state}
        self.base_learner.set_params(**params)

        # list of different DecisionTrees instances
        trees = [self._make_one_tree(rs_generator) for i in
                 range(self.n_trees)]

        # bootstrapped dataset
        # indices = .randint(0, n_samples, n_samples_bootstrap)

        # train each tree and save it in a list

        self.trained_trees = [tree.fit(X, y) for tree in trees]

        assert len(self.trained_trees) == self.n_trees

    def _fit_one_tree(self, X, y, tree, max_features):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        rs_generator = np.random.RandomState(tree.random_state)
        indices = rs_generator.randint(0, self.n_samples,
                                                 self.n_samples)
        X_bootstrapped = X[indices]


    def predict(self, X: Union[list, np.ndarray]) -> np.ndarray:
        """
        Predict the class labels for the input data.

        Args:
            X: The input data to predict on.

        Returns:
            The predicted class labels.
        """

        if isinstance(X, list):
            X = np.array(X)
        assert X.shape[1] == self.n_features, \
            f'X should have {self.n_features} features, but it has {X.shape[1]}'
        predictions = np.array([tree.predict(X) for tree
                                in self.trained_trees]).astype(
            np.int32)
        assert predictions.shape[0] == self.n_trees
        _dist = np.apply_along_axis(np.bincount, axis=1, arr=predictions)
        prediction = np.argmax(_dist, axis=1)
        return prediction


