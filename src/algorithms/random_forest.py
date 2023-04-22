"""
Implementation of the Random Forest algorithm.
"""
import copy
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier

from src.algorithms.decision_tree import DecisionTree


class RandomFeaturesMethods(Enum):
    SQUARED = auto()
    LOG = auto()
    INT = auto()


@dataclass
class RandomForest:
    # Hyperparameters
    random_state: int = field(default=0)
    n_trees: int = field(default=100)
    max_depth: int = field(default=2)
    random_features_method: RandomFeaturesMethods = field(default=RandomFeaturesMethods.SQUARED)
    base_learner: DecisionTree = field(default=DecisionTreeClassifier())

    # Data
    n_samples: Optional[int] = field(init=False, default=None)
    n_features: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.trained_trees = []
        self.feature_subsets = []

    @property
    def n_random_features(self) -> int:
        method_functions = {'SQUARED': lambda n: int(np.sqrt(n)),
                            'LOG': lambda n: int(np.log2(n)),
                            'INT': lambda n: n}

        return method_functions[self.random_features_method.name](
            self.n_features)

    def _check_types(self) -> None:
        if not isinstance(self.random_state, int):
            raise TypeError('random_state must be an integer')
        if not isinstance(self.n_trees, int):
            raise TypeError('n_trees must be an integer')
        if not isinstance(self.random_features_method, RandomFeaturesMethods):
            raise TypeError(
                'random_features_method must be a '
                'RandomFeaturesMethod or integer')
        if not isinstance(self.base_learner, DecisionTreeClassifier):
            raise TypeError('base_learner must be an instance of DecisionTree')

    def _bootstrap(self, 
                   X: np.ndarray,
                   y: np.ndarray,
                   random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.randint(0, self.n_samples, self.n_samples)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap

    def _random_features(self,
                         X: np.ndarray,
                         random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.choice(self.n_features,
                                      self.n_random_features,
                                      replace=False)
        X_selected_feature = X[:, indices]
        return X_selected_feature, indices

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

        assert X.ndim == 2, 'X must be two-dimensional array'

        # Data
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        # Check types
        self._check_types()

        # Random state generator with self.random_state
        rs_generator = np.random.RandomState(self.random_state)

        # Base_learner
        params = {'max_depth': self.max_depth,
                  'max_features': None,
                  'random_state': self.random_state}
        self.base_learner.set_params(**params)

        # List of all DecisionTrees instances
        trees = [self._make_one_tree(rs_generator) for _ in
                 range(self.n_trees)]

        # Train trees
        for tree in trees:
            random_state = tree.random_state
            X_bootstrap, y_bootstrap = self._bootstrap(X, y, random_state)
            X_bootstrap_random_feature, features_idxs = self._random_features(
                X_bootstrap,
                random_state)
            tree.fit(X_bootstrap_random_feature, y_bootstrap)
            self.feature_subsets.append(features_idxs)
            self.trained_trees.append(tree)

        assert len(self.trained_trees) == self.n_trees

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
        assert X.ndim == 2, 'X must be two-dimensional array'
        assert X.shape[1] == self.n_features, \
            f'X should have {self.n_features} features, but it has {X.shape[1]}'

        predictions = []
        for i, tree in enumerate(self.trained_trees):
            feature_idxs = self.feature_subsets[i]
            X_subset = X[:, feature_idxs]
            predictions.append(tree.predict(X_subset))
        predictions = np.array(predictions)

        assert len(predictions) == self.n_trees

        result = mode(predictions, keepdims=True)
        predictions = result.mode.ravel()

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = []
        for i, tree in enumerate(self.trained_trees):
            feature_idxs = self.feature_subsets[i]
            X_subset = X[:, feature_idxs]
            probabilities.append(tree.predict_proba(X_subset))

        probabilities = np.array(probabilities)

        return np.mean(probabilities, axis=0)

# array([99, 50, 32, 64, 54, 41, 39, 67, 73, 70,  1, 49,  7, 59, 31, 41, 39,
#        34, 48, 39, 57, 31, 49, 90, 76, 11, 85, 80, 69, 48, 81, 28, 20,  3,
#        86, 16, 75, 78, 80, 26, 83, 32, 87, 96, 89, 54,  6, 38,  6, 20, 79,
#        62, 36, 30, 60, 16,  8, 97,  0, 43, 47, 47,  9,  1, 92, 64, 86, 92,
#        87, 66, 92,  1, 90, 47, 37, 56, 34, 88,  6, 95, 69, 33, 56, 20, 85,
#        97, 22, 85, 45, 23, 79, 47,  1, 30,  4, 93,  3, 68, 55, 20])
