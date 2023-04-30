"""
Implementation of the Decision Forest algorithm
"""
import copy
from enum import Enum, auto
from typing import Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy.stats import mode

from sklearn.tree import DecisionTreeClassifier
from src.algorithms.decision_tree import DecisionTree


class RandomFeaturesMethods(Enum):
    SQUARED = auto()
    LOG = auto()
    INT14 = auto()
    INT12 = auto()
    INT34 = auto()
    RUNIF = auto()


class BaseForest:
    def __init__(self,
                 bootstrap: bool,
                 random_subspace: bool,
                 random_state: int = 0,
                 n_trees: int = 100,
                 max_depth: int = 2,
                 min_samples_split: int = 2,
                 max_random_features: Union[str, int] = 'sqrt'):

        # Hyper parameters
        self.bootstrap = bootstrap
        self.random_subspace = random_subspace
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_random_features = max_random_features

        self.base_learner = DecisionTree(random_state=self.random_state,
                                         max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         max_random_features=self.max_random_features,
                                         random_subspace=self.random_subspace)

        # Data
        self.n_samples: Optional[int] = None
        self.n_features: Optional[int] = None
        self.trained_trees: list = []
        self.feature_subsets: list = []

    @property
    def F(self) -> int:
        method_functions = {'SQUARED': lambda n: int(np.sqrt(n)),
                            'LOG': lambda n: int(np.log2(n)),
                            'INT14': lambda n: int(n/4),
                            'INT12': lambda n: int(n/2),
                            'INT34': lambda n: int(3*n/4),
                            'RUNIF': lambda n: np.random.randint(1, n+1)}

        return method_functions[self.max_random_features.name](
            self.n_features)

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
                  'random_state': self.random_state}
        self.base_learner.set_params(**params)

        # List of all DecisionTrees instances
        trees = [self._make_one_tree(rs_generator) for _ in
                 range(self.n_trees)]

        # Train trees
        for tree in trees:
            random_state = tree.random_state
            if self.bootstrap:
                X_bootstrap, y_bootstrap = self._make_bootstrap(X, y,
                                                                random_state)
                tree.fit(X_bootstrap, y_bootstrap)
                # self.feature_subsets.append(features_idxs)
                self.trained_trees.append(tree)
            else:
                X_random_feature, features_idxs = self._random_features(X,
                                                                        random_state)
                tree.fit(X_random_feature, y)
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

    def _make_one_tree(self,
                       rs_generator: np.random.RandomState) -> DecisionTree:

        # copy the base learner algorithm (Decision Tree CART)
        tree = copy.deepcopy(self.base_learner)

        # get random state
        random_state = rs_generator.randint(np.iinfo(np.int32).max)
        to_set = {'random_state': random_state}

        # set random state
        tree.set_params(**to_set)
        return tree

    def _check_types(self) -> None:
        if not isinstance(self.random_state, int):
            raise TypeError('random_state must be an integer')
        if not isinstance(self.n_trees, int):
            raise TypeError('n_trees must be an integer')
        if not isinstance(self.random_features_method, RandomFeaturesMethods):
            raise TypeError(
                'random_features_method must be a '
                'RandomFeaturesMethod or integer')
        if not isinstance(self.base_learner, DecisionTree):
            raise TypeError('base_learner must be an instance of DecisionTree')

    def _random_features(self,
                         X: np.ndarray,
                         random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.choice(self.n_features,
                                      self.F,
                                      replace=False)
        X_selected_feature = X[:, indices]
        return X_selected_feature, indices

    def _make_bootstrap(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.randint(0, self.n_samples, self.n_samples)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap

    # def predict_proba(self, X: np.ndarray) -> np.ndarray:
    #     probabilities = []
    #     for i, tree in enumerate(self.trained_trees):
    #         feature_idxs = self.feature_subsets[i]
    #         X_subset = X[:, feature_idxs]
    #         probabilities.append(tree.predict_proba(X_subset))
    #
    #     probabilities = np.array(probabilities)
    #
    #     return np.mean(probabilities, axis=0)


class DecisionForest(BaseForest):
    def __init__(self,
                 random_state: int = 0,
                 n_trees: int = 100,
                 max_depth: int = 2,
                 min_samples_split: int = 2,
                 random_features_method: RandomFeaturesMethods =
                 RandomFeaturesMethods.SQUARED,
                 bootstrap: bool = False):
        super().__init__(random_state=random_state,
                         n_trees=n_trees,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         random_features_method=random_features_method,
                         bootstrap=bootstrap)


class RandomForest(BaseForest):
    def __init__(self,
                 random_state,
                 n_trees,
                 max_depth,
                 min_samples_split,
                 random_features,
                 random_subspace,
                 bootstrap: bool = True):
        super().__init__(random_state=random_state,
                         n_trees=n_trees,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         max_random_features=random_features,
                         bootstrap=bootstrap)
