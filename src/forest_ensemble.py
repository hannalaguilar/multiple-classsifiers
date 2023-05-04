"""
This module implements the RandomForest and DecisionForest classes
for classification problems.

- RandomForest is an ensemble method that uses a collection of
decision trees with random bootstrap resamples of the data and random features
used in the splitting of the nodes, which makes it robust against overfitting.

- DecisionForest is an ensemble method that uses a collection of decision
trees using a subspace of features.
"""
from typing import Optional, Union
import numpy as np
from scipy.stats import mode

from src.decision_tree import DecisionTree
from src.forest_tools import (check_types, make_one_tree, FMethodRF, FMethodDF)


class RandomForest:
    def __init__(self,
                 random_state: int = 0,
                 n_trees: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_random_features: Union[FMethodRF, int] = FMethodRF.SQRT,
                 bootstrap: bool = True,
                 random_subspace_node: bool = True,
                 ):

        # Hyper parameters
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_random_features = max_random_features
        self.bootstrap = bootstrap
        self.random_subspace_node = random_subspace_node

        clf = DecisionTree(random_state=self.random_state,
                           max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split,
                           random_subspace_node=self.random_subspace_node)

        self.base_learner = clf

        # Data
        self.n_samples: Optional[int] = None
        self.n_features: Optional[int] = None

        # Trained
        self.trained_trees: list = []
        self.feature_importance: list = []

    @property
    def F(self) -> int:
        method_functions = {FMethodRF.SQRT: lambda n: int(np.sqrt(n)),
                            FMethodRF.LOG: lambda n: int(np.log2(n))}

        if isinstance(self.max_random_features, int) and \
                0 < self.max_random_features <= self.n_features:
            return self.max_random_features
        elif isinstance(self.max_random_features, FMethodRF):
            return method_functions[self.max_random_features](self.n_features)
        else:
            raise TypeError('random_features must be FMethodRF '
                            'instance or an integer between 0 and n_features')

    def fit(self, X: np.ndarray,
            y: np.ndarray,
            cat_features: Optional[list] = None) -> None:

        assert X.ndim == 2, 'X must be two-dimensional array'

        # Data
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        # Check types
        check_types(self.random_state, self.n_trees, self.max_depth,
                    self.min_samples_split, self.base_learner)

        # Random state generator with self.random_state
        rs_generator = np.random.RandomState(self.random_state)

        # Base_learner
        params = {'F': self.F}
        self.base_learner.set_params(**params)

        # List of all DecisionTrees instances
        trees = [make_one_tree(self.base_learner, rs_generator) for _ in
                 range(self.n_trees)]

        # Train trees
        for tree in trees:
            random_state = tree.random_state
            X_bootstrap, y_bootstrap = self._make_bootstrap(X, y,
                                                            random_state)
            tree.fit(X_bootstrap, y_bootstrap, cat_features=cat_features)
            self.trained_trees.append(tree)
            self.feature_importance.append(tree.feature_importance)

        assert len(self.trained_trees) == self.n_trees
        self.feature_importance = np.mean(self.feature_importance, axis=0)
        assert round(self.feature_importance.sum(), 3) == 1.0

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
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)

        assert len(predictions) == self.n_trees

        result = mode(predictions, keepdims=True)
        predictions = result.mode.ravel()

        return predictions

    def _make_bootstrap(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.randint(0, self.n_samples, self.n_samples)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap


class DecisionForest:
    def __init__(self,
                 random_state: int = 0,
                 n_trees: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_random_features: Union[FMethodDF, float] = 0.5,
                 ):

        # Hyper parameters
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_random_features = max_random_features

        clf = DecisionTree(random_state=self.random_state,
                           max_depth=self.max_depth,
                           min_samples_split=self.min_samples_split)

        self.base_learner = clf

        # Data
        self.n_samples: Optional[int] = None
        self.n_features: Optional[int] = None

        # Trained
        self.trained_trees: list = []
        self.feature_subsets: list = []
        self.feature_importance: Union[list, np.ndarray] = []

    @property
    def F(self):
        method_functions = {FMethodDF.RUNIF: lambda n:
        np.random.randint(1, n + 1)}

        if isinstance(self.max_random_features, float) and \
                0 < self.max_random_features <= 1:
            return int(self.max_random_features * self.n_features)

        elif isinstance(self.max_random_features, FMethodDF):
            return method_functions[self.max_random_features](self.n_features)
        else:
            raise TypeError('max_random_features must be a FMethodDF '
                            'instance or a float between 0 - 1')

    def fit(self, X: np.ndarray,
            y: np.ndarray,
            cat_features: Optional[list] = None) -> None:

        assert X.ndim == 2, 'X must be two-dimensional array'

        # Data
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        # Check types
        check_types(self.random_state, self.n_trees, self.max_depth,
                    self.min_samples_split, self.base_learner)

        # Random state generator with self.random_state
        rs_generator = np.random.RandomState(self.random_state)

        # Base_learner
        params = {'F': self.F}
        self.base_learner.set_params(**params)

        # List of all DecisionTrees instances
        trees = [make_one_tree(self.base_learner, rs_generator) for _ in
                 range(self.n_trees)]

        # Train trees
        for tree in trees:
            random_state = tree.random_state
            X_random_feature, features_idxs = self._random_features(X,
                                                                    random_state)
            if cat_features:
                new_cat_features = self._map_cat_features(features_idxs,
                                                          cat_features)
            else:
                new_cat_features = None
            tree.fit(X_random_feature, y, cat_features=new_cat_features)
            self.feature_subsets.append(features_idxs)
            self.trained_trees.append(tree)
            feature_importance = tree.feature_importance
            feature_importance_reshape = self._mapping_feature_importance(
                features_idxs, feature_importance, self.n_features)
            self.feature_importance.append(feature_importance_reshape)

        assert len(self.trained_trees) == self.n_trees
        self.feature_importance = np.mean(self.feature_importance, axis=0)
        assert round(self.feature_importance.sum(), 3) == 1.0

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

    def _random_features(self,
                         X: np.ndarray,
                         random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.choice(self.n_features,
                                      self.F,
                                      replace=False)
        X_selected_feature = X[:, indices]
        return X_selected_feature, indices

    @staticmethod
    def _map_cat_features(features_idxs: np.ndarray,
                          cat_features: list):
        mapping = {val: i for i, val in enumerate(features_idxs)}
        return [mapping[val] for val in cat_features if val in mapping.keys()]

    @staticmethod
    def _mapping_feature_importance(features_idxs, feature_importance,
                                    n_features):
        fi = np.zeros(n_features)
        mapping = {i: val for i, val in enumerate(features_idxs)}
        for i, value in mapping.items():
            fi[value] = feature_importance[i]
        return fi
