"""
Implementation of the Decision Forest algorithm
"""
import copy
from typing import Optional, Union
import numpy as np
from scipy.stats import mode

from src.decision_tree import DecisionTree


class RandomForest:
    def __init__(self,
                 random_state: int = 0,
                 n_trees: int = 100,
                 max_depth: int = 2,
                 min_samples_split: int = 2,
                 max_random_features: Union[str, int] = 'sqrt',
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
        self.trained_trees: list = []

    @property
    def F(self):
        method_functions = {'sqrt': lambda n: int(np.sqrt(n)),
                            'log': lambda n: int(np.log2(n))}
        if self.max_random_features in method_functions.keys():
            return method_functions[self.max_random_features](self.n_features)
        elif isinstance(self.max_random_features, int):
            return self.max_random_features
        else:
            raise TypeError('random_features must be an integer, '
                            '"sqrt" or "log"')

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

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
            tree.fit(X_bootstrap, y_bootstrap)
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
                 max_depth: int = 2,
                 min_samples_split: int = 2,
                 max_random_features: Union[str, float] = 0.5,
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
        self.trained_trees: list = []
        self.feature_subsets: list = []

    @property
    def F(self):
        method_functions = {'runif': lambda n: np.random.randint(1, n + 1)}
        if self.max_random_features in method_functions.keys():
            return method_functions[self.max_random_features](self.n_features)
        elif isinstance(self.max_random_features, float) and \
                self.max_random_features <= 1:
            return int(self.max_random_features * self.n_features)
        else:
            raise TypeError('max_random_features must be a value between '
                            '0 and 1 or "runif"')

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

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

    def _random_features(self,
                         X: np.ndarray,
                         random_state: int) -> tuple[np.ndarray, np.ndarray]:
        rs_generator = np.random.RandomState(random_state)
        indices = rs_generator.choice(self.n_features,
                                      self.F,
                                      replace=False)
        X_selected_feature = X[:, indices]
        return X_selected_feature, indices


def check_types(random_state,
                n_trees,
                max_depth,
                min_samples_split,
                base_learner) -> None:
    if not isinstance(random_state, int):
        raise TypeError('random_state must be an integer')
    if not isinstance(n_trees, int):
        raise TypeError('n_trees must be an integer')
    if not isinstance(max_depth, int):
        raise TypeError('max_depth must be an integer')
    if not isinstance(min_samples_split, int):
        raise TypeError('min_samples_split must be an integer')
    if not isinstance(base_learner, DecisionTree):
        raise TypeError('base_learner must be an instance of DecisionTree')


def make_one_tree(classifier: DecisionTree,
                  rs_generator: np.random.RandomState) -> DecisionTree:
    # copy the classifier algorithm (Decision Tree CART)
    tree = copy.deepcopy(classifier)

    # get random state
    random_state = rs_generator.randint(np.iinfo(np.int32).max)
    to_set = {'random_state': random_state}

    # set random state
    tree.set_params(**to_set)
    return tree
