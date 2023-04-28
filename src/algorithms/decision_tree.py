"""
Implementation of the Decision Tree algorithm using CART method.
"""
from __future__ import annotations
from typing import List, Optional, Union
import operator
from enum import Enum
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Operator(Enum):
    LE = operator.le
    GT = operator.gt
    EQ = operator.eq
    NE = operator.ne


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    gini: Optional[float] = None
    n_samples: Optional[int] = None
    class_dist: Optional[np.ndarray] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    leaf_value: Optional[int] = None


class DecisionTree:

    def __init__(self,
                 random_state=0,
                 criterion='gini',
                 splitter='best',
                 max_depth=2,
                 min_samples_split=2):
        self.random_state: int = random_state
        self.criterion: str = criterion
        self.splitter: str = splitter
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split

        self.n_classes = None
        self.tree = None

    @property
    def method(self):
        return 'CART'

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        n_side = len(y)
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum([np.square(c / n_side) for c in counts])

    @staticmethod
    def _set_operators(feature_idx: int, cat_features: Optional[list]) -> \
            tuple[Operator, Operator]:
        op_1 = Operator.LE
        op_2 = Operator.GT
        if cat_features:
            if feature_idx in cat_features:
                op_1 = Operator.EQ
                op_2 = Operator.NE
        return op_1, op_2

    def fit(self, X, y, cat_features: Optional[list] = None):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, 0, cat_features)

    def predict(self, X: np.ndarray):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x: np.ndarray, node: Node) -> float:
        if node.leaf_value is not None:
            return node.leaf_value
        if x[node.feature] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

    def predict_proba(self, X):
        pass

    def set_params(self, **params):
        pass

    def _build_tree(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    depth: int = 0,
                    cat_features: Optional[list] = None):

        n_samples, n_features = X.shape

        # Stopping criteria: if the tree has reached its maximum depth
        # or if there are too few samples to split
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(feature=None,
                        threshold=None,
                        gini=self._gini(y),
                        n_samples=n_samples,
                        class_dist=self._compute_class_dist(y),
                        left=None,
                        right=None,
                        leaf_value=self._compute_leaf_value(y))

        # Find the best feature and threshold to split on
        best_gini_gain, best_feature_idx, best_threshold = self._best_split(X,
                                                                            y,
                                                                            cat_features)

        # If best_feature_idx is None and best_threshold is None
        # return a leaf node
        if best_feature_idx is None and best_threshold is None:
            return Node(feature=None,
                        threshold=None,
                        gini=self._gini(y),
                        n_samples=n_samples,
                        class_dist=self._compute_class_dist(y),
                        left=None,
                        right=None,
                        leaf_value=self._compute_leaf_value(y))

        # Set operator if is continuous or categorical data
        op_1, op_2 = self._set_operators(best_feature_idx, cat_features)
        op_1, op_2 = op_1.value, op_2.value

        # Split the dataset into two subsets
        left_indices = op_1(X[:, best_feature_idx], best_threshold)
        right_indices = op_2(X[:, best_feature_idx], best_threshold)

        # Create the left and right subtrees recursively
        left_subtree = self._build_tree(X[left_indices],
                                        y[left_indices],
                                        depth + 1,
                                        cat_features)
        right_subtree = self._build_tree(X[right_indices],
                                         y[right_indices],
                                         depth + 1,
                                         cat_features)

        node = Node(feature=best_feature_idx,
                    threshold=best_threshold,
                    gini=self._gini(y),
                    n_samples=n_samples,
                    class_dist=self._compute_class_dist(y),
                    left=left_subtree,
                    right=right_subtree)

        return node

    def _best_split(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    cat_features: Optional[list] = None) -> tuple[float,
    Optional[int], Union[Optional[float], Optional[str]]]:

        best_gini_gain = 0
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                op_1, op_2 = self._set_operators(feature_idx, cat_features)
                op_1, op_2 = op_1.value, op_2.value
                left_indices = op_1(X[:, feature_idx], threshold)
                right_indices = op_2(X[:, feature_idx], threshold)

                if np.sum(left_indices) > 0 and np.sum(right_indices) > 0:
                    gini_gain = self._gini_gain(y[left_indices],
                                                y[right_indices])
                    if gini_gain > best_gini_gain:
                        best_feature_idx = feature_idx
                        if cat_features:
                            if best_feature_idx in cat_features:
                                best_threshold = threshold
                        else:
                            best_threshold = float(threshold)
                        best_gini_gain = gini_gain

        return best_gini_gain, best_feature_idx, best_threshold

    @staticmethod
    def _compute_leaf_value(y: np.ndarray):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def _compute_class_dist(self, y: np.ndarray) -> np.ndarray:
        class_dist = np.zeros(self.n_classes)
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            class_dist[u] = c
        return class_dist

    def _gini_gain(self,
                   left_side: np.ndarray,
                   right_side: np.ndarray) -> float:

        # gini before the split
        parent_gini = self._gini(np.hstack([left_side, right_side]))

        # child gini
        n_left = len(left_side)
        n_right = len(right_side)
        n_total = n_left + n_right
        left_contribution = n_left / n_total * self._gini(left_side)
        right_contribution = n_right / n_total * self._gini(right_side)
        child_gini = left_contribution + right_contribution

        # gini gain
        gini_gain = parent_gini - child_gini

        return gini_gain

#
#
# @dataclass
# class DecisionTree:
#     criterion: TreeMethods = field(default=TreeMethods.CART)
#     max_depth: int = field(default=2)
#     min_samples_split: int = field(default=2)
#     root = None
#
#     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
#         self.root = self._build_tree(X, y)
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         return np.array([self._predict_tree(x, self.root) for x in X])
#
#     def _build_tree(self, X, y, depth=0):
#         n_samples, n_features = X.shape
#
#         # Stopping criteria: if the tree has reached its maximum depth or if there are too few samples to split
#         if depth >= self.max_depth or n_samples < self.min_samples_split:
#             return Node(value=self._most_common_label(y))
#
#         # Find the best feature and threshold to split on
#         best_feature, best_threshold = self._best_split(X, y, n_samples,
#                                                         n_features)
#
#         # Split the dataset into two subsets
#         left_indices = X[:, best_feature] <= best_threshold
#         right_indices = X[:, best_feature] > best_threshold
#
#         # Create the left and right subtrees recursively
#         left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
#         right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
#
#         # Create a new node with the best feature and threshold values and return it
#         return Node(best_feature, best_threshold, left, right)
#
#     def _best_split(self, X, y, n_samples, n_features):
#         best_feature = None
#         best_threshold = None
#         best_gini = 1
#
#         # Calculate the Gini impurity for each feature and threshold
#         for feature_idx in range(n_features):
#             thresholds = np.unique(X[:, feature_idx])
#             for threshold in thresholds:
#                 left_indices = X[:, feature_idx] <= threshold
#                 right_indices = X[:, feature_idx] > threshold
#
#                 if np.sum(left_indices) > 0 and np.sum(right_indices) > 0:
#                     gini = self._gini(y[left_indices], y[right_indices])
#                     if gini < best_gini:
#                         best_feature = feature_idx
#                         best_threshold = threshold
#                         best_gini = gini
#
#         return best_feature, best_threshold
#
#     def _gini(self, left_labels, right_labels):
#         n_left = len(left_labels)
#         n_right = len(right_labels)
#         n_total = n_left + n_right
#         gini_left = 1 - np.sum(
#             [(np.sum(left_labels == c) / n_left) ** 2 for c in
#              np.unique(left_labels)])
#         gini_right = 1 - np.sum(
#             [(np.sum(right_labels == c) / n_right) ** 2 for c in
#              np.unique(right_labels)])
#         return (n_left / n_total) * gini_left + (
#                     n_right / n_total) * gini_right
#
#     def _predict_tree(self, x: np.ndarray, node: Node) -> float:
#         if node.value is not None:
#             return node.value
#         if x[node.feature] < node.threshold:
#             return self._predict_tree(x, node.left)
#         else:
#             return self._predict_tree(x, node.right)
#
#     def _most_common_label(self, y):
#         """
#         Find the most common label in the array y.
#
#         Parameters
#         ----------
#         y : array-like, shape (n_samples,)
#             The array of labels.
#
#         Returns
#         -------
#         The most common label in y.
#         """
#         return Counter(y).most_common(1)[0][0]
