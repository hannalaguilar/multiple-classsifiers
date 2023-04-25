"""
Implementation of the Decision Tree algorithm using CART method.
"""
from __future__ import annotations
import pandas as pd

import itertools
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self,
                 random_state=0,
                 criterion='gini',
                 splitter='best',
                 max_depth=2):
        self.random_state: int = random_state
        self.criterion: str = criterion
        self.splitter: str = splitter
        self.max_depth: int = max_depth

    @property
    def method(self):
        return 'CART'

    def set_params(self, **params):
        pass

    def fit(self, X, y):
        self._build_tree(X, y)
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def _build_tree(self, X, y):
        pass

    def _best_split(self, X: pd.DataFrame,
                    y: np.ndarray) -> tuple[float, str, list[list]]:
        best_gini = 100
        best_feature = None
        best_subset = None

        for feature in X:
            feature_values = X[feature]
            split_point_subsets = self._get_subsets(feature_values)
            for split_point_subset in split_point_subsets:
                s1 = list(split_point_subset[0])
                s2 = list(split_point_subset[1])
                left_side = y[np.isin(feature_values, s1)]
                right_side = y[np.isin(feature_values, s2)]
                gini = self._gini_index(left_side, right_side)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_subset = self._subset2list(split_point_subset)

        return best_gini, best_feature, best_subset

    @staticmethod
    def _get_subsets(feature: np.ndarray) -> list[tuple[set, set]]:
        values = np.unique(feature).ravel()
        if isinstance(values, np.ndarray):
            values = list(values)
        if len(values) == 2:
            return [({values[0]}, {values[1]})]
        combs = itertools.combinations(values, 2)
        subsets = []
        for c in combs:
            c = set(c)
            subset = set(values) - c
            subsets.append((c, subset))
        return subsets

    @staticmethod
    def _subset2list(subset: tuple[set, set]) -> list[list]:
        subset = [list(s) for s in subset]
        subset = [sorted(s, key=lambda x: (len(x[0]), sorted(x[0])))
                  for s in subset]
        return subset

    def _gini_index(self,
                    left_side: np.ndarray,
                    right_side: np.ndarray) -> float:
        n_left = len(left_side)
        n_right = len(right_side)
        n_total = n_left + n_right

        # left side
        _, left_counts = np.unique(left_side, return_counts=True)
        left_contribution = self._gini_util(n_left, n_total, left_counts)

        # right side
        _, right_counts = np.unique(right_side, return_counts=True)
        right_contribution = self._gini_util(n_right, n_total, right_counts)

        # gini index
        gini = left_contribution + right_contribution

        return gini

    @staticmethod
    def _gini_util(n_side: int, n_total: int, counts: np.ndarray) -> float:
        return n_side / n_total * (1 - np.sum([np.square(c / n_side)
                                               for c in counts]))



# class DecisionTree(DecisionTreeClassifier):
#     def __init__(self, criterion='gini', splitter='best', max_depth=None,
#                  min_samples_split=2, min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.0, max_features=None,
#                  random_state=None, max_leaf_nodes=None,
#                  min_impurity_decrease=0.0,
#                  class_weight=None):
#         super().__init__(
#             criterion=criterion, splitter=splitter, max_depth=max_depth,
#             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
#             random_state=random_state, max_leaf_nodes=max_leaf_nodes,
#             min_impurity_decrease=min_impurity_decrease,
#             class_weight=class_weight)
#
#     def set_params(self, **params):
#         super().set_params(**params)
#         return self
#
#     # def fit(self, X, y):
#     #     self.tree = self._build_tree(X, y)
#
#     def _build_tree(self, X, y, depth):
#         n_samples = X.shape[0]
#         n_features = X.shape[1]
#
#         # Stopping criteria: if the tree has reached its maximum depth or if there are too few samples to split
#         if depth >= self.max_depth or n_samples < self.min_samples_split:
#             pass
#
#         # Find the best feature and threshold to split on
#         best_feature, best_threshold = self._best_split(X, y, n_features)
#
#         # Split the dataset into two subsets
#         left_indices = X[:, best_feature] <= best_threshold
#         right_indices = X[:, best_feature] > best_threshold
#
#         # Create the left and right subtrees recursively
#         left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
#         right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
#
#     def _best_split(self,  X, y, n_features):
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
#         pass
#
#     def _gini(self):
#         pass


# class TreeMethods(Enum):
#     CART = auto()
#     ID3 = auto()
#     C45 = auto()
#
#
# @dataclass
# class Node:
#     feature: Optional[int] = None
#     threshold: Optional[float] = None
#     left: Optional[Node] = None
#     right: Optional[Node] = None
#     value: Optional[float] = None
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

