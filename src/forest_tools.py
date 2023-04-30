"""
This module provides functions and classes for creating and manipulating
random forest and decision forests models.
"""
import copy
from typing import Optional
from enum import Enum
import numpy as np

from src.decision_tree import DecisionTree


def check_types(random_state: int,
                n_trees: int,
                max_depth: int,
                min_samples_split: int,
                base_learner: DecisionTree) -> None:
    """
    Checks the types of the input arguments.
    """

    if not isinstance(random_state, int):
        raise TypeError('random_state must be an integer')
    if not isinstance(n_trees, int):
        raise TypeError('n_trees must be an integer')
    if not isinstance(max_depth, Optional[int]):
        raise TypeError('max_depth must be an integer or None')
    if not isinstance(min_samples_split, int):
        raise TypeError('min_samples_split must be an integer')
    if not isinstance(base_learner, DecisionTree):
        raise TypeError('base_learner must be an instance of DecisionTree')


def make_one_tree(classifier: DecisionTree,
                  rs_generator: np.random.RandomState) -> DecisionTree:
    """
    Creates a copy of a decision tree classifier, sets its random seed,
    and returns the modified copy.
    """

    # copy the classifier algorithm (DecisionTree)
    tree = copy.deepcopy(classifier)

    # get random state
    random_state = rs_generator.randint(np.iinfo(np.int32).max)
    to_set = {'random_state': random_state}

    # set random state
    tree.set_params(**to_set)
    return tree


class FMethodRF(Enum):
    """
    An enumeration of available methods for determining the number of random
    features used in the splitting of the nodes in a random forest model.

    Constants:
    ---------
    SQRT : str
        The square root method.
    LOG : str
        The logarithm method.
    """

    SQRT = 'sqrt'
    LOG = 'log'


class FMethodDF(Enum):
    """
    An enumeration of available methods for determining the number of random
    features used in each tree in a decision forest model.

    Constants:
    ---------
    RUNIF : str
        The random uniform method.
    """

    RUNIF = 'runif'
