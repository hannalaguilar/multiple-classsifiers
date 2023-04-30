"""
This module contains functions and classes to experiment with Random Forest and
Decision Forest algorithms using different hyperparameters and datasets.

"""
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.forest_ensemble import RandomForest, DecisionForest
from src.forest_tools import FMethodRF, FMethodDF

CURRENT_PATH = Path(__file__).parent.parent

ALGORITHM_NAMES = ['random forest', 'decision forest']
N_TREES = [1, 10, 25, 50, 75, 100]
F_RANDOM_FOREST = [1, 2, FMethodRF.LOG, FMethodRF.SQRT]
F_DECISION_FOREST = [0.25, 0.5, 0.75, FMethodDF.RUNIF]


@dataclass
class Dataset:
    """
    Main class for handle datasets.

    """

    name: str
    cat_features: Optional[list[int]] = None

    def __post_init__(self):
        X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, random_state=0)

    @property
    def data_path(self):
        return CURRENT_PATH / 'data/raw' / f'{self.name}.csv'

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, index_col=0)


@dataclass
class Experiment:
    """
    Class for experimenting with Random Forest and Decision Forest algorithms.

    """

    algorithm_name: str
    n_trees: int
    max_random_features: Union[str, float, int]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    cat_features: Optional[list[int]] = None
    algorithm: Optional[Union[RandomForest, DecisionForest]] = None

    def __post_init__(self):
        if self.algorithm_name == 'random forest':
            self.algorithm = RandomForest(n_trees=self.n_trees,
                                          max_random_features=self.max_random_features)
        elif self.algorithm_name == 'decision forest':
            self.algorithm = DecisionForest(n_trees=self.n_trees,
                                            max_random_features=self.max_random_features)
        else:
            raise TypeError('algorithm must be "random forest" '
                            'or "decision forest"')

    def accuracy(self) -> float:
        self.algorithm.fit(self.X_train, self.y_train, self.cat_features)
        y_pred = self.algorithm.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)


def print_verbose(result: dict) -> None:
    """
    Prints a dictionary in a verbose format.
    """
    print("{" + ", ".join(f"'{k}': {v}" for k, v in result.items()) + "}")


def run_experiment(name: str,
                   n_trees: int,
                   max_random_features: Union[str, float, int],
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   cat_features: Optional[list[int]] = None) -> dict:
    """
    Runs an experiment using the Experiment class to test the accuracy
    of the Random Forest or Decision Forest algorithms.

    """

    clf = Experiment(name, n_trees, max_random_features,
                     X_train, y_train, X_test, y_test,
                     cat_features=cat_features)
    acc = round(clf.accuracy(), 3)
    return {'algorithm': name,
            'n_trees': n_trees,
            'F': max_random_features,
            'test_acc': acc}


def forest_interpreter(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       cat_features: Optional[list[int]] = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Runs a series of experiments with Random Forest and Decision Forest
    algorithms with different hyperparameters on the given train and test data,
    and returns a dataframe with the results.

    """

    data = []
    for name in ALGORITHM_NAMES:
        for n_trees in N_TREES:
            if name == 'random forest':
                for max_random_features in F_RANDOM_FOREST:
                    result = run_experiment(name, n_trees, max_random_features,
                                            X_train, y_train, X_test, y_test,
                                            cat_features=cat_features)
                    if verbose:
                        print_verbose(result)
                    data.append(result)
            else:
                for max_random_features in F_DECISION_FOREST:
                    result = run_experiment(name, n_trees, max_random_features,
                                            X_train, y_train, X_test, y_test,
                                            cat_features=cat_features)
                    if verbose:
                        print_verbose(result)
                    data.append(result)

    data_df = pd.DataFrame(data)
    return data_df


def main() -> None:
    """Train and test several datasets using the
    forest_interpreter function and save the results to a CSV file.
    """

    # Five datasets
    names = ['titanic', 'iris', 'glass', 'wine-red', 'wine-white']
    cat_features_datasets = [[0, 1, 6], None, None, None, None]
    datasets = [Dataset(name, cat) for name, cat in zip(names,
                                                        cat_features_datasets)]
    print(f'You will train and test the following datasets: {names}\n')

    for dataset in datasets:
        print('{:-^85}'.format(dataset.name.upper()))
        start_time = time.time()
        data_df = forest_interpreter(dataset.X_train,
                                     dataset.y_train,
                                     dataset.X_test,
                                     dataset.y_test,
                                     dataset.cat_features,
                                     verbose=True)

        path_to_save = f'data/processed/{dataset.name}_results.csv'
        data_df.to_csv(CURRENT_PATH / path_to_save)
        elapse_time = time.time() - start_time
        print(f'\n{dataset.name.capitalize()} dataset:'
              f' time: {elapse_time:.2f} s, '
              f'results are save in {path_to_save}')


if __name__ == "__main__":
    main()
