"""
ESCRIBIR XXXXX
"""
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.ensemble_algorithms import RandomForest, DecisionForest

CURRENT_PATH = Path(__file__).parent.parent

ALGORITHM_NAMES = ['random forest', 'decision forest']
N_TREES = [1, 10, 25, 50, 75, 100]
F_RANDOM_FOREST = [1, 2, 'log', 'sqrt']
F_DECISION_FOREST = [0.25, 0.5, 0.75, 'runif']


@dataclass
class Dataset:
    """
    This is the main class for handle data.
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
    algorithm_name: str
    n_trees: int
    max_random_features: Union[str, float, int]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
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
        self.algorithm.fit(self.X_train, self.y_train)
        y_pred = self.algorithm.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)


def run_experiment(name: str,
                   n_trees: int,
                   max_random_features: Union[str, float, int],
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    clf = Experiment(name, n_trees, max_random_features,
                     X_train, y_train, X_test, y_test)
    acc = clf.accuracy()
    return {'algorithm': name,
            'n_trees': n_trees,
            'F': max_random_features,
            'test_acc': acc}


def forest_interpreter(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray) -> pd.DataFrame:
    data = []
    for name in ALGORITHM_NAMES:
        for n_trees in N_TREES:
            if name == 'random forest':
                for max_random_features in F_RANDOM_FOREST:
                    result = run_experiment(name, n_trees, max_random_features,
                                            X_train, y_train, X_test, y_test)
                    data.append(result)
            else:
                for max_random_features in F_DECISION_FOREST:
                    result = run_experiment(name, n_trees, max_random_features,
                                            X_train, y_train, X_test, y_test)
                    data.append(result)

    data_df = pd.DataFrame(data)
    return data_df


def main():
    # Three datasets
    names = ['iris', 'iris']
    datasets = [Dataset(name) for name in names]

    for dataset in datasets:
        start_time = time.time()
        data_df = forest_interpreter(dataset.X_train,
                                     dataset.y_train,
                                     dataset.X_test,
                                     dataset.y_test)

        path_to_save = f'data/processed/{dataset.name}_results.csv'
        data_df.to_csv(CURRENT_PATH / path_to_save)
        elapse_time = time.time() - start_time
        print(f'{dataset.name.capitalize()} dataset:'
              f' time: {elapse_time:.2f} s, '
              f'results are save in {path_to_save}')


if __name__ == "__main__":
    main()
