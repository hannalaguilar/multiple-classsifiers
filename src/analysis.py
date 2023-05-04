"""
This module contains functions to plot and analyze the results related
to random forests and decision forests.
"""
from typing import Optional
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CURRENT_PATH = Path(__file__).parent.parent


def plot_accuracy_vs_n_trees(df: pd.DataFrame,
                             title: Optional[str] = None):
    """
    Plot test accuracy vs. the number of trees for random forest
    and decision forest models results.

    """

    sel_col = ['algorithm', 'n_trees', 'test_acc']
    df_groupby = df[sel_col].groupby(['n_trees', 'algorithm']). \
        agg(['mean', 'std']).reset_index()
    random_forest_df = df_groupby[df_groupby.algorithm == 'random forest']
    decision_forest_df = df_groupby[df_groupby.algorithm == 'decision forest']

    fig = plt.figure()
    plt.title(title)
    plt.errorbar(x=random_forest_df.n_trees,
                 y=random_forest_df['test_acc']['mean'],
                 yerr=random_forest_df['test_acc']['std'], fmt='o',
                 capsize=5, alpha=.85, color='peru',
                 markersize=10, label='Random Forest')
    plt.errorbar(x=decision_forest_df.n_trees,
                 y=decision_forest_df['test_acc']['mean'],
                 yerr=decision_forest_df['test_acc']['std'], fmt='o',
                 capsize=5, alpha=.85,
                 color='seagreen',
                 markersize=10, label='Decision Forest')
    plt.ylim(0.5, 1.0)
    plt.ylabel('Test accuracy (%)')
    plt.xlabel('Number of trees (NT)')
    plt.legend()

    return fig


def plot_F_vs_test_accuracy(df: pd.DataFrame,
                            title: Optional[str] = None):
    """
    Plots test accuracy vs. the number of random features (F) for
    random forest and decision forest models results.

    """

    sel_col = ['algorithm', 'F', 'test_acc']
    df_groupby = df[sel_col].groupby(['algorithm', 'F']). \
        agg(['mean', 'std']).reset_index()
    random_forest_df = df_groupby[df_groupby.algorithm == 'random forest']
    decision_forest_df = df_groupby[df_groupby.algorithm == 'decision forest']
    x_rf = [el if len(el) <= 4 else el.split('.')[1] for el in
            random_forest_df.F]
    x_df = [el if len(el) <= 4 else el.split('.')[1] for el in
            decision_forest_df.F]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle(title)
    axs = axs.flatten()
    axs[0].errorbar(x=x_rf,
                    y=random_forest_df['test_acc']['mean'],
                    yerr=random_forest_df['test_acc']['std'], fmt='s',
                    capsize=5, alpha=.85, color='peru',
                    markersize=10, label='Random Forest')
    axs[0].set_ylim(0.5, 1.0)
    axs[0].set_ylabel('Test accuracy (%)')
    axs[0].set_xlabel('F (Number of random features)')
    axs[0].legend()
    axs[1].errorbar(x=x_df,
                    y=decision_forest_df['test_acc']['mean'],
                    yerr=decision_forest_df['test_acc']['std'], fmt='s',
                    capsize=5, alpha=.85,
                    color='seagreen',
                    markersize=10, label='Decision Forest')
    axs[1].set_ylim(0.5, 1.0)
    axs[1].set_xlabel('F (Number of random features)')
    axs[1].legend()

    return fig


def feature_importance_mean(df: pd.DataFrame, columns_name: list) -> dict:
    """
    Calculates the mean feature importance across all decision trees
    in a random forest or decision forest model results.

    """

    n = df.shape[0]
    d = {key: 0 for key in columns_name}

    for row in df.feature_importance:
        for elem in ast.literal_eval(row):
            d[elem[0]] += elem[1]
    d = {key: value / n for key, value in d.items()}
    assert (np.array(list(d.values())).sum().round(3)) == 1.0
    return d


def plot_feature_importance(df: pd.DataFrame,
                            columns_name: list,
                            title: Optional[str] = None):
    """
    Plots the mean feature importance for a random forest and decision forest
    models results.

    """

    df1 = df[df.algorithm == 'random forest']
    df2 = df[df.algorithm == 'decision forest']

    d1 = feature_importance_mean(df1, columns_name)
    d2 = feature_importance_mean(df2, columns_name)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(title)
    axs = axs.flatten()
    axs[0].bar(x=list(d1.keys()), height=list(d1.values()),
               color=df1.shape[0] * ['lavender'], edgecolor='k', alpha=0.7,
               label='Random forest')
    axs[0].set_xlabel('Features')
    axs[0].set_ylabel('Feature importance (normalize)')
    axs[0].set_xticklabels(list(d1.keys()), rotation=90)
    axs[0].legend()

    axs[1].bar(x=list(d2.keys()), height=list(d2.values()),
               color=df2.shape[0] * ['papayawhip'], edgecolor='k', alpha=0.7,
               label='Decision forest')
    axs[1].set_xlabel('Features')
    axs[1].set_xticklabels(list(d2.keys()), rotation=90)
    axs[1].legend()
    plt.tight_layout()

    return fig


def main():
    """
    Runs the main script for generating and saving visualizations and
    statistics for a set of results datasets.

    """

    names = ['titanic', 'iris', 'glass', 'wine-red', 'wine-white']
    for name in names:
        path_results = CURRENT_PATH / 'data/processed'
        path_raw = CURRENT_PATH / 'data/raw'
        df_raw = pd.read_csv(path_raw / f'{name}.csv', index_col=0)
        df_results = pd.read_csv(path_results / f'{name}_results.csv',
                                 index_col=0)
        columns_name = df_raw.columns[:-1].tolist()
        stats = f'{name}, train_acc: ' \
                f'{df_results.train_acc.mean():.3f}+-{df_results.train_acc.std():.3f}, ' \
                f'test_acc:{df_results.test_acc.mean():.3f} +- {df_results.test_acc.std():.3f}'
        print(stats)
        fig1 = plot_accuracy_vs_n_trees(df_results, name.capitalize())
        fig2 = plot_F_vs_test_accuracy(df_results, name.capitalize())
        fig3 = plot_feature_importance(df_results, columns_name,
                                       name.capitalize())

        figures = [fig1, fig2, fig3]
        for i, fig in enumerate(figures):
            fig.savefig(CURRENT_PATH / f'docs/figures/{name}_{i + 1}.jpg')


if __name__ == "__main__":
    main()
