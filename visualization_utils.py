from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def barplot(data: Iterable[Tuple], title: str) -> None:
    """
    Makes a barplot of data.
    
    Args:
        data (Iterable[tuple]): Iterable of tuples the first two entries of each of which are str and int.
        title (str): Title of the figure.
    Raises:
        TypeError: If data is not as described above.
    """
    
    try:
        iter(data)
    except Exception:
        raise TypeError("data should be an iterable.")
    for datum in data:
        if not isinstance(datum, tuple) or len(datum) < 2:
            raise TypeError("Each element of data should be a length two tuple.")
            
    labels = [datum[0] for datum in data]
    values = [datum[1] for datum in data]
    plt.figure(figsize=(16 * len(labels) / 35 + 1, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()


def bar_subplots(info: List[Tuple], ncols: int = 3, xlabel: str = 'Weight/importance') -> None:
    """
    Takes info list of tuples and plots len(info) barplots. Each element of info is a tuple
    whose 0th coordinate is the subplot name and whose 1th coordinate is a list of couples
    with 0th coordinates being feature names and 1th coordinates being their values.

    Args:
        info (list): List of infos to be plotted.
        ncols (int, optional): Number of columns in the subplots. Defaults to 3.
        xlabel (str, optional): Label of x axis. Defaults to 'Weight/importance'.
    
    Raises:
        TypeError: If info is not as described above.
    """

    if not isinstance(info, list):
        raise TypeError("info should be a list.")
    for item in info:
        if not isinstance(item, tuple) or len(item) < 2:
            raise TypeError("Every element of info should be a tuple of length 2 at least.")
        if not isinstance(item[1], list):
            raise TypeError("The 1st coordinate of each element of info should be a list.")
        for couple in item[1]:
            if not isinstance(couple, (tuple, list)) or len(couple) != 2:
                raise TypeError("The 1st coordinate of each element of info should be a list of couples.")

    nshow = len(info)
    nrows = int(np.ceil(nshow / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4*nrows))

    for i, ax in enumerate(axes.flat[:nshow]):
        feats, weights = zip(*info[i][1])
        ax.barh(feats[::-1], weights[::-1])
        ax.set_title(f'Top features for {info[i][0]}')
        ax.set_xlabel(xlabel)
    
    # remove empty subplots if langs < grid
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes.flat[j])
        
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    matrix: np.ndarray | pd.DataFrame,
    index: np.ndarray,
    columns: np.ndarray,
    sign_is: np.ndarray | None = None,
    sign_js: np.ndarray | None = None,
    title: str | None = None,
    annot: bool = True,
    rotate_xticks: bool = True,
    fontsize: int = 10,
) -> None:
    """
    Plots the heatmap of matrix representing token-language counts.
    
    Args:
        matrix (np.ndarray or pd.DataFrame): Matrix to plot heatmap for.
        index (np.ndarray): List representing index of matrix.
        columns (np.ndarray): List representing columns of matrix.
        sign_is (np.ndarray or None, optional): List of significant row indices to be selected. If None
                                                implies all rows. Defaults to None.
        sign_js (np.ndarray or None, optional): List of significant column indices to be selected. If None
                                                implies all columns. Defaults to None.
        title (str or None, optional): Figure title. Defaults to None.
        annot (bool, optional): If true annotate the heatmap, else no. Defaults to True.
        rotate_xticks (bool, optional): If True rotation of xticks is set to 90, else 0. Defaults to True.
        fontsize (int, optional): Diagram's text's font size. Defaults to 10.
    Raises:
        TypeError: If any variable is not of the specifed type.
    """

    type_list = [
        ('matrix', matrix, (np.ndarray, pd.DataFrame), 'np.ndarray or pd.DataFrame'),
        ('annot', annot, bool, 'bool'),
        ('rotate_xticks', rotate_xticks, bool, 'bool'),
        ('fontsize', fontsize, int, 'int')
    ]
    for varname, var, typ, typname in type_list:
        if not isinstance(var, typ):
            raise TypeError(f"{varname} should be {typname}.")
            
    for name, var in [('index', index), ('columns', columns)]:
        try:
            iter(var)
        except Exception:
            raise TypeError(f"{name} should be iterable.")
            
    for name, var in [('sign_is', sign_is), ('sign_js', sign_js)]:
        if var is not None and not isinstance(var, np.ndarray):
            raise TypeError(f"{name} should be np.ndarray.")
    if title is not None and not isinstance(title, str):
        raise TypeError("title should be str.")

    if sign_is is None:
        sign_is = np.arange(len(index))
    if sign_js is None:
        sign_js = np.arange(len(columns))
    
    # adaptive figsize
    plt.figure(figsize=(0.19 * len(sign_is) + 1, 0.19 * len(sign_js) + 1))

    ax = sns.heatmap(
        pd.DataFrame(
            matrix[np.ix_(sign_is, sign_js)],
            index=index[sign_is],
            columns=columns[sign_js]
        ).T,
        annot=annot, cmap='crest', annot_kws={"size": 6}
    )
    
    # Force all x labels to show
    ax.set_xticks(np.arange(len(sign_is)) + 0.58)
    ax.set_xticklabels(list(index[sign_is]), rotation=90 if rotate_xticks else 0, ha='right', fontsize=fontsize)
    ax.set_yticklabels(list(columns[sign_js]), rotation=0, fontsize=fontsize)
    
    plt.title(title)
    plt.show()