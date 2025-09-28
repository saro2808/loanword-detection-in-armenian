from typing import Tuple, TypeAlias, Protocol, TypeVar, Callable

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

import optuna


T = TypeVar("T", covariant=True)

class SupportsArrayIndexing(Protocol[T]):
    def __getitem__(self, key: np.ndarray) -> T: ...
    
ArrayLike = SupportsArrayIndexing
TrainTestSplit: TypeAlias = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
TrainValTestSplit: TypeAlias = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]


def subarray(arr: ArrayLike, idx: np.ndarray) -> ArrayLike:
    return arr.iloc[idx, :] if isinstance(arr, pd.DataFrame) else arr[idx]


def multi_label_train_test_split(
    X: ArrayLike,
    Y: ArrayLike,
    test_size: float = 0.2,
    random_state: int = 42
) -> TrainTestSplit:
    """
    Splits multi-label data into train and test sets using stratified sampling.

    This function ensures that the distribution of multiple labels is preserved
    in both the training and test sets using MultilabelStratifiedShuffleSplit.

    Args:
        X (ArrayLike): Feature matrix.
        Y (ArrayLike): Multi-label target matrix.
        test_size (float, optional): Proportion of test part. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: X_train, X_test, Y_train, Y_test
    
    Raises:
        TypeError: If X or Y is not a supported type.
        ValueError: If X and Y have different numbers of samples.
    """
    
    try:
        if not isinstance(X, pd.DataFrame):
            _ = X[np.arange(1)]
        if not isinstance(Y, (pd.DataFrame, pd.Series)):
            _ = Y[np.arange(1)]
    except Exception as e:
        raise TypeError("X and Y must support NumPy-style array indexing (e.g., DataFrame or ndarray).") from e
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples. Got {len(X)} and {len(Y)}.")
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    for train_idx, test_idx in msss.split(X, Y):
        X_train, X_test = subarray(X, train_idx), subarray(X, test_idx)
        Y_train, Y_test = subarray(Y, train_idx), subarray(Y, test_idx)
    
    return X_train, X_test, Y_train, Y_test


def multi_label_train_val_test_split(
    X: ArrayLike,
    Y: ArrayLike,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> TrainValTestSplit:
    """
    Splits multi-label data into train, val and test sets using stratified sampling.

    This function ensures that the distribution of multiple labels is preserved
    in all of the training, validation and test sets using MultilabelStratifiedShuffleSplit.

    Args:
        X (ArrayLike): Feature matrix.
        Y (ArrayLike): Multi-label target matrix.
        test_size (float, optional): Proportion of test part. Defaults to 0.2.
        val_size (float, optional): Proportion of val part. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    Raises:
        TypeError: If X or Y is not a supported type.
        ValueError: If X and Y have different numbers of samples.
    """

    try:
        if not isinstance(X, pd.DataFrame):
            _ = X[np.arange(1)]
        if not isinstance(Y, (pd.DataFrame, pd.Series)):
            _ = Y[np.arange(1)]
    except Exception as e:
        raise TypeError("X and Y must support NumPy-style array indexing (e.g., DataFrame or ndarray).") from e
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples. Got {len(X)} and {len(Y)}.")

    # First split train_val and test
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(msss1.split(X, Y))

    X_train_val, X_test = subarray(X, train_val_idx), subarray(X, test_idx)
    Y_train_val, Y_test = subarray(Y, train_val_idx), subarray(Y, test_idx)

    # Compute val proportion relative to train_val
    val_relative_size = val_size / (1 - test_size)

    # Second split train and val
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_relative_size, random_state=random_state
    )
    train_idx, val_idx = next(msss2.split(X_train_val, Y_train_val))

    X_train, X_val = subarray(X_train_val, train_idx), subarray(X_train_val, val_idx)
    Y_train, Y_val = subarray(Y_train_val, train_idx), subarray(Y_train_val, val_idx)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def label_distribution(Y: np.ndarray) -> np.ndarray:
    """
    Calculates class distribution of multilabel matrix Y.

    Args:
        Y (np.ndarray): Multilabel matrix.
    """
    return np.sum(Y, axis=0) / Y.shape[0]


def optimize_model(
    model_class,
    param_space_func: Callable[[optuna.trial.Trial], dict],
    X: pd.DataFrame | np.ndarray,
    Y: pd.DataFrame | np.ndarray | pd.Series,
    features: CountVectorizer | TfidfVectorizer | TruncatedSVD,
    n_trials: int = 30,
    scoring: str = "f1_micro",
    n_jobs: int = -1,
    random_state: int = 42
) -> optuna.study.Study:
    """
    General Optuna CV optimizer for OneVsRest multilabel classifiers.
    
    Args:
        model_class: Model class (e.g. RandomForestClassifier).
        param_space_func (callable): Function that tells Optuna how to sample params.
        X (ArrayLike): Feature matrix.
        Y (ArrayLike): Multilabel matrix.
        features (transformer): Feature transformer.
        n_trials (int, optional): Number of trials. Defaults to 30.
        scoring (str, optional): Metric for Optuna optimization.
        n_jobs (int, optional): Number of jobs. Defaults to -1.
        random_state (int, optional): Random state. Defaults to 42.
        
    Returns:
        optuna.study.Study: study
        
    Raises:
        TypeError: If model_class object cannot be initialized from what param_space_func returned.
                   Or if param_spec_func cannot be called from trial.
    """

    def objective(trial):
        try:
            params = param_space_func(trial)
        except Exception as e:
            raise TypeError(f"Failed to call {param_spec_func.__name__} from trial {trial}") from e

        try:
            model = model_class(random_state=random_state, **params)
        except Exception as e:
            raise TypeError(f"Failed to initialize {model_class.__name__} with random_state and params {params}") from e

        # wrap with OVR
        clf = OneVsRestClassifier(model, n_jobs=n_jobs)

        pipe = Pipeline([
            ("features", features),
            ("clf", clf)
        ])

        cv = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        scores = cross_val_score(
            pipe, X, Y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    return study