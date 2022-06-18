from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = np.array_split(X, cv)
    folds_labels = np.array_split(y, cv)
    train_scores = np.empty(cv)
    validation_scores = np.empty(cv)
    for k in range(cv):
        train = deepcopy(folds)
        test = train.pop(k)
        train = np.concatenate(train)

        train_labels = deepcopy(folds_labels)
        test_labels = train_labels.pop(k)
        train_labels = np.concatenate(train_labels)

        test_pred = estimator.fit(train, train_labels).predict(test)
        validation_scores[k] = scoring(test_labels, test_pred)

        train_pred = estimator.predict(train)
        train_scores[k] = scoring(train_labels, train_pred)
    return train_scores.mean(), validation_scores.mean()
