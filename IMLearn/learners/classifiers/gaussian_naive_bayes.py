from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_k_array = np.bincount(y)
        samples_num = X.shape[0]
        self.pi_ = n_k_array / samples_num
        y_i_equals_k_indicator = np.zeros((y.shape[0], n_k_array.shape[0]))  # creates indicators matrix
        y_i_equals_k_indicator[np.arange(y.shape[0]), y] = 1
        self._y_i_equals_k_indicator = y_i_equals_k_indicator
        self.mu_ = y_i_equals_k_indicator.T @ X  # shape: n_classes x n_features
        self.mu_ = self.mu_ * np.reciprocal(np.repeat(
            n_k_array, self.mu_.shape[1]).reshape(self.mu_.shape), dtype=float)
        self.vars_ = np.zeros(self.mu_.shape)
        for k in range(self.vars_.shape[0]):
            self.vars_[k] = X[y == self.classes_[k]].var(axis=0, ddof=1)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((len(X), len(self.mu_)))
        for i in range(len(X)):
            for k in range(len(self.mu_)):
                likelihood[i][k] = self.pi_[k] * (1 / (np.sqrt((2 * np.pi * np.prod(self.vars_[k]))**X.shape[1]))) * \
                                   np.exp(-0.5 * (X[i] - self.mu_[k]).T @ np.diag(np.reciprocal(self.vars_[k])) @
                                          (X[i] - self.mu_[k]))
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
