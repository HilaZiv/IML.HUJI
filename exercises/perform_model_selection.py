from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)

    def _f(x):
        return (x+3) * (x+2) * (x+1) * (x-1) * (x-2)
    y = _f(X)
    y_with_noise = y + np.random.normal(0, noise, n_samples)
    X = pd.DataFrame(X)
    y_with_noise = pd.Series(y_with_noise)
    train_X, train_y, test_X, test_y = split_train_test(X, y_with_noise, train_proportion=2/3)

    plt.scatter(X, y, c="black", label="true labels")
    plt.scatter(train_X, train_y, label="train labels, noise " + str(noise))
    plt.scatter(test_X, test_y, label="test labels noise " + str(noise))
    plt.xlabel("samples - x")
    plt.ylabel("labels - f(x)")
    plt.legend()
    plt.title("f(x) as a function of x\nnoise: " + str(noise) + ", samples: " + str(n_samples))
    plt.savefig(r"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex5\f(x)_noise"
                + str(noise) + "_samples" + str(n_samples))
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 11
    degrees = np.arange(max_degree)
    train_errors = np.empty(max_degree)
    validation_errors = np.empty(max_degree)

    train_X = train_X[0].to_numpy()
    train_y = train_y.to_numpy()
    test_X = test_X[0].to_numpy()
    test_y = test_y.to_numpy()

    for k in range(max_degree):
        train_errors[k], validation_errors[k] = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error)
    plt.plot(degrees, train_errors, c="black", label="average train error")
    plt.plot(degrees, validation_errors, label="average validation error")
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.legend()
    plt.title("Train and validation average errors in 5-fold cross-validation"
              "\nas a function of the degree of fitted polynom\nnoise: " + str(noise) + ", samples: " + str(n_samples))
    plt.ticklabel_format(style='plain')
    plt.savefig(r"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex5\average_error_noise"
                + str(noise) + "_samples" + str(n_samples))
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = validation_errors.argmin()
    best_k_test_error = np.round(PolynomialFitting(best_k).fit(train_X, train_y).loss(test_X, test_y), 2)
    print("besk k for noise ", noise, ": ", best_k)
    print("besk k validation error for noise ", noise, ": ", np.round(validation_errors[best_k], 2))
    print("besk k test error for noise ", noise, ": ", best_k_test_error)


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_samples, shuffle=False)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lams = np.linspace(0, n_evaluations, 100)
    ridge_cv_train_error = np.empty(len(lams))
    ridge_cv_validation_error = np.empty(len(lams))
    lasso_cv_train_error = np.empty(len(lams))
    lasso_cv_validation_error = np.empty(len(lams))
    ridge_cv_train_error[0], ridge_cv_validation_error[0] = cross_validate(RidgeRegression(0), X_train,
                                                                           y_train, mean_square_error, 5)
    lasso_cv_train_error[0], lasso_cv_validation_error[0] = cross_validate(LinearRegression(), X_train, y_train,
                                                                           mean_square_error, 5)
    for i in range(1, len(lams)):
        ridge_cv_train_error[i], ridge_cv_validation_error[i] = cross_validate(RidgeRegression(lams[i]), X_train,
                                                                               y_train, mean_square_error, 5)
        lasso_cv_train_error[i], lasso_cv_validation_error[i] = cross_validate(Lasso(alpha=lams[i]), X_train, y_train,
                                                                               mean_square_error, 5)
    plt.plot(lams, ridge_cv_train_error, label="ridge train error")
    plt.plot(lams, lasso_cv_train_error, label="lasso train error")
    plt.plot(lams, ridge_cv_validation_error, label="ridge validation error")
    plt.plot(lams, lasso_cv_validation_error, label="lasso validation error")
    plt.xlabel("lambda")
    plt.ylabel("error")
    plt.legend()
    plt.title("Ridge and Lasso train and validation average errors"
              "\nin 5-fold cross-validation"
              "\nas a function of the regularization parameter")
    plt.ticklabel_format(style='plain')
    plt.savefig(r"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex5\ridge_lasso_average_error")
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lams[ridge_cv_validation_error.argmin()]
    best_lam_lasso = lams[lasso_cv_validation_error.argmin()]
    print("best_lam_ridge = ", best_lam_ridge)
    print("best_lam_lasso = ", best_lam_lasso)

    ridge_test_error = np.round(RidgeRegression(best_lam_ridge).fit(X_train, y_train).loss(X_test, y_test), 2)
    lasso_pred = Lasso(best_lam_lasso).fit(X_train, y_train).predict(X_test)
    lasso_test_error = np.round(mean_square_error(y_test, lasso_pred), 2)
    linear_regression_test_error = np.round(LinearRegression().fit(X_train, y_train).loss(X_test, y_test), 2)
    print("ridge_test_error = ", ridge_test_error)
    print("lasso_test_error = ", lasso_test_error)
    print("linear_regression_test_error = ", linear_regression_test_error)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_polynomial_degree(n_samples=1500, noise=0)
    select_regularization_parameter(n_evaluations=3)
