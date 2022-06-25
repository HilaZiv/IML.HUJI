import numpy
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import plotly.io

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go
import matplotlib.pyplot as plt



def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])
    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_lst = []

    def _callback(solver, weights, val, grad, t, eta, delta):
        values.append(val)
        weights_lst.append(weights)

    return _callback, values, weights_lst


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # Q1
    l1_output = np.empty((len(etas), init.shape[0]))
    l2_output = np.empty((len(etas), init.shape[0]))
    modules_weights = [[], []]
    modules_values = [[], []]
    modules = [L1, L2]
    modules_names = ["L1", "L2"]

    for i in range(len(etas)):
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=FixedLR(etas[i]), callback=callback_l1)
        l1_output[i] = gd_l1.fit(L1(init), None, None)

        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        gd_l2 = GradientDescent(learning_rate=FixedLR(etas[i]), callback=callback_l2)
        l2_output[i] = gd_l2.fit(L2(init), None, None)

        modules_weights[0].append(weights_l1)
        modules_weights[1].append(weights_l2)
        modules_values[0].append(values_l1)
        modules_values[1].append(values_l2)

    for m in range(len(modules)):
        for i in range(len(etas)):
            fig = plot_descent_path(modules[m], numpy.array(modules_weights[m][i]),
                                    f"of {modules_names[m]} for rate {etas[i]}")
            if i == 2:
                plotly.io.write_image(fig, fr"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex6\GD_path_of_"
                                           fr"{modules_names[m]}.png")
            fig.show()

    # Q3+Q4
    for m in range(len(modules)):
        for i in range(len(etas)):
            iterations = np.arange(1, len(modules_values[m][i]) + 1)
            plt.scatter(iterations, np.array(modules_values[m][i]), s=2,  label=f"rate {etas[i]}")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.legend()
            plt.title(f"{modules_names[m]} GD convergence rate for different fixed rates")
            plt.ticklabel_format(style='plain')
            print(f"{modules_names[m]} minimum loss for rate {etas[i]}: ", np.array(modules_values[m][i]).min())
        plt.savefig(fr"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex6\{modules_names[m]}_loss_per_iteration")
        plt.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    l1_output = np.empty((len(gammas), init.shape[0]))
    l2_output = np.empty((len(gammas), init.shape[0]))
    modules_weights = [[], []]
    modules_values = [[], []]
    modules = [L1, L2]
    modules_names = ["L1", "L2"]

    for i in range(len(gammas)):
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=ExponentialLR(eta, gammas[i]), callback=callback_l1)
        l1_output[i] = gd_l1.fit(L1(init), None, None)

        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        gd_l2 = GradientDescent(learning_rate=ExponentialLR(eta, gammas[i]), callback=callback_l2)
        l2_output[i] = gd_l2.fit(L2(init), None, None)
        modules_weights[0].append(weights_l1)
        modules_weights[1].append(weights_l2)
        modules_values[0].append(values_l1)
        modules_values[1].append(values_l2)

    # Plot algorithm's convergence for the different values of gamma
    for m in range(len(modules)):
        for i in range(len(gammas)):
            iterations = np.arange(1, len(modules_values[m][i]) + 1)
            plt.plot(iterations, np.array(modules_values[m][i]), label=f"gamma {gammas[i]}")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.legend()
            plt.title(f"{modules_names[m]} GD convergence rate for rate {eta} and different exponential decays")
            plt.ticklabel_format(style='plain')
            print(f"{modules_names[m]} minimum loss for gamma {gammas[i]}: ", np.array(modules_values[m][i]).min())
        plt.savefig(fr"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex6\{modules_names[m]}_exponential_loss_per_iteration")
        plt.show()

    # Plot descent path for gamma=0.95
    for m in range(len(modules)):
        fig = plot_descent_path(modules[m], numpy.array(modules_weights[m][1]), f"of {modules_names[m]} for gamma 0.95")
        plotly.io.write_image(fig, fr"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex6\exponential_GD_path_of_"
                                   fr"{modules_names[m]}.png")
        fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regression = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000)).fit(X_train, y_train)
    y_prob = logistic_regression.predict_proba(X_train)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    ROC = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    plotly.io.write_image(ROC, fr"C:\Users\Hila Ziv\OneDrive\Documents\university\year 3\IML\ex6\ROC.png")
    ROC.show()
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print("best alpha: ", best_alpha)
    lr_with_best_alpha = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
                                            alpha=best_alpha).fit(X_train, y_train)
    print("best alpha test error: ", lr_with_best_alpha.loss(X_test, y_test))


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lams = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    len_lams = len(lams)
    modules = ["l1", "l2"]
    for m in modules:
        train_errors = np.empty(len_lams)
        validation_errors = np.empty(len_lams)
        from IMLearn.metrics import misclassification_error
        for i in range(len_lams):
            train_errors[i], validation_errors[i] = cross_validate(LogisticRegression(solver=GradientDescent(FixedLR(1e-4),
                                                                                                             max_iter=20000),
                                                                                      penalty=m, lam=lams[i]), X_train,
                                                                   y_train, misclassification_error)
        best_lam = lams[validation_errors.argmin()]
        test_error = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty=m, lam=best_lam)\
            .fit(X_train, y_train).loss(X_test, y_test)
        print(f"best lam for {m}: ", best_lam)
        print(f"best lam test error for {m}: ", test_error)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
