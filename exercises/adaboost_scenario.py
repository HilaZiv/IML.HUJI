import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    number_of_learners = np.arange(1, n_learners + 1)
    training_errors = np.array([adaboost.partial_loss(train_X, train_y, n) for n in number_of_learners])
    test_errors = np.array([adaboost.partial_loss(test_X, test_y, n) for n in number_of_learners])
    fig = go.Figure([go.Scatter(x=number_of_learners, y=training_errors, name=r"$\text{Training error}$"),
                     go.Scatter(x=number_of_learners, y=test_errors, name=r"$\text{Test error}$")],
                    layout=go.Layout(
        title=rf"$\textbf{{Training and test errors as function of numbers of fitted learners, with noise {noise}}}$",
                                     xaxis=dict(title=r"$\text{Number of fitted learners}$"),
                                     yaxis=dict(title=r"$\text{Error}$"),
                                     width=800, height=500))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[rf"$\text{{{n} fitted learners, with noise {noise}}}$" for n in T],
        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, n in enumerate(T):
        fig.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, n), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol="x", colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i//2) + 1, cols=(i % 2)+1)

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries of Adaboost for Different Number of Learners, with noise {noise}}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble
    min_test_error_index = np.argmin(test_errors)
    min_test_error = test_errors[min_test_error_index]
    go.Figure([decision_surface(lambda x: adaboost.partial_predict(x, min_test_error_index + 1), lims[0], lims[1],
                                showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', marker=dict(color=test_y, colorscale=custom),
                          showlegend=False)],
    layout=go.Layout(title=
    rf"$\textbf{{Ensemble with lowest test error: size {min_test_error_index+1}, accuracy {1-min_test_error}, noise {noise}}}$")).show()

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_[n_learners - 1]
    D = (D / D.max()) * 20
    go.Figure([decision_surface(lambda x: adaboost.partial_predict(x, n_learners), lims[0], lims[1],
                                showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                          mode='markers', marker=dict(color=train_y, colorscale=custom, size=D), showlegend=False)],
              layout=go.Layout(title=
    rf"$\textbf{{Training set with size indicating weights of last iteration, with noise {noise}}}$")).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, n_learners=250)
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250)
