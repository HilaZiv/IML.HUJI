from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y_true = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def calculate_loss(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, y_true))

        Perceptron(callback=calculate_loss).fit(X, y_true)

        # Plot figure of loss as function of fitting iteration
        loss_as_function_of_fitting_iteration_plot = \
            px.line(x=np.arange(1, len(losses) + 1), y=losses,
                    title="Loss as a function of fitting iteration using " + n + " data",
                    labels=dict(x="Iteration", y="Loss"), markers=True)
        loss_as_function_of_fitting_iteration_plot.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda_pred = lda.fit(X, y).predict(X)
        gnb_pred = gnb.fit(X, y).predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        models = [lda, gnb]
        predictions = [lda_pred, gnb_pred]
        subplots_names = ["LDA prediction", "GNB prediction"]
        fig = make_subplots(rows=1, cols=2,
        subplot_titles=[rf"$\textbf{{{subplots_names[i]} with accuracy {accuracy(y, predictions[i])}}}$" for i in range(2)],
        horizontal_spacing=0.01, vertical_spacing=.03)

        # Add traces for data-points setting symbols and colors
        for i, p in enumerate(predictions):
            fig.add_traces([go.Scatter(x=X.T[0], y=X.T[1], mode="markers",
                                       marker=dict(color=predictions[i], symbol=y), showlegend=False)], rows=1, cols=i+1)

        # Add `X` dots specifying fitted Gaussians' means
        for i, m in enumerate(models):
            for k in lda.classes_:
                fig.add_trace(row=1, col=i+1,
                              trace=go.Scatter(x=[m.mu_[k][0]], y=[m.mu_[k][1]], mode="markers",
                                               marker=go.scatter.Marker(color="black", symbol=4, size=15)))

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(len(lda.classes_)):
            # LDA cov
            fig.add_trace(row=1, col=1, trace=get_ellipse(lda.mu_[k], lda.cov_))

            # GNB cov
            fig.add_trace(row=1, col=2, trace=get_ellipse(gnb.mu_[k], np.diag(gnb.vars_[k])))

        fig.update_layout(
            title=rf"$\textbf{{LDA and GNB predictions over %s}}$" % f,
            margin=dict(t=80)).update_xaxes(visible=True).update_yaxes(visible=True).show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
