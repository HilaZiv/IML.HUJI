from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_prices_data = pd.read_csv(filename)
    df = house_prices_data[(house_prices_data.price >= 0) & (house_prices_data.sqft_lot15 >= 0)].reset_index(drop=True)
    df.dropna().reset_index(drop=True)
    df['price_per_land_space'] = df['price'] / df['sqft_lot']
    df['price_per_house_size'] = df['price'] / df['sqft_living']
    df = df.drop(columns=['id', 'date', 'zipcode'])
    return df.loc[:, df.columns != 'price'], df['price']


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X.apply(__plot_feature_response, args=(y, output_path))


def __pearson_correlation(feature: pd.Series, response: pd.Series) -> float:
    return feature.cov(response) / (feature.std() * response.std())


def __plot_feature_response(feature: pd.Series, response: pd.Series, output_path: str) -> NoReturn:
    pearson_correlation = __pearson_correlation(feature, response)
    feature_name = str(feature.name)
    plt.scatter(feature, response)
    plt.title("price as a function of " + feature_name + "\n with correlation " + str(pearson_correlation))
    plt.xlabel(feature_name)
    plt.ylabel("price")
    plt.savefig(output_path + "/" + feature_name)
    plt.clf()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression()
    mean_loss_per_p = np.empty(91)
    std_of_loss_per_p = np.empty(91)
    training_percentages = np.arange(10, 101)
    for p in range(10, 101):
        loss_values = np.empty(10)
        for i in range(10):
            sample_indexes = train_X.sample(frac=(p/100)).index
            test_indexes = train_X.drop(sample_indexes).index
            samples = train_X.drop(test_indexes)
            response = train_y.drop(test_indexes)
            linear_regression.fit(samples, response)
            loss_values[i] = linear_regression.loss(test_X, test_y)
        mean_loss_per_p[p-10] = loss_values.mean()
        std_of_loss_per_p[p-10] = loss_values.std()

    plot = (go.Scatter(x=training_percentages, y=mean_loss_per_p, mode="markers+lines", line=dict(dash="dash"),
                       marker=dict(color="green", opacity=.7)),
            go.Scatter(x=training_percentages, y=mean_loss_per_p-2*std_of_loss_per_p, fill=None, mode="lines",
                       line=dict(color="lightgrey"), showlegend=False),
            go.Scatter(x=training_percentages, y=mean_loss_per_p+2*std_of_loss_per_p, fill='tonexty', mode="lines",
                       line=dict(color="lightgrey"), showlegend=False),)
    fig = go.Figure(data=plot, layout=go.Layout(
        title="Mean loss as a function of sample percentage",
        xaxis={"title": "sample percentage"},
        yaxis={"title": "mean loss"}))
    fig.show()
