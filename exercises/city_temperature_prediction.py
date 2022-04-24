
import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    city_temp_data = pd.read_csv(filename, parse_dates=['Date'])
    df = city_temp_data[(city_temp_data.Year > 0) & (city_temp_data.Month > 0) & (city_temp_data.Month > 0) &
                        (city_temp_data.Temp < 60) & (city_temp_data.Temp > -60)].reset_index(drop=True)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df.dropna().reset_index(drop=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temperatures = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_temperatures = city_temperatures.loc[city_temperatures['Country'] == 'Israel'].reset_index(drop=True)
    px.scatter(israel_temperatures, title="Average daily temperature in Israel as a function of day of year",
               x="DayOfYear", y="Temp", color=israel_temperatures["Year"].astype(str)).show()

    israel_temp_grouped_by_month_according_temp_std = israel_temperatures.groupby('Month')['Temp'].std()
    px.bar(israel_temp_grouped_by_month_according_temp_std, title="Std of daily temperature as a function of the month",
           x=israel_temp_grouped_by_month_according_temp_std.index, y=israel_temp_grouped_by_month_according_temp_std,
           labels=dict(x="Month", y="std")).show()

    # Question 3 - Exploring differences between countries
    city_temp_grouped_by_country_and_month = \
        city_temperatures.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    px.line(title="Average daily temp as a function of the month, with std error bars",
            x=city_temp_grouped_by_country_and_month['Month'], y=city_temp_grouped_by_country_and_month[('Temp', 'mean')],
            color=city_temp_grouped_by_country_and_month['Country'], error_y=city_temp_grouped_by_country_and_month[('Temp', 'std')],
            labels=dict(x="Month", y="average daily temp")).show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = \
        split_train_test(israel_temperatures['DayOfYear'], israel_temperatures['Temp'])
    loss_per_k = np.empty(10)
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X, train_y)
        loss_per_k[k-1] = np.round(poly_fit.loss(test_X, test_y), 2)
    print(loss_per_k)
    px.bar(title="Loss (test error) as a function of k (degree of polynom)",
           x=np.arange(1, 11), y=loss_per_k, labels=dict(x="k", y="loss")).show()

    # Question 5 - Evaluating fitted model on different countries
    poly_fit_israel = PolynomialFitting(5)
    poly_fit_israel.fit(israel_temperatures['DayOfYear'], israel_temperatures['Temp'])
    south_africa_temperatures = city_temperatures.loc[city_temperatures['Country'] == 'South Africa'].reset_index(drop=True)
    netherlands_temperatures = city_temperatures.loc[city_temperatures['Country'] == 'The Netherlands'].reset_index(drop=True)
    jordan_temperatures = city_temperatures.loc[city_temperatures['Country'] == 'Jordan'].reset_index(drop=True)
    south_africa_error = poly_fit_israel.loss(south_africa_temperatures['DayOfYear'], south_africa_temperatures['Temp'])
    netherlands_error = poly_fit_israel.loss(netherlands_temperatures['DayOfYear'], netherlands_temperatures['Temp'])
    jordan_error = poly_fit_israel.loss(jordan_temperatures['DayOfYear'], jordan_temperatures['Temp'])

    countries_errors_df = pd.DataFrame({'Country': ['South Africa', 'The Netherlands', 'Jordan'],
                                        'Error': [south_africa_error, netherlands_error, jordan_error]})
    px.bar(title="Israel fitted model's error over other countries",
           x=countries_errors_df['Country'], y=countries_errors_df['Error'],
           labels=dict(x="countries", y="error")).show()
