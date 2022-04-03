from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_samples = np.random.normal(10, 1, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(univariate_samples)
    print("(" + str(univariate_gaussian.mu_) + ", " + str(univariate_gaussian.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    increasing_sample_size_expectations = []
    for i in range(10, 1001, 10):
        increasing_sample_size_expectations = np.append(increasing_sample_size_expectations,
                                                        univariate_gaussian.fit(univariate_samples[0:i]).mu_)
    distance_from_true_expectation_plot = \
        px.line(x=np.arange(10, 1001, 10), y=np.abs(increasing_sample_size_expectations - 10),
                title="Distance between estimated and true value of expectation as a function of sample size"
                      " for N(10,1) distribution",
                labels=dict(x="Sample size", y="Distance between estimated and true value of expectation"), markers=True)
    distance_from_true_expectation_plot.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_of_samples_plot = px.scatter(x=univariate_samples, y=univariate_gaussian.pdf(univariate_samples),
                                     title="PDF as a function of sample value for N(10,1) distribution",
                                     labels=dict(x="Sample value", y="PDF"))
    pdf_of_samples_plot.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu_question_4 = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    multivariate_samples = np.random.multivariate_normal(mu_question_4, cov, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(multivariate_samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1_f3_values = np.linspace(-10, 10, 200)
    cartesian_product_of_f1_f3_values = np.transpose([np.repeat(f1_f3_values, 200), np.tile(f1_f3_values, 200)])
    expectations = np.apply_along_axis(lambda row: [row[0], 0, row[1], 0], 1, cartesian_product_of_f1_f3_values)
    log_likelihoods = np.apply_along_axis(MultivariateGaussian.log_likelihood, 1, expectations, cov,
                                          multivariate_samples)
    go.Figure(go.Heatmap(x=np.repeat(f1_f3_values, 200), y=np.tile(f1_f3_values, 200), z=log_likelihoods,
                         colorbar=dict(title='log-likelihood')),
              layout=go.Layout(title="Heatmap of log-likelihoods according to f1 and f3 values",
                               xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='f1 values')),
                               yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='f3 values')))).show()

    # Question 6 - Maximum likelihood
    print(np.round(cartesian_product_of_f1_f3_values[np.argmax(log_likelihoods)], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()



