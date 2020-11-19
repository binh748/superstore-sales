"""This module contains helper functions for time series modeling."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.api as smt # smt stands for statsmodels time series I think
from statsmodels.stats.stattools import durbin_watson as db

# Changed this Dickey-Fuller test (DFT) function based on Metis instructor's code
def df_test(time_series):
    """Calculates Dickey-Fuller test statistics and plots rolling mean, rolling std. dev.
    and the original time series as visual checks of stationarity.

    Args:
        time_series: The time series to test and plot.

    Returns:
        None
    """
    df_test = ts.adfuller(time_series)
    df_output = pd.Series(df_test[:4],
                          index=['Test Statistic', 'p-value',
                                 '# Lags Used', '# Observations Used'])

    for key, value in df_test[4].items():
        df_output[f'Critical Value {key}'] = value
    print(df_output)

    # Calculate rolling mean and std. dev.
    roll_mean = time_series.rolling(window=12).mean()
    roll_std = time_series.rolling(window=12).std()

    # Plot rolling mean and std. dev.
    fig, ax = plt.subplots(figsize=[9, 4])
    orig = plt.plot(time_series, color='blue', label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling mean')
    std = plt.plot(roll_std, color='black', label='Rolling std. dev.')
    plt.legend(loc='best')
    plt.title('Rolling mean, rolling std. dev., and original time series')
    sns.despine()
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Order date grouped by month')
    ax.set_ylabel('Sales')
    plt.show()

# Took Metis instructor's helper plot function for ACF/PACF visualizations
# and made some minor changes
def corr_plots(time_series, lags=None):
    """Helper function to plot original time series along with ACF and PACF plots.

    Args:
        time_series: The time series to plot.

    Returns:
        None
    """
    layout = (1, 3)
    # Define the axes for each subplot
    orig = plt.subplot2grid(layout, (0, 0))
    acf = plt.subplot2grid(layout, (0, 1))
    pacf = plt.subplot2grid(layout, (0, 2))

    time_series.plot(ax=orig, figsize=(12, 4))
    orig.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    smt.graphics.plot_acf(time_series, lags=lags, ax=acf)
    smt.graphics.plot_pacf(time_series, lags=lags, ax=pacf)
    sns.despine()
    plt.tight_layout()


def plot_pred(time_series, time_series_pred,
              orig_label='Original', pred_label='Prediction',
              title='Original v. prediction'):
    """Helper function to plot original time series versus predicted time series.

    Args:
        time_series: Original time series
        time_series_pred: Predicted time series generated from a time series model.
        orig_label: Label name for actual target values.
        pred_label: Label name for predicted/forecasted target values.
        Title: Title for the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=[9, 4])
    sales_orig_plot = plt.plot(time_series, color='blue', label=orig_label)
    sales_pred_plot = plt.plot(time_series_pred, color='orange', label=pred_label)
    plt.legend(loc='best')
    plt.title(title)
    sns.despine()
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Order date grouped by month')
    ax.set_ylabel('Sales')
    plt.show()


def residual_tests(model):
    """Prints results of various statistical tests on time series model residuals.

    Jarque-Bera normality test: Null hypothesis is residuals are normally distributed.

    Ljung-Box serial correlation test: Null hypothesis is no serial correlations
    in residuals.

    Durbin-Watson serial correlation test: Want a score between 1 and 3. 2 is ideal score.

    Heteroskedasticity test: Null hypothesis is no heteroskedasticity.

    Args:
        model: Model to run statistical tests on.

    Returns:
        None
    """
    jb_val, jb_p, skew, kurtosis = model.test_normality(method='jarquebera')[0]
    lb_val, lb_p = model.test_serial_correlation(method='ljungbox')[0]

    # We want to look at the largest lag for Ljung-Box, so we take the last value,
    # which corresponds to the test statistic for the largest lag
    # Ljung-Box figures out the number of lags to use if we don't specify it
    lb_val = lb_val[-1]
    lb_p = lb_p[-1]

    het_val, het_p = model.test_heteroskedasticity('breakvar')[0]
    durbin_watson = db(model.filter_results. \
        standardized_forecasts_error[0, model.loglikelihood_burn:])

    print(f'{"Jarque-Bera normality test:": <35} val={jb_val:.2f} p={jb_p:.2f}')
    print(f'{"Ljung-Box serial corr. test:": <35} val={lb_val:.2f} p={lb_p:.2f}')
    print(f'{"Durbin-Watson serial corr. test:": <35} d={durbin_watson:.2f}')
    print(f'{"Heteroskedasticity test:": <35} val={het_val:.2f} p={het_p:.2f}')


def simple_validation(model, X_train, y_val):
    """Compares model's forecast for y_val versus actual y_val using plots and RMSE.

    Args:
        model: Model used to forecast y_val.
        X_train: Data that model was trained on.
        y_val: True target values for validation set.

    Returns:
        None
    """
    y_val_pred = model.predict(start=len(X_train), end=len(X_train)+len(y_val)-1, dynamic=True)
    plot_pred(y_val, y_val_pred,
              orig_label='y_val', pred_label='y_val_pred',
              title='y_val v. y_val_pred')
    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f'Val RMSE: {val_rmse:.2f}')


def test_model(model, X_train_val, y_test):
    """Compares model's forecast for y_test versus actual y_test using plots and RMSE.

    Args:
        model: Model used to forecast y_test.
        X_train_val: Data that model was trained on.
        y_test: True target values for test set.

    Returns:
        None
    """
    y_test_pred = model.predict(start=len(X_train_val),
                                end=len(X_train_val)+len(y_test)-1, dynamic=True)
    plot_pred(y_test, y_test_pred,
              orig_label='y_test', pred_label='y_test_pred',
              title='y_test v. y_test_pred')
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    print(f'Test RMSE: {test_rmse:.2f}')
