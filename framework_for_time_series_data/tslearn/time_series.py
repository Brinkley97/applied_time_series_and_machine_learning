"""
Detravious Jamari Brinkley (aka FitToCode)

Factory Pattern: https://refactoring.guru/design-patterns/factory-method/python/example#lang-features
"""

from __future__ import annotations # must occur at the beginning of the file
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from abc import ABC, abstractmethod
# test for stationarity
from statsmodels.tsa.stattools import adfuller, bds
from sklearn.model_selection import train_test_split


# partial autocorrelation
from statsmodels.graphics import tsaplots

from constants import Number, TimeSeriesData
from typing import List, Tuple, Union, Any, TypedDict

TimeSeries = Union["UnivariateTimeSeries", "MultivariateTimeSeries"]

class TimeSeriesParameters(TypedDict):
    """Typed dict for passing arbitrary named parameters to a time series
    object.

    This represents the bare minimum of parameters that must be passed to a
    time series object.

    Parameters
    ----------
    time_col: `str`
        The name of the column corresponding to the time index
    time_values: `List[Any]`
        The values of the time index
    value_cols: `List[str]`
        The name of the column(s) corresponding to the univariate or
        multivariate time series data
    value: `TimeSeriesData`
        The univariate or multivariate time series raw data
    """
    # dictionary-like structure:
    # key : value
    time_col: str
    time_values: List[Any]
    values_cols: List[str]
    values: TimeSeriesData

class TimeSeriesFactory:
    """Abstract factory for creating time series objects (UnivariateTimeSeries, MultivariateTimeSeries)."""
    @staticmethod
    def create_time_series(**kwargs: TimeSeriesParameters) -> TimeSeries:
        """Create a time series object from a time index and univariate or
        multivariate time series data.
        """
        if TimeSeriesFactory._is_univariate_time_series(**kwargs):
            return UnivariateTimeSeries(**kwargs)
        elif TimeSeriesFactory._is_multivariate_time_series(**kwargs):
            return MultivariateTimeSeries(**kwargs)
        else:
            raise TypeError(
                "We could not determine if the given data"
                + " belongs to a univariate or multivariate time series."
            )

    @staticmethod
    def _is_univariate_time_series(**kwargs: TimeSeriesParameters):
        values = kwargs["values"]
        values_cols = kwargs["values_cols"]
        time_values = kwargs["time_values"]

        if (
            (isinstance(values_cols, list) and len(values_cols) == 1) and
            len(values) == len(time_values)
        ):
            return True
        elif isinstance(values_cols, str) and len(values) == len(time_values):
            return True
        return False

    @staticmethod
    def _is_multivariate_time_series(**kwargs: TimeSeriesParameters):
        values = kwargs["values"]
        # print("values", values)
        values_cols = kwargs["values_cols"]
        # print("values_cols", len(values_cols))
        time_values = kwargs["time_values"]
        # print("time_values", time_values)
        

        # TODO: Should be strictly > 1. With >=, we're able to process a single stock without having to include another that matches the exact time frame
        if len(values_cols) >= 1:
            len_values = []
            for v in values:
                len_values.append(len(v))

            print("Len", len_values)
            all_lengths_match = True
            for len_value in len_values:
                print("Len", len_values)
                if len_value != len(time_values):
                    all_lengths_match = False
                    break
                    
            if all_lengths_match:
                return True
            else:
                raise ValueError(
                    "All time series dimensions must be"
                    + f" equivalent. Lengths: {len_values}"
                )
        return False
        
        # if isinstance(values, list) and isinstance(values[0], list) and \
        #     isinstance(values_cols, list) and isinstance(values_cols[0], str):
        #     num_cols_values = len(values[0])
        #     num_cols_values_cols = len(values_cols)

        #     if num_cols_values == num_cols_values_cols:
        #         return True  # It's multivariate

        # if TimeSeriesFactory._is_univariate_time_series(values=values, values_cols=values_cols, time_values=time_values):
        #     counter = 1  # Counter to track the number of columns
        #     if isinstance(values_cols, list):
        #         counter += len(values_cols) - 1  # Increment counter by the number of extra columns
        #     if counter > 1:
        #         # Reformat data for multivariate case
        #         ts_params = {
        #             "time_col": kwargs["time_col"],
        #             "time_values": time_values,
        #             "values_cols": values_cols,
        #             "values": values  # Assuming TimeSeriesData is already in the correct format
        #         }
        #         return ts_params
        # return None  # Return None if it's not univariate or counter <= 1

class TimeSeriesMixin(ABC):
    "from https://chatgpt.com/c/fa331693-fa17-4397-be72-a1b5413c6a41"

    def __init__(self, **kwargs: TimeSeriesParameters):

        col_names, col_values, df = TimeSeriesMixin._get_col_names_and_values(
            **kwargs
        )
        # print("col_names: ", col_names)
        # print("col_values: ", col_values)
        # print()

        if not TimeSeriesFactory._is_univariate_time_series(**kwargs):
            time_col, time_values, values_cols, values = df[0], df[1], df[2], df[3]

            self.data = pd.DataFrame()
            # print("values: ", type(values), values)
            for values_idx in range(len(values)):
                nth_col = values[values_idx]
                self.data[values_cols[values_idx]] = nth_col
                self.data.index.name = time_col
                self.data.index = time_values

            # if len(values) == 1:
            #     for value_col in values_per_value_col:
            #         print("value_col: ", value_col)
            #         cvs.append(value_col)
            #     col_values = cvs
            #     print("col_values-2: ", type(col_values), col_values)

            # self.data = pd.DataFrame({col: vals for col, vals in zip(values_cols, value)})
            # self.data.set_index(kwargs["time_col"], inplace=True)
        if TimeSeriesFactory._is_univariate_time_series(**kwargs):
            self.data = pd.DataFrame(
                {
                    name: data for name, data in zip(col_names, col_values)
                }
            )
            self.data.set_index(kwargs["time_col"], inplace=True)
    
    @staticmethod
    def _get_col_names_and_values(**kwargs: TimeSeriesParameters) -> Tuple[List[str], List[Any]]:
        """Get the column names and values from the time series parameters."""
        
        time_col = kwargs["time_col"]
        time_values = kwargs["time_values"]
        values_cols = kwargs["values_cols"]
        values = kwargs["values"]

        if isinstance(values_cols, list):
            col_names = [time_col] + values_cols
        elif isinstance(values_cols, str):
            col_names = [time_col, values_cols]
        else:
            raise TypeError(
                "Values columns must be a list or a string."
                + f" Received: {type(values_cols)}"
            )

        col_values = [time_values, values]

        df = [time_col, time_values, values_cols, values]

        return col_names, col_values, df

    def get_statistics(self) -> pd.DataFrame:
        """Get the statistics of the univariate time series data.

        Returns
        -------
        stats: `pd.DataFrame`
            The statistics of the univariate time series data
        """
        return self.data.describe()

    # write code to support the returning of the specific date for the max, min, and range
    def range_skewness_kurtosis(self, axis: int = 0) -> pd.Series:
        max_value = self.data.max(axis=axis)
        min_value = self.data.min(axis=axis)
        range_value = max_value - min_value
        skewness_value = self.skewness(axis=axis)
        kurtosis_value = self.kurt(axis=axis)
        return ({"Range": range_value, "Skewness": skewness_value, "Kurtosis": kurtosis_value})

    def mean(self, axis: int = 0):
        return self.data.mean(axis=axis)

    def std(self, axis: int = 0):
        return self.data.std(axis=axis)

    def variance(self, axis: int = 0) -> pd.Series:
        return self.data.var(axis=axis)

    def skewness(self, axis: int = 0) -> pd.Series:
        """The distribution of the values in the time series will become asymmetric. 3rd moment (see (2))

        If skewness is less than -1 or greater than 1, the distribution is highly skewed.
        If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
        If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

        Reference: (1) https://www.early-warning-signals.org/?page_id=117, (2) https://detraviousjbrinkley.notion.site/545-L7-Multi-variate-Normal-Distribution-Conditional-Distributions-Weak-and-Strict-Stationarity-9a56848d712c475b88cb0b6745e44bfb?pvs=4
        """
        return self.data.skew(axis=axis)

    def kurt(self, axis: int = 0) -> pd.Series:
        """Describes the shape of the distribution's tails (how heavy (become fatter due to the increased presence of rare values in the TS) or light they are). 4th moment (see (2))

        Reference: (1) https://www.early-warning-signals.org/?page_id=117, (2) https://detraviousjbrinkley.notion.site/545-L7-Multi-variate-Normal-Distribution-Conditional-Distributions-Weak-and-Strict-Stationarity-9a56848d712c475b88cb0b6745e44bfb?pvs=4
        """
        return self.data.kurtosis(axis=axis)

    def __str__(self) -> str:
        columns = ", ".join(self.data.columns)
        return f"{self.__name__}({columns})"

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_train_validation_test_split(X, y, test_size: 0.2, shuffle: bool =False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        return X_train, X_test, y_train, y_test

    def get_historical_data(self, forecasting_step) -> np.array:
        historical_data = len(self.data) - forecasting_step
        return self.data[:historical_data]

    def get_true_label_data(self, forecasting_step) -> np.array:
        """Split to only get the true label time series data.

        Parameters
        ----------
        time_series: `np.array`
            The time series

        Returns
        -------

        """

        return self.data[-forecasting_step:]

class UnivariateTimeSeries(TimeSeriesMixin):

    __name__ = "UnivariateTimeSeries"

    def __init__(self, **kwargs: TimeSeriesParameters):
        super().__init__(**kwargs)

    @property # a built-in Python decorator [a function that takes another function and extends the behavior of the latter function without explicitly modifying it]. It is used to give "special" functionality to certain methods
    def get_time_col_name(self) -> str:
        """Get the name of the time column."""
        return self.data.index.name

    @property
    def get_value_col_name(self) -> str:
        """Get the name of the value column."""
        return self.data.columns[0]

    def get_as_df(self) -> pd.DataFrame:
        """Get the name and data."""
        return self.data

    def get_series(self, with_label: bool = False, is_copy = True) -> np.array:
            """Get the univariate time series data."""
            data = self.data.copy() if is_copy else self.data
            if not with_label:
                return data[self.get_value_col_name].values
            return (data[self.get_value_col_name].values, self.get_value_col_name)

    def autocovariance(self, lag: int = 0) -> Number:
        """Compute the autocovariance of the time series data at a given lag.

        Parameters
        ----------
        lag: `int`
            The lag at which to compute the autocovariance

        Returns
        -------
        autocovariance: `Number`
            The autocovariance of the time series data at the given lag
        """
        assert lag < len(self), "The lag must be less than the length of" \
            + " the data"
        if lag == 0:
            return self.variance()[0]
        mean = self.mean()[0]
        data = self.data[self.get_value_col_name].values
        autocovariance = np.sum((data[lag:] - mean) * (data[:-lag] - mean)) \
            / len(self)
        return autocovariance

    def autocorrelation(self, lag: int = 0) -> Number:
        """Compute the autocorrelation of the time series data at a given lag.

        Parameters
        ----------
        lag: `int`
            The lag at which to compute the autocorrelation

        Returns
        -------
        autocorrelation: `Number`
            The autocorrelation of the time series data at the given lag
        """
        autocovariance = self.autocovariance(lag)
        return autocovariance / self.variance()[0]

    def autocorrelation_with_threshold(self, threshold: float = 0.1) -> list:
        """Check if autocorrelation value is above threshold and save

        Parameters
        ----------
        acorr_value: `int`
            The autocorrelation value
            acorr_value: float,
        threshold: `float`
            The cutoff value for our autocorrelations

        Returns
        -------
        threshold_autocorrelations: `list`
            A list of autocorrelations above our threshold
        """

        threshold_autocorrelations = []
        time_series = self.data

        for lag in range(1, len(time_series)):
            acorr_value = self.autocorrelation(lag)
            # print(acorr_value)
            if acorr_value > threshold:
                threshold_autocorrelations.append(lag)

        return threshold_autocorrelations

    def autocovariance_matrix(self, max_lag: int) -> np.array:
        """Compute the autocovariance matrix of the time series data.

        Parameters
        ----------
        max_lag: `int`
            The maximum lag at which to compute the autocovariance matrix

        Returns
        -------
        autocovariance_matrix: `np.array`
            The autocovariance matrix of the time series data
        """
        autocovariance_matrix = np.zeros((max_lag + 1, max_lag + 1))
        # Compute the autocovariance matrix using half the iterations taking
        # advantage of symmetry
        for i in range(max_lag + 1):
            for j in range(i, max_lag + 1):
                autocovariance_matrix[i, j] = self.autocovariance(np.abs(i - j))
                autocovariance_matrix[j, i] = autocovariance_matrix[i, j]
        return autocovariance_matrix

    def autocorrelation_matrix(self, max_lag: int) -> np.array:
        """Compute the autocorrelation matrix of the time series data.

        Parameters
        ----------
        max_lag: `int`
            The maximum lag at which to compute the autocorrelation matrix

        Returns
        -------
        autocorrelation_matrix: `np.array`
            The autocorrelation matrix of the time series data
        """
        autocovariance_matrix = self.autocovariance_matrix(max_lag)
        return autocovariance_matrix / self.variance()[0]

    def plot(self, tick_skip=90):
        # Plot the time series data

        plt.figure(figsize=(20, 5))  # Optional: Adjust the figure size

        # self.data is a pd.DataFrame
        plt.plot(self.data.index, self.data[self.get_value_col_name])
        plt.xlabel(self.get_time_col_name)
        plt.ylabel(self.get_value_col_name)
        plt.title(f"Plot of {self}")

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=tick_skip))

        # Rotate the x-axis tick labels for better visibility (optional)
        plt.xticks(rotation=45)

        # Display the plot
        plt.show()

    def stationarity_test(self, series):
        """Determine if the mean and variance of the time series is stationary, nonstationary, weak stationary, strong stationary.

        Null hypothesis: data has a unit root (data is non-stationary)
        Alt hypothesis: data is stationary

        If we reject the Null, then the data is stationary.
        In order to reject the null, we need our p-value to be less than our stat. sig. level

        In order to use most models inclusing machine learning models, the data must be stationary.

        Parameters
        ----------
        series: `list` or `pd.DataFrame`
            The list of observations
`
        """
        if type(series) == pd.DataFrame:
            series = self.get_series(False)

        adfuller_result = adfuller(series)
        adfuller_p_value = adfuller_result[1]
        significance_level = 0.05

        if adfuller_p_value < significance_level:
            print('ADF Statistic: %f' % adfuller_result[0])
            print('p-value: %f' % adfuller_result[1], '<', significance_level, ', so reject null-hypothesis as the TS is stationary')
            print('Critical Values:' )
            for key, value in adfuller_result[4].items():
                print('\t%s: %.3f' % (key, value))
        else:
            print('ADF Statistic: %f' % adfuller_result[0])
            print('p-value: %f' % adfuller_result[1], '>', significance_level, ', so accept the null-hypothesis as the TS is non-stationary')
            print('Critical Values:' )
            for key, value in adfuller_result[4].items():
                print('\t%s: %.3f' % (key, value))

    def independence_test(self, series):
        """Using the BDS test (after the initials of W. A. Brock, W. Dechert and J. Scheinkman), detect non-linear serial dependence in the TS by testing the null hypothesis that the remaining residuals are independent and identically distributed (i.i.d.).


        When: After detrending (or first-differencing) [remove linear structure]

        Null hypothesis: differenced data has no structure (differenced data is independent and i.i.d., so correlation is low and non-linear, non-stationary)
        Alt hypothesis: differenced data has structure (differenced data is dependent so correlation is high and linearity and stationarity are prevelant)


        If we reject the i.i.d. hypothesis, then the differenced data has some structure (thus dependent), which could include a hidden non-linearity (ie: NOT linear), hidden non-stationarity (ie: has some mean or variance), or other type of structure missed by detrending (or first-differencing) or model fitting.
        In order to reject the null, we need our P-value to be...

        Can help to avoid false detections of critical transitions due to model misspecification. Learn more on this.

        Reference: https://www.early-warning-signals.org/?page_id=121

        Parameters
        ----------
        series: `list` or `pd.DataFrame`
            The list of observations
`
        """
        if type(series) == pd.DataFrame:
            series = self.get_series(False)

        bds_result = bds(series)
        bds_p_value = bds_result[1]
        significance_level = 0.05

        if bds_p_value < significance_level:
            print('BDS Statistic: %f' % bds_result[0])
            print('p-value: %f' % bds_result[1], '<', significance_level, ', so reject null-hypothesis as the differenced TS is independent and i.i.d.')

        else:
            print('BDS Statistic: %f' % bds_result[0])
            print('p-value: %f' % bds_result[1], '>', significance_level, ', so accept the null-hypothesis as the differenced TS is dependent')

    def plot_autocorrelation(self, max_lag: int = 1, plot_full: bool = False):
        """Plot the autocorrelation of the time series data.

        Parameters
        ----------
        max_lag: `int`
            The maximum lag at which to compute the autocorrelation matrix
        """
        # Compute the autocorrelation matrix
        autocorrelation_matrix = self.autocorrelation_matrix(max_lag)

        # Plot the autocorrelation matrix
        plt.figure(figsize=(20, 4))
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation Coefficient")
        plt.title(f"Autocorrelation Matrix of {self} with Lag {max_lag}")

        # Plot the autocorrelation matrix as a bar plot where the height of the
        # bar is the autocorrelation for the given lag. The range of the bar
        # plot is [-max_lag, max_lag]
        x_axis_data, y_axis_data = None, None
        if plot_full:
            x_axis_data = np.arange(-max_lag, max_lag + 1)
            y_axis_data = np.concatenate([
                # Reverse the array to match the order of the lags
                autocorrelation_matrix[1:, 0][::-1],
                # The autocorrelation of the time series with itself is 1
                [1],
                # The autocorrelation plot is symmetric
                autocorrelation_matrix[1:, 0]
            ])
        else:
            x_axis_data = np.arange(0, max_lag + 1)
            y_axis_data = autocorrelation_matrix[:, 0]

        plt.bar(x_axis_data, y_axis_data)

        # Make the plot y bounds go from -1 to 1
        PADDING = .1
        plt.ylim(np.min(y_axis_data) - PADDING, np.max(y_axis_data) + PADDING)

        plt.show()

        tsaplots.plot_acf(self.data)
        plt.show()

    def plot_partial_autocorrelation(self, max_lag: int = 1):

        tsaplots.plot_pacf(self.data.squeeze(), lags=max_lag, method='ywm')
        plt.show()

    def scatter_plot(self, lag: int = 1):
        """Plot the univariate time series data against its lagged values.

        Parameters
        ----------
        lag: `int`
            The lag at which to plot the time series data
        """
        assert lag > 0, "Lag must be greater than 0"
        # Plot the time series data
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.data[self.get_value_col_name].values[:-lag],
            self.data[self.get_value_col_name].values[lag:]
        )
        plt.xlabel(f"{self.get_value_col_name} at t")
        plt.ylabel(f"{self.get_value_col_name} at t + {lag}")
        plt.title(f"Scatter Plot of {self} at lag {lag}")

        # Using the normal equations, add a line of best fit to the scatter
        # plot
        x = self.data[self.get_value_col_name].values[:-lag]
        y = self.data[self.get_value_col_name].values[lag:]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        plt.plot(x, m * x + c, 'r', label='Fitted line')

        # Display the plot
        plt.show()

    def data_augment_for_returns(self) -> UnivariateTimeSeries:
        """Calculate the percent change."""
        returns = self.data[self.get_value_col_name].pct_change().dropna().values.copy()

        returns_uts = type(self)(
            time_col=self.get_time_col_name,
            time_values=self.data.index[1:],
            values_cols="Returns",
            values=returns
        )

        return returns_uts

    def data_augment_reverse(self) -> UnivariateTimeSeries:
        """Reorganize the UTS with respect to the rows.
        """
        reverse_ts = self.data.iloc[::-1]
        # print(type(reverse_ts), reverse_ts)
        reversed_time_values = reverse_ts.index
        # print()
        # print(reversed_time_values)
        # print()
        reversed_values = reverse_ts[self.get_value_col_name].values
        # print(reversed_values)

        reversed_uts = type(self)(
            time_col=self.get_time_col_name,
            time_values=reversed_time_values,
            values_cols=self.get_value_col_name,
            values=reversed_values
        )

        return reversed_uts

    def data_augment_with_differencing(self, k_difference_order: int) -> UnivariateTimeSeries:
        """Calculate the differences between current observation and k previous observation for all observations.

        Parameters
        ----------
        k: `int`
            The k-th order difference to compute

        Returns
        -------
        uts: `UnivariateTimeSeries`
            An new instance of univariate time series with updated value column
            name
        """
        assert k_difference_order + 1 <= len(self), f"Order-{k_difference_order} differences can't be applied" \
            + f" to data with {len(self.data)} elements"

        returns = self.data[self.get_value_col_name].diff(k_difference_order).dropna().values.copy()

        order_k_diff_uts = type(self)(
            time_col=self.get_time_col_name,
            time_values=self.data.index[1:],
            values_cols=f"Order-{k_difference_order} Difference of {self.get_value_col_name}",
            values=returns
        )

        return order_k_diff_uts
    
    def data_augment_to_mvts(self, prior_observations: int, forecasting_step: int) -> Tuple[MultivariateTimeSeries, ...]:
        """Splits a given UvTS into multiple input rows where each input row has a specified number of timestamps and the output is a single timestamp.

        Parameters
        ----------
        prior_observations: `int`
            All observations before we get to where we want to start making the predictions.

        forecasting_step: `int`
            How far out to forecast, 1 only the next timestamp, 2 the next two timestamps, ... n the next n timestamps. 
            See data to accurately state how far to forecast.

        Returns
        -------
        Tuple[MultivariateTimeSeries, ...] 
            Training data for both X and y
        """

        df = self.data[self.get_value_col_name]

        cols = list()
        lag_col_names = []

        # input sequence (t-n, ... t-1)
        for prior_observation in range(prior_observations, 0, -1):
            # print("prior_observation: ", prior_observation)
            cols.append(df.shift(prior_observation))
            new_col_name = "t-" + str(prior_observation)
            # print(new_col_name)
            lag_col_names.append(new_col_name)


        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, forecasting_step):
            cols.append(df.shift(-i))
            new_col_name = "t"
            if forecasting_step == 1:
                lag_col_names.append(new_col_name)

            else:
                if i == 0:
                    lag_col_names.append(new_col_name)
                else:
                    new_col_name = "t+" + str(i)
                    lag_col_names.append(new_col_name)
            # put it all together
            uts_sml_df = pd.concat(cols, axis=1)
            uts_sml_df.columns=[lag_col_names]
            # drop rows with NaN values
            uts_sml_df.dropna(inplace=True)

        # colums to use to make prediction for last col
        X_train_df = uts_sml_df.iloc[:, :forecasting_step]

        X_values = []
        for col in X_train_df.columns:
            X_values.append(X_train_df[col].tolist())

        y_train_df = uts_sml_df.iloc[:, -prior_observations:]
        y_values = []
        for col in y_train_df.columns:
            y_values.append(y_train_df[col].tolist())
        
        return (MultivariateTimeSeries(
                    time_col=df.index.name,
                    time_values=X_train_df.index,
                    values_cols=lag_col_names[: forecasting_step],
                    values=X_values
                    ),
                MultivariateTimeSeries(
                    time_col=df.index.name,
                    time_values=y_train_df.index,
                    values_cols=lag_col_names[forecasting_step:],
                    values=y_values
                    )
                )
        # if get_train_or_test_mvts == "Train":
        #     return MultivariateTimeSeries(
        #         time_col=df.index.name,
        #         time_values=time_values,
        #         values_cols=values_cols,
        #         values=values
        #     )
        # elif get_train_or_test_mvts == "Test":
        #     return MultivariateTimeSeries(
        #         time_col=df.index.name,
        #         time_values=time_values,
        #         values_cols=values_cols,
        #         values=values
        #     )

    def old_augment_data(self, forecasting_step: int, prior_observations: int) -> pd.DataFrame:
        """Splits a given UTS into multiple input rows where each input row has a specified number of timestamps and the output is a single timestamp.

        Parameters:
        prior_observations -- py int (of all observations before we get to where we want to start making the predictions)
        forecasting_step -- py int (of how far out to forecast, 1 only the next timestamp, 2 the next two timestamps, ... n the next n timestamps)

        Return:
        agg.values -- np array (of new sml data)
        """

        df = self.data[self.get_value_col_name]
        # index_col = df.index.name

        cols = list()
        lag_col_names = []

        # input sequence (t-n, ... t-1)
        for prior_observation in range(prior_observations, 0, -1):
            # print("prior_observation: ", prior_observation)
            cols.append(df.shift(prior_observation))
            new_col_name = "t-" + str(prior_observation)
            # print(new_col_name)
            lag_col_names.append(new_col_name)


        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, forecasting_step):
            cols.append(df.shift(-i))
            new_col_name = "t"
            # print(new_col_name)
            lag_col_names.append(new_col_name)

            # put it all together
            uts_sml_df = pd.concat(cols, axis=1)
            uts_sml_df.columns=[lag_col_names]
            # drop rows with NaN values
            uts_sml_df.dropna(inplace=True)

        # colums to use to make prediction for last col
        # X_train = uts_sml_df.iloc[:, 0: -1]
        X_train_df = uts_sml_df.iloc[:, :prior_observations]
        # print("X_train_df: \n", X_train_df)

        time_values = X_train_df.index
        # print("time_values: \n", time_values)

        # values_cols = lag_col_names
        values_cols = list(X_train_df.columns)
        # print("values_cols: \n", values_cols)
        
        values_num = X_train_df.values
        # print("values: \n", len(values_num))
        

        # last column
        y_train_df = uts_sml_df.iloc[:, [-1]]
        
        # return X_train_df, y_train_df
        # return X_train_df.index, list(X_train_df.columns), X_train_df.values
        # print(len(X_train_df.values), X_train_df)
        # print(df.index.name)
        # print(len(X_train_df.index), X_train_df.index)
        # print(X_train_df.columns, lag_col_names, lag_col_names[0])
        # print(len(X_train_df.values), X_train_df.values)

        # expand_time_values = []
        # expand_values_col = []
        # num_columns = X_train_df.shape[1]  # Number of columns in X_train_df
        # for _ in range(num_columns):
        #     expand_values_col.extend([values_cols])

        
        for lag_col_names_idx in range(len(lag_col_names)):
            lag_col_name = lag_col_names[lag_col_names_idx]
            # print("X_train_df.index", X_train_df.index)
            # print("lag_col_name", lag_col_name)
            # print("X_train_df.iloc[0:, lag_col_names_idx]", X_train_df.iloc[0:, lag_col_names_idx])

            return UnivariateTimeSeries(
            time_col=df.index.name,
            time_values=X_train_df.index,
            values_cols=lag_col_name,
            values=X_train_df.iloc[0:, lag_col_names_idx]
        )
        
        # print(df.index.name)
        # print()
        # print(X_train_df.index)
        # print()
        # print(len(expand_values_col))
        # print()
        # print(len(X_train_df.values))
        # print()


        # print(df.index.name)
        # print()
        # print(lag_col_names.index)
        # print()
        # print(len(expand_values_col))
        # print()
        # print(len(lag_col_names))
        # print()
   

            # return MultivariateTimeSeries(
            # time_col=df.index.name,
            # time_values=X_train_df[lag_col_name].index,
            # values_cols=lag_col_name,
            # values=col_df.values[lag_col_names_idx]
        # return MultivariateTimeSeries(
        #     time_col=df.index.name,
        #     time_values=lag_col_names.index,
        #     values_cols=expand_values_col,
        #     values=lag_col_name
        # )

        
        # try to return a UnivariateTimeSeries and convert to MultivariateTimeSeries
        # as in return col 1 as UnivariateTimeSeries
        # return col 2 as UnivariateTimeSeries
        # then, combine UnivariateTimeSeries s to a single MultivariateTimeSeries
        # return MultivariateTimeSeries(
        #     time_col=df.index.name,
        #     time_values=X_train_df.index,
        #     values_cols=lag_col_names[0],
        #     values=X_train_df.values
        # )
    
        # time_col="date",
        # time_values=["2020-01-01", "2020-01-02"],
        # values_cols=["value1", "value2", "value3"],
        # values=[[1, 2], [3, 4], [5, 6]]

    def average_smoothing(self, sliding_window: int, with_plot=True) -> UnivariateTimeSeries:
        """Data prep step to smooth original TS data

        Parameters
        ----------
        sliding_window: `int`
            The number of observations to group

        """

        rolling = self.data.rolling(window=sliding_window)
        rolling_mean = rolling.mean().dropna()
        rolling_mean_time_values = rolling_mean.index
        rolling_mean_values = rolling_mean[self.get_value_col_name].values

        if with_plot == True:
            # NOTE: We're NOT using our plot func, thus can't update tick_skip
            plt.figure(figsize=(20, 5))

            plt.plot(self.get_as_df()[self.get_value_col_name], label='Original of Raw TS', linestyle='-')
            plt.plot(rolling_mean[self.get_value_col_name], label='Average Smoothing of Raw TS', linestyle='-')

            plt.xlabel(self.get_time_col_name)
            plt.ylabel(self.get_value_col_name)

            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=15))

            plt.xticks(rotation=45)
            plt.legend()  # Display legend for clarity
            plt.show()

        average_smoothed_uts = type(self)(
            time_col=self.get_time_col_name,
            time_values=rolling_mean_time_values,
            values_cols=self.get_value_col_name,
            values=rolling_mean_values
        )

        return average_smoothed_uts

    def get_slice(self, start: int, end: int, both_train_test: bool) -> UnivariateTimeSeries:
        """Get a slice of the univariate time series data. Use for TS Models

        Verifying with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

        Parameters
        ----------
        start: `int`
            The index to start the slice
        end: `int`
            The index to end the slice
        both_train_test: `bool`
            If False, call twice, separately for train and test
            If True, call once, together for train and test

        Returns
        -------
        if both_train_test == False
            sliced_uts: `UnivariateTimeSeries`
                A new instance of univariate time series with the sliced data

        if both_train_test == True
            train_sliced_uts, test_slice_uts: `UnivariateTimeSeries`
                A new instance of univariate time series with the sliced data

        """
        print(start, end)
        if both_train_test == False:
            sliced_uts = type(self)(
                time_col=self.get_time_col_name,
                time_values=self.data.index[start:end],
                values_cols=f"{self}[{start}:{end}]",
                values=self.data[self.get_value_col_name].values[start:end].copy()
            )

            return sliced_uts

        elif both_train_test == True:
            N = len(self.get_series())
            train_sliced_uts = type(self)(
                time_col=self.get_time_col_name,
                time_values=self.data.index[start:end],
                values_cols=f"{self}[{start}:{end}]",
                values=self.data[self.get_value_col_name].values[start:end].copy()
            )

            test_sliced_uts = type(self)(
                time_col=self.get_time_col_name,
                time_values=self.data.index[end:N],
                values_cols=f"{self}[{end}:{N}]",
                values=self.data[self.get_value_col_name].values[end:N].copy()
            )

            return train_sliced_uts, test_sliced_uts
        else:
            print("{both_train_test} is an invalid parameter")

    def get_slice_with_percentage(self, train_percent: float, both_train_test: bool, train_or_test: str) -> UnivariateTimeSeries:
        """Get a slice of the univariate time series data. Use for TS Models

        Parameters
        ----------
        train_percent: `float`
            Amount to use for training data

        train_or_test: `str`
            Specify if we want to split for train or test

        both_train_test: `bool`
            If False, call twice, separately for train and test
            If True, call once, together for train and test

        Returns
        -------
        if both_train_test == False
            sliced_uts: `UnivariateTimeSeries`
                A new instance of univariate time series with the sliced data

        if both_train_test == True
            train_sliced_uts, test_slice_uts: `UnivariateTimeSeries`
                A new instance of univariate time series with the sliced data

        """
        N = len(self.get_series())
        train_size = int(N * train_percent)

        if both_train_test == False:
            if train_or_test == 'Train':
                print(f"{train_or_test} size is", train_size)
                train_sliced_uts = type(self)(
                    time_col=self.get_time_col_name,
                    time_values=self.data.index[:train_size],
                    values_cols=f"{self}[{1}:{train_size}]",
                    values=self.data[self.get_value_col_name].values[:train_size].copy()
                )

                return train_sliced_uts

            elif train_or_test == 'Test':
                test_size = N - train_size
                print(f"{train_or_test} size is", test_size)
                test_sliced_uts = type(self)(
                    time_col=self.get_time_col_name,
                    time_values=self.data.index[test_size:],
                    values_cols=f"{self}[{test_size}:{N}]",
                    values=self.data[self.get_value_col_name].values[test_size:N].copy()
                )

                return test_sliced_uts

        elif both_train_test == True:
            train_sliced_uts = type(self)(
                time_col=self.get_time_col_name,
                time_values=self.data.index[:train_size],
                values_cols=f"{self}[{1}:{train_size}]",
                values=self.data[self.get_value_col_name].values[:train_size].copy()
            )

            test_sliced_uts = type(self)(
                time_col=self.get_time_col_name,
                time_values=self.data.index[train_size:N],
                values_cols=f"{self}[{train_size}:{N}]",
                values=self.data[self.get_value_col_name].values[train_size:N].copy()
            )

            return train_sliced_uts, test_sliced_uts
        else:
            print("{both_train_test} is an invalid parameter")

    def split_sequence(self, forecasting_step: int, prior_observations: int):
        """Splits a given UTS into multiple input rows where each input row has a specified number of timestamps and the output is a single timestamp. Use for ML models.

        Parameters
        ----------
        forecasting_step: `int`
            How far out to forecast (ie: 1 only the next timestamp, 2 the next two timestamps, ... n the next n timestamps)

        prior_observations: `int`
            The number of input observations to use to make our forecast
        """
        df = pd.DataFrame(self.get_series())
        cols = list()

        lag_col_names = []

        # input sequence (t-n, ... t-1)
        for prior_observation in range(prior_observations, 0, -1):
            cols.append(df.shift(prior_observation))
            new_col_name = "t-" + str(prior_observation)
            lag_col_names.append(new_col_name)

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, forecasting_step):
            cols.append(df.shift(-i))

            new_col_name = "t"
            if forecasting_step == 1:
                lag_col_names.append(new_col_name)

            else:
                if i == 0:
                    lag_col_names.append(new_col_name)
                else:
                    new_col_name = "t+" + str(i)
                    lag_col_names.append(new_col_name)

            # put it all together
            uts_sml_df = pd.concat(cols, axis=1)
            uts_sml_df.columns=[lag_col_names]
            # drop rows with NaN values
            uts_sml_df.dropna(inplace=True)

        return uts_sml_df

    def normalize(self) -> UnivariateTimeSeries:
        """Normalize the univariate time series data by subtracting the mean and
        dividing by the standard deviation.

        Returns
        -------
        uts: `UnivariateTimeSeries`
            An new instance of univariate time series with updated value column
            name
        """
        mean = self.mean()[0]
        std = self.std()[0]
        # Copy the data and grab the values from the value column
        normalized_data = self.data[self.get_value_col_name].copy().values
        normalized_data = (normalized_data - mean) / std

        normalized_uts = type(self)(
            time_col=self.time_col,
            time_values=self.data.index.values,
            values_cols=f"Normalized({self.get_value_col_name})",
            values=normalized_data
        )

        return normalized_uts

class MultivariateTimeSeries(TimeSeriesMixin):
    __name__ = "MultivariateTimeSeries"

    def __init__(self, **kwargs: TimeSeriesParameters):
        super().__init__(**kwargs)

    @property
    def get_columns(self) -> List[str]:
        """Return the column names of the time series data."""
        cols = self.data.columns.to_list()
        # print(cols)
        return cols

    def __getitem__(self, col_name: str) -> UnivariateTimeSeries:
        """Return a univariate time series of the given column name."""
        return UnivariateTimeSeries(
            time_col=self.data.index.name,
            time_values=self.data.index.tolist(),
            values_cols=[col_name],
            values=self.data[col_name].tolist()
        )
    
    def get_as_df(self) -> pd.DataFrame:
        """Get the name and data."""
        return self.data
    
    # def df_to_tensor(self, df, requires_grad, torch_dtype) -> torch.Tensor:
    #     """Convert DF to torch. Want to remove and use get_as_tensor() once I get running"""
    #     return torch.tensor(df.data.values, requires_grad=requires_grad, dtype=torch_dtype)
    
    # def get_as_tensor(self, requires_grad, torch_dtype) -> torch.Tensor:
    #     """Convert MvTS to torch tensor."""
    #     return torch.tensor(self.data.values, requires_grad=requires_grad, dtype=torch_dtype)
    
    def _get_train_validation_test_split(
        self,
        train_size: int,
        validation_size: int
    ) -> Tuple[MultivariateTimeSeries, ...]:
        # train = self.get_slice(0, train_size)
        # validation = self.get_slice(train_size, train_size + validation_size)
        # test = self.get_slice(train_size + validation_size, len(self))

        # return (train, validation, test)
        return (
            train_size,
            validation_size,
            len(self) - train_size - validation_size
        )

    def plot(self):
        """Create a plot of each column in the multivariate time series data.

        Normalize each time series to be within the same range and plot each
        series with their corresponding label.
        """
        # Plot the time series data
        plt.figure(figsize=(10, 6))
        plt.xlabel(self.data.index.name)
        plt.ylabel("Normalized Values")
        plt.title(f"Plot of {self}")

        # Normalize the time series data
        normalized_data = self.data.copy()
        normalized_data[self.get_columns] = normalized_data[self.get_columns].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        # Plot each time series
        for col in self.get_columns:
            plt.plot(self.data.index, normalized_data[col], label=col)

        # Display the plot
        plt.legend()
        plt.show()
    
    def data_augment_to_test(self, y_train_mvts: MultivariateTimeSeries, forecasting_step: int, prior_observations: int) -> Tuple[MultivariateTimeSeries, ...]:
        """Get docstring from data_augment_to_mvts in UvTS class. Adjust for here."""
        X_train_df = self.get_as_df()
        y_train_df = y_train_mvts.get_as_df()
        df = pd.concat([X_train_df, y_train_df], axis=1)
        lag_col_names = list(df.columns.values)

        predict_X_test_df = df.iloc[[-forecasting_step], -forecasting_step:]
        X_values = []
        for col in predict_X_test_df.columns:
            X_values.append(predict_X_test_df[col].tolist())

        predict_y_test_df = df.iloc[[-1], -prior_observations:]
        y_values = []
        for col in predict_y_test_df.columns:
            y_values.append(predict_y_test_df[col].tolist())
        
        return (MultivariateTimeSeries(
                    time_col=df.index.name,
                    time_values=predict_X_test_df.index,
                    values_cols=lag_col_names[: forecasting_step],
                    values=X_values
                    ),
                MultivariateTimeSeries(
                    time_col=df.index.name,
                    time_values=predict_y_test_df.index,
                    values_cols=lag_col_names[forecasting_step:],
                    values=y_values
                    )
                )
        
        # return predict_X_test_df, predict_y_test_df

if __name__ == "__main__":
    uts = TimeSeriesFactory.create_time_series(
        time_col="date",
        time_values=["2020-01-01", "2020-01-02", "2020-01-03"],
        values_cols="value",
        values=[1, 2, 3]
    )
    mvts = TimeSeriesFactory.create_time_series(
        time_col="date",
        time_values=["2020-01-01", "2020-01-02"],
        values_cols=["value1", "value2", "value3"],
        values=[[1, 2], [3, 4], [5, 6]]
    )
    print(uts.variance(axis=0))
    print(mvts.variance(axis=0))
