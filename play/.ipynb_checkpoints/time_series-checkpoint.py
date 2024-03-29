from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import List, Tuple, Union, Any, TypedDict
from abc import ABC, abstractmethod

from constants import Number, TimeSeriesData

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
    time_col: str
    time_values: List[Any]
    values_cols: List[str]
    values: TimeSeriesData


class TimeSeriesFactory:
    """Abstract factory for creating time series objects."""
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
        values_cols = kwargs["values_cols"]
        time_values = kwargs["time_values"]

        if len(values_cols) > 1:
            len_values = [len(v) for v in values]
            equiv_dimensions = [len(time_values) == lv for lv in len_values]
            if all(equiv_dimensions):
                return True
            else:
                raise ValueError(
                    "All time series dimensions must be"
                    + f" equivalent. Lengths: {len_values}"
                )
        return False


class TimeSeriesMixin(ABC):
    def __init__(self, **kwargs: TimeSeriesParameters):
        """Build a time series object from a time index and univariate or
        multivariate time series data.
        """
        col_names, col_values = TimeSeriesMixin._get_col_names_and_values(
            **kwargs
        )

        if not TimeSeriesFactory._is_univariate_time_series(**kwargs):
            # Unpack column values for multivariate time series
            cvs = [col_values[0]]
            # Exclude the time column values
            values_per_value_col = col_values[1]
            for value_col in values_per_value_col:
                cvs.append(value_col)
            col_values = cvs

        self.data = pd.DataFrame(
            {
                name: data for name, data in zip(col_names, col_values)
            }
        )
        self.data.set_index(kwargs["time_col"], inplace=True)

    @staticmethod
    def _get_col_names_and_values(
        **kwargs: TimeSeriesParameters
    ) -> Tuple[List[str], List[Any]]:
        """Get the column names and values from the time series parameters."""
        values = kwargs["values"]
        values_cols = kwargs["values_cols"]
        time_values = kwargs["time_values"]
        time_col = kwargs["time_col"]

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

        return col_names, col_values

    def mean(self, axis: int = 0):
        return self.data.mean(axis=axis)

    def std(self, axis: int = 0):
        return self.data.std(axis=axis)

    def variance(self, axis: int = 0) -> pd.Series:
        return self.data.var(axis=axis)

    def max_min_range(self, axis: int = 0) -> pd.Series:
        max_value = self.data.max(axis=axis)
        min_value = self.data.min(axis=axis)
        range = max_value - min_value
        return max_value, min_value, range

    def __str__(self) -> str:
        columns = ", ".join(self.data.columns)
        return f"{self.__name__}({columns})"

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_statistics(self) -> pd.DataFrame:
        """Get the statistics of the univariate time series data.

        Returns
        -------
        stats: `pd.DataFrame`
            The statistics of the univariate time series data
        """
        return self.data.describe()

    def get_train_validation_test_split(
        self,
        train_size: float = 0.6,
        validation_size: float = 0.2
    ) -> Tuple[TimeSeries, ...]:
        """Get the train, validation, and test splits of the time series data.

        Parameters
        ----------
        train_size: `float`
            The size of the training split
        validation_size: `float`
            The size of the validation split
        test_size: `float`
            The size of the test split

        Returns
        -------
        train: `TimeSeries`
            The training split
        validation: `TimeSeries`
            The validation split
        test: `TimeSeries`
            The test split
        """
        train_size = int(train_size * len(self))
        validation_size = int(validation_size * len(self))

        train_set, validation_set, test_set = self.\
            _get_train_validation_test_split(
                train_size=train_size,
                validation_size=validation_size,
        )

        return (train_set, validation_set, test_set)

    @abstractmethod
    def _get_train_validation_test_split(
        self,
        train_size: int,
        validation_size: int
    ) -> Tuple[TimeSeries, ...]:
        pass

    def get_historical_data(self, time_series: np.array) -> np.array:
        """Split to only get the historical time series data.

        Parameters
        ----------
        time_series: `np.array`
            The time series

        Returns
        -------

        """

        return time_series[:-1]

    def get_true_label_data(self, time_series: np.array) -> np.array:
        """Split to only get the true label time series data.

        Parameters
        ----------
        time_series: `np.array`
            The time series

        Returns
        -------

        """

        return time_series[-1:]


class UnivariateTimeSeries(TimeSeriesMixin):

    __name__ = "UnivariateTimeSeries"

    def __init__(self, **kwargs: TimeSeriesParameters):
        super().__init__(**kwargs)

    @property
    def time_col(self) -> str:
        """Get the name of the time column."""
        return self.data.index.name

    @property
    def value_col(self) -> str:
        """Get the name of the value column."""
        return self.data.columns[0]

    def _get_train_validation_test_split(
        self,
        train_size: int,
        validation_size: int,
    ) -> Tuple[UnivariateTimeSeries, ...]:
        """Get the train, validation, and test splits of the time series data.

        Parameters
        ----------
        train_size: `int`
            The size of the training split
        validation_size: `int`
            The size of the validation split

        Returns
        -------
        train: `UnivariateTimeSeries`
            The training split
        validation: `UnivariateTimeSeries`
            The validation split
        test: `UnivariateTimeSeries`
            The test split
        """
        train = self.get_slice(0, train_size)
        validation = self.get_slice(train_size, train_size + validation_size)
        test = self.get_slice(train_size + validation_size, len(self))

        return (train, validation, test)

    # TODO: This should support start and end values that correspond to the
    # type of the time index.
    def get_slice(self, start: int, end: int) -> UnivariateTimeSeries:
        """Get a slice of the univariate time series data.

        Parameters
        ----------
        start: `int`
            The index to start the slice
        end: `int`
            The index to end the slice

        Returns
        -------
        uts: `UnivariateTimeSeries`
            A new instance of univariate time series with the sliced data
        """
        slice_uts = type(self)(
            time_col=self.time_col,
            time_values=self.data.index[start:end],
            values_cols=f"{self}[{start}:{end}]",
            values=self.data[self.value_col].values[start:end].copy()
        )

        return slice_uts

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
        data = self.data[self.value_col].values
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
        normalized_data = self.data[self.value_col].copy().values
        normalized_data = (normalized_data - mean) / std

        normalized_uts = type(self)(
            time_col=self.time_col,
            time_values=self.data.index.values,
            values_cols=f"Normalized({self.value_col})",
            values=normalized_data
        )

        return normalized_uts

    def get_series(self, with_label: bool = False, is_copy=True) -> np.array:
        """Get the univariate time series data."""
        data = self.data.copy() if is_copy else self.data
        if not with_label:
            return data[self.value_col].values
        return (data[self.value_col].values, self.value_col)

    def get_order_k_diff(self, k: int = 1) -> UnivariateTimeSeries:
        """Compute an order-k difference on the time series.

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
        assert k + 1 <= len(self), f"Order-{k} differences can't be applied" \
            + f" to data with {len(self.data)} elements"
        diff = np.diff(self.data[self.value_col].values, n=k)

        order_k_diff_uts = type(self)(
            time_col=self.time_col,
            time_values=self.data.index[:diff.shape[0]],
            values_cols=f"Order-{k} Difference of {self.value_col}",
            values=diff
        )

        return order_k_diff_uts

    def plot(self, tick_skip=90):
        # Plot the time series data
        plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size

        plt.plot(self.data.index, self.data[self.value_col])
        plt.xlabel(self.time_col)
        plt.ylabel(self.value_col)
        plt.title(f"Plot of {self}")

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=tick_skip))

        # Rotate the x-axis tick labels for better visibility (optional)
        plt.xticks(rotation=45)

        # Display the plot
        plt.show()

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
        plt.figure(figsize=(10, 6))
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
            self.data[self.value_col].values[:-lag],
            self.data[self.value_col].values[lag:]
        )
        plt.xlabel(f"{self.value_col} at t")
        plt.ylabel(f"{self.value_col} at t + {lag}")
        plt.title(f"Scatter Plot of {self} at lag {lag}")

        # Using the normal equations, add a line of best fit to the scatter
        # plot
        x = self.data[self.value_col].values[:-lag]
        y = self.data[self.value_col].values[lag:]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        plt.plot(x, m * x + c, 'r', label='Fitted line')

        # Display the plot
        plt.show()
    
    def stationarity_test(self, t: int, t_lag_k: int) -> str:
        """Determine if the mean and variance of the time series is stationary, nonstationary, weak stationary, strong stationary. See page 25.
        
        Parameters
        ----------
        t: `int`
            The t-th observation
        t_lag_k: `int`
            The t-th observation with lag k
        
        Returns:
        type_of_ts: `str`
        """
        X = self.get_series(True)
        split_series = int(len(X) / 2)
        print(split_series)
        X1, X2 = X[0:split_series], X[split_series:]
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%f, mean2=%f' % (mean1, mean2))
        print('variance1=%f, variance2=%f' % (var1, var2))
        


class MultivariateTimeSeries(TimeSeriesMixin):
    __name__ = "MultivariateTimeSeries"

    def __init__(self, **kwargs: TimeSeriesParameters):
        super().__init__(**kwargs)

    @property
    def columns(self) -> List[str]:
        """Return the column names of the time series data."""
        return self.data.columns.tolist()

    def __getitem__(self, col_name: str) -> UnivariateTimeSeries:
        """Return a univariate time series of the given column name."""
        return UnivariateTimeSeries(
            time_col=self.data.index.name,
            time_values=self.data.index,
            values_cols=col_name,
            values=self.data[col_name].values
        )

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
        normalized_data[self.columns] = normalized_data[self.columns].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        # Plot each time series
        for col in self.columns:
            plt.plot(self.data.index, normalized_data[col], label=col)

        # Display the plot
        plt.legend()
        plt.show()


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
