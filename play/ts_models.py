import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from abc import ABC
from typing import List
from abc import abstractmethod
from dataclasses import dataclass

from constants import Number, TimeSeriesData
from time_series import UnivariateTimeSeries

# Define the abstract base class
@dataclass
class Model(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """
    data: pd.DataFrame

    def plot1D(col_name: str, dataset_name: str, data):
        """

        Parameters
        ----------
        col_name: `str`
            The name of the column(s) corresponding to the univariate or
            multivariate time series data
        dataset_name: `str`
            The name of the dataset(s) corresponding to the univariate or
            multivariate time series data
        data: `TimeSeriesData`
            The univariate or multivariate time series raw data
        """

        plt.plot(data, col_name)

        plt.title(f"Univariate Time Series {ticker}")
        plt.xlabel("Time")
        plt.ylabel("Price")

        plt.show()
