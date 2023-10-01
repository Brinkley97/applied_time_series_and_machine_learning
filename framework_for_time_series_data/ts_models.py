import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from abc import ABC
from math import sqrt
from typing import List
from abc import abstractmethod
from dataclasses import dataclass

from statsmodels.tsa.ar_model import AutoReg

from constants import Number, TimeSeriesData
from time_series import UnivariateTimeSeries

from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error, mean_absolute_percentage_error

# Define the abstract base class
@dataclass
class Model(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

class AR(Model):
    def __name__(self):
        return "AR"

    def train_ar_model(self, train_data, lag):
        ar_model = AutoReg(train_data, lags=lag)
        train_ar_model = ar_model.fit()
        train_ar_model.summary()

        return train_ar_model

    def ar_predict(self, trained_ar_model, len_historical_data: np.array, train: np.array, test: np.array, dynamic) -> np.array:

        return trained_ar_model.predict(start=len_historical_data, end=len(train)+len(test)-1, dynamic=dynamic)

@dataclass
class EvaluationMetric:
    """Investigate the philosphy/design behind typing in python.

    https://realpython.com/python-type-checking/
    """

    def eval_mse(true_labels: np.array, predictions: np.array) -> float:
        """Calculate the mean squared error"""
        for predictions_idx in range(len(predictions)):
            prediction = predictions[predictions_idx]
            true_label = true_labels[predictions_idx]
            print('expected=%f, predicted=%f' % (true_label, prediction))
        mse = sqrt(mean_squared_error(true_labels, predictions))
        return mse

    def plot_forecast(true_labels: np.array, predictions: np.array):
        """Plots the forecast of each model respectively on the same plot."""
        plt.plot(true_labels, color='blue')
        plt.plot(predictions, color='red')
        plt.show()
