import matplotx

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from abc import ABC
from math import sqrt
from typing import List
from abc import abstractmethod
from dataclasses import dataclass

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from constants import Number, TimeSeriesData
from time_series import UnivariateTimeSeries

from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error, mean_absolute_percentage_error

# Define the abstract base class
@dataclass
class Model(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    def augment_data(self):
        pass

    @abstractmethod
    def predict(self):
        pass


# class AR(AutoReg):
# OR
class AR(Model):
    def __name__(self):
        return "AR"

    def train_ar_model(self, train_data: np.array, test_lags: list) -> list:
        """Initial and train an autoregressive model.

        Parameters
        ----------
        train_data: `np.array`
            Data to train our autoregressive model on
        test_lags: `list`
            A list of lag values to pass to autoregressive model

        Returns
        ------
        trained_ar_models: `list`
            A list of trained autoregressive models with each differing by lag value

        """
        trained_ar_models = []
        for test_lags_idx in range(len(test_lags)):
            test_lag = test_lags[test_lags_idx]
            print("Model", test_lags_idx + 1, "with a lag of", test_lag)

            ar_model = AutoReg(train_data, lags=test_lag)
            trained_ar_model = ar_model.fit()
            trained_ar_model.summary()
            trained_ar_models.append(trained_ar_model)

        return trained_ar_models

    def predict(self, trained_ar_models, len_historical_data: np.array, train: np.array, test: np.array) -> np.array:
        """Make predictions with trained autoregressive models.

        Parameters
        ----------
        trained_ar_models: AR models
            Trained autoregressive models
        len_historical_data: `np.array`
            The length of our historical data
        train: `np.array`
            The training data
        test: `np.array`
            The testing data

        Returns
        ------
        predictions: `list`
            A list of predictions for each autoregressive model with each differing by lag value

        """

        predictions = []
        for trained_ar_models_idx in range(len(trained_ar_models)):
            trained_ar_model = trained_ar_models[trained_ar_models_idx]
            print("Model", trained_ar_models_idx + 1, trained_ar_model)
            model_prediction = trained_ar_model.predict(start=len_historical_data, end=len(train)+len(test)-1, dynamic=False)
            predictions.append(model_prediction)

        return predictions

class PersistenceWalkForward(Model):
    def __name__(self):
        return "Persistence Walk Forward"

    def augment_data(self, df: pd.DataFrame, sliding_window: int) -> pd.DataFrame:
        lags_df = pd.concat([df.shift(sliding_window), df], axis=1)
        lags_df.columns = ['t - 1', 't + 1']

        return lags_df

    def pwf_model(self, x):
        return x

    def predict(self, test_X):
        predictions = []
        for x in test_X:
            yhat = self.pwf_model(x)
            predictions.append(yhat)
            print('Predicted Forecasts:', predictions)

        return predictions

class MA(Model):
    def __name__(self):
        return "MA"

    def train_model(self, train_data: np.array, test_error_terms: list) -> list:
        """Initial and train an autoregressive model.

        Parameters
        ----------
        train_data: `np.array`
            Data to train our autoregressive model on
        test_lags: `list`
            A list of lag values to pass to autoregressive model

        Returns
        ------
        trained_ar_models: `list`
            A list of trained autoregressive models with each differing by lag value

        """
        trained_ma_models = []
        for test_error_terms_idx in range(len(test_error_terms)):
            test_error_term = test_error_terms[test_error_terms_idx]
            print("MA(", test_error_term, ")")

            ma_model = ARIMA(train_data, order=(0, 0, test_error_terms))
            trained_ma_model = ma_model.fit()
            trained_ma_model.summary()
            trained_ma_models.append(trained_ma_model)

        return trained_ma_models

    def predict(self, trained_ma_models, len_historical_data: np.array, train: np.array, test: np.array) -> np.array:
        """Make predictions with trained autoregressive models.

        Parameters
        ----------
        trained_ar_models: AR models
            Trained autoregressive models
        len_historical_data: `np.array`
            The length of our historical data
        train: `np.array`
            The training data
        test: `np.array`
            The testing data

        Returns
        ------
        predictions: `list`
            A list of predictions for each autoregressive model with each differing by lag value

        """

        predictions = []
        for trained_ma_models_idx in range(len(trained_ma_models)):
            trained_ma_model = trained_ma_models[trained_ma_models_idx]
            print("MA(", trained_ma_model, ")")
            model_prediction = trained_ma_model.predict(start=len_historical_data, end=len(train)+len(test)-1, dynamic=False)
            predictions.append(model_prediction)

        return predictions

@dataclass
class EvaluationMetric:
    """Investigate the philosphy/design behind typing in python.

    https://realpython.com/python-type-checking/
    """

    def eval_mse(true_labels: np.array, predictions: np.array, per_element=True):
        """Calculate the mean squared error"""
        if per_element == True:
            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]
                mse = sqrt(mean_squared_error(true_labels, prediction))
                print("expected", true_labels, "predicted", prediction, "mse", mse)
        else:
            mse = mean_squared_error(true_labels, predictions)
            print('Test MSE: %.3f' % mse)

    def plot_forecast(true_labels: np.array, predictions: np.array, test_lags: list, with_lags= True):
        """Plots the forecast of each model respectively on the same plot."""

        if with_lags == True:
            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]
                lag = test_lags[predictions_idx]

                plt.figure(figsize=(20, 4))
                plt.xlabel("Observations")
                plt.ylabel("Values")
                plt.title(f"Model {predictions_idx + 1} with Lag {lag}")

                plt.plot(true_labels, color='blue', label='Actual Forecasts', linewidth=4)
                plt.plot(prediction, color='red', label='Predicted Forecasts', linewidth=4)
        else:
            plt.figure(figsize=(20, 4))
            plt.xlabel("Observations")
            plt.ylabel("Values")
            plt.title(f"Model")

            plt.plot(true_labels, color='blue', label='Actual Forecasts', linewidth=4)
            plt.plot(predictions, color='red', label='Predicted Forecasts', linewidth=4)


        matplotx.line_labels()
        plt.show()
