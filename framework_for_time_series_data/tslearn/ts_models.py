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

class RandomWalk(Model):
    def __name__(self):
        return "Persistence Walk Forward"

    def predict(self, train_raw_x: np.array, test_raw_y: np.array):
        """Make predictions with the Random Walk Model using the raw data. We use the raw data because we know there's a dependence of the current observation on the previous observation. We're able to capture the overall direction of the data.

        Formally stated by Jason Brownlee in: https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/
            'We can expect that the best prediction we could make would be to use the observation at the previous time step as what will happen in the next time step. Simply because we know that the next time step will be a function of the prior time step.'

        With this, we don't difference nor do we get the returns.

        Parameters
        ----------
        train_raw_x: `np.array`
            The raw train data
         test_raw_y: `np.array`
            The raw test data

        """
        predictions = list()
        history = train_raw_x[-1]
        # print(history)

        for i in range(len(test_raw_y)):
            yhat = history
            predictions.append(yhat)
            history = test_raw_y[i]

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

# extend class AR(AutoReg):
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
            print(trained_ar_model.summary())
            print()
            trained_ar_models.append(trained_ar_model)

        return trained_ar_models

    def predict(self, trained_ar_models, train_data: np.array, test_data: np.array) -> np.array:
        """Make predictions with trained autoregressive models.

        Parameters
        ----------
        trained_ar_models: AR models
            Trained autoregressive models


        Returns
        ------
        predictions: `list`
            A list of predictions for each autoregressive model with each differing by lag value

        """
        # This is correct. Example: Days 1, 2, 3, ..., 10. We want to predict day 8, 9, and 10. We train on days 1, 2, ..., 7. We test on days 8, 9, and 10. Start is length of historical data, here 7. End is 7 + 3 - 1 = 9. So, our model will make predictions from 7, 8, 9?
        start = len(train_data)
        end = start + len(test_data) - 1

        predictions = []
        for trained_ar_models_idx in range(len(trained_ar_models)):
            trained_ar_model = trained_ar_models[trained_ar_models_idx]
            print("Model", trained_ar_models_idx + 1, trained_ar_model)
            model_prediction = trained_ar_model.predict(start=start, end=end, dynamic=False)
            predictions.append(model_prediction)

        return predictions

class MA(Model):
    def __name__(self):
        return "MA"

    def train_model(self, train_data: np.array, test_error_terms: list) -> list:
        """Initial and train an moving average model.

        Parameters
        ----------
        train_data: `np.array`
            Data to train our autoregressive model on
        test_error_terms: `list`
            A list of error terms to pass to moving average model

        Returns
        ------
        trained_ma_models: `list`
            A list of trained moving average models with each differing by the moving average value we provide

        """
        trained_ma_models = []
        for test_error_terms_idx in range(len(test_error_terms)):
            test_error_term = test_error_terms[test_error_terms_idx]
            print("MA(", test_error_term, ")")

            ma_model = ARIMA(train_data, order=(0, 0, test_error_terms))
            trained_ma_model = ma_model.fit()
            print(trained_ma_model.summary())
            print()
            trained_ma_models.append(trained_ma_model)

        return trained_ma_models

    def predict(self, trained_ma_models, go: int, stop: int) -> np.array:
        """Make predictions with trained moving average models.

        Parameters
        ----------
        trained_ar_models: AR models
            Trained autoregressive models


        Returns
        ------
        predictions: `list`
            A list of predictions for each moving average model with each differing by lag value

        """

        predictions = []
        for trained_ma_models_idx in range(len(trained_ma_models)):
            trained_ma_model = trained_ma_models[trained_ma_models_idx]
            print("MA(", trained_ma_model, ")")
            model_prediction = trained_ma_model.predict(start=go, end=stop, dynamic=False)
            predictions.append(model_prediction)

        return predictions

class ARMA(Model):
    def __name__(self):
        return "ARMA"

    def train_arma_model(self, train_data: np.array, test_lags: list, test_error_terms: list) -> list:
        """Initial and train an autoregressive moving average model.

        Parameters
        ----------
        train_data: `np.array`
            Data to train our autoregressive model on
        test_lags: `list`
            A list of lag values to pass to autoregressive model
        test_error_terms: `list`
            A list of error terms to pass to moving average model

        Returns
        ------
        trained_ar_models: `list`
            A list of trained autoregressive moving average models with each differing by lag value

        """
        if len(test_lags) != len(test_error_terms):
            raise ValueError("Lengths of test_lags and test_error_terms must be the same")

        test_lags_and_error_terms = len(test_lags)
        trained_arma_models = []
        for test_lags_and_error_terms_idx in range(test_lags_and_error_terms):
            test_lag_term = test_lags[test_lags_and_error_terms_idx]
            test_error_term = test_error_terms[test_lags_and_error_terms_idx]
            print("ARMA(", test_lag_term, 0, test_error_term, ")")

            arma_model = ARIMA(train_data, order=(test_lag_term, 1, test_error_terms), trend="n")
            trained_arma_model = arma_model.fit()
            print(trained_arma_model.summary())
            trained_arma_models.append(trained_arma_model)

        return trained_arma_models

    def predict(self, trained_arma_models, len_historical_data: np.array, train: np.array, test: np.array) -> np.array:
        """Make predictions with trained autoregressive moving average models.

        Parameters
        ----------
        trained_arma_models: ARMA models
            Trained autoregressive moving average models
        len_historical_data: `np.array`
            The length of our historical data
        train: `np.array`
            The training data
        test: `np.array`
            The testing data

        Returns
        ------
        predictions: `list`
            A list of predictions for each autoregressive moving average model with each differing by lag value

        """

        predictions = []
        for trained_arma_models_idx in range(len(trained_arma_models)):
            trained_arma_model = trained_arma_models[trained_arma_models_idx]
            print("ARMA(", trained_arma_model, ")")
            model_prediction = trained_arma_model.predict(start=len_historical_data, end=len(train)+len(test)-1, dynamic=False)
            predictions.append(model_prediction)

        return predictions

class ARIMA_model(Model):
    def __name__(self):
        return "ARIMA"

    def train_arima_model(self, train_data: np.array, test_lag_term: int, integrated: int, test_error_term: int) -> list:
        """Initial and train an autoregressive integrated moving average model.

        Parameters
        ----------
        train_data: `np.array`
            Data to train our autoregressive model on
        test_lags: `list`
            A list of lag values to pass to autoregressive model
        test_error_terms: `list`
            A list of error terms to pass to moving average model
        integrated: `int`
            An integer value to difference the TS

        Returns
        ------
        trained_arima_models: `list`
            A list of trained autoregressive integrated moving average models

        """
        trained_arima_models = []

        arima_model = ARIMA(train_data, order=(test_lag_term, integrated, test_error_term))
        trained_arima_model = arima_model.fit()
        print(trained_arima_model.summary())
        trained_arima_models.append(trained_arima_model)

        return trained_arima_models

    def predict(self, trained_arima_models, go: int, stop: int) -> np.array:
        """Make predictions with trained autoregressive integrated moving average models.

        Parameters
        ----------
        trained_arma_models: ARMA models
            Trained autoregressive moving average models
        len_historical_data: `np.array`
            The length of our historical data
        train: `np.array`
            The training data
        test: `np.array`
            The testing data

        Returns
        ------
        predictions: `list`
            A list of predictions for each autoregressive integrated moving average model

        """

        predictions = []

        for trained_arima_models_idx in range(len(trained_arima_models)):
            trained_arima_model = trained_arima_models[trained_arima_models_idx]
            print("ARIMA(", trained_arima_model, ")")
            model_prediction = trained_arima_model.predict(start=go, end=stop, dynamic=False)
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
                mse = mean_squared_error(true_labels, prediction)
                print("expected", true_labels, "predicted", prediction, "mse", mse)
        else:
            mse = mean_squared_error(true_labels, predictions)
            print('Test MSE: %.3f' % mse)

    def eval_rmse(true_labels: np.array, predictions: np.array, per_element=True):
        """Calculate the root mean squared error"""
        if per_element == True:
            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]
                rmse = sqrt(mean_squared_error(true_labels, prediction))
                print("expected", true_labels, "predicted", prediction, "rmse", rmse)
        else:
            rmse = sqrt(mean_squared_error(true_labels, predictions))
            print('Test RMSE: %.3f' % rmse)


    def plot_forecast(true_labels: np.array, predictions: np.array, test_lags: list, with_lags=True):
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
                plt.plot(prediction, color='red', label='Predicted Forecasts', linewidth=1)
        else:
            plt.figure(figsize=(20, 4))
            plt.xlabel("Observations")
            plt.ylabel("Values")
            plt.title(f"Model")

            plt.plot(true_labels, color='blue', label='Actual Forecasts', linewidth=4)
            plt.plot(predictions, color='red', label='Predicted Forecasts', linewidth=1)


        matplotx.line_labels()
        plt.show()
