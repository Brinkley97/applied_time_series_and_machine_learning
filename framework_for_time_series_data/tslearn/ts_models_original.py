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
from statsmodels.tsa.statespace.sarimax import SARIMAX


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


    def predict(self):
        pass

class RandomWalk(Model):
    def __name__(self):
        return "Persistence Walk Forward"

    def predict(self, train_raw_x: pd.DataFrame, test_raw_y: pd.DataFrame):
        """Make predictions with the Random Walk Model using the raw data. We use the raw data because we know there's a dependence of the current observation on the previous observation. We're able to capture the overall direction of the data.

        Formally stated by Jason Brownlee in: https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/
            'We can expect that the best prediction we could make would be to use the observation at the previous time step as what will happen in the next time step. Simply because we know that the next time step will be a function of the prior time step.'

        With this, we don't difference nor do we get the returns.

        Parameters
        ----------
        train_raw_x: `pd.DataFrame`
            The raw train data
         test_raw_y: `pd.DataFrame`
            The raw test data

        """
        # Convert to array
        train_values = train_raw_x.values

        # Convert to array
        test_values = test_raw_y.values

        predictions = list()

        # Get last value in training set
        history = train_values[-1]
        # print(history)

        for i in range(len(test_values)):
            # Set our predicted value to the last value, hence us depending on the previous observation
            yhat = history
            predictions.append(yhat)

            # Set history value to the testing/true value at this current index
            history = test_values[i]

        return predictions

# Need to verify
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
    """A class used to initialize, train, and forecast predictions with our autoregressive model

    Methods
    -------
    train_ar_model(train_data_df: pd.DataFrame, threshold_lags: list)
        Initial and train an autoregressive model
    predict(trained_ar_model, train_data_df: pd.DataFrame, test_data_df: pd.DataFrame)
        Make predictions with trained autoregressive models.

    """
    def __name__(self):
        return "AR"

    def train_ar_model(self, train_data_df: pd.DataFrame, threshold_lags: list):
        """Initial and train an autoregressive model.

        Parameters
        ----------
        train_data_df: `pd.DataFrame`
            Data to train our autoregressive model on
        threshold_lags: `list`
            A list of lag values that are over a threshold to pass to autoregressive model

        Returns
        ------
        trained_ar_model: `statsmodel AutoReg model`
            A list of trained autoregressive models with each differing by lag value

        """

        ar_model = AutoReg(train_data_df, lags=threshold_lags)
        trained_ar_model = ar_model.fit()

        return trained_ar_model

    def predict(self, trained_ar_model, train_data_df: pd.DataFrame, test_data_df: pd.DataFrame) -> list:
        """Make predictions with trained autoregressive models.

        Parameters
        ----------
        trained_ar_models: AR models
            Trained autoregressive models

        train_data_df: `pd.DataFrame`
            The data we used to train our model(s)

        test_data_df: `pd.DataFrame`
            The actual forecasts

        Returns
        ------
        predictions: `list`
            A list of predictions for each autoregressive model with each differing by lag value

        """
        # This is correct. Example: Days 1, 2, 3, ..., 10. We want to predict day 8, 9, and 10. We train on days 1, 2, ..., 7. We test on days 8, 9, and 10. Start is length of historical data, here 7. End is 7 + 3 - 1 = 9. So, our model will make predictions from 7, 8, 9?
        start = len(train_data_df)
        end = start + len(test_data_df) - 1

        predictions = []

        model_prediction = trained_ar_model.predict(start=start, end=end, dynamic=False)
        predictions.append(model_prediction)

        return predictions

# Need to rebuild and verify
class MA(Model):
    def __name__(self):
        return "MA"

    def train_ma_model(self, df, train_data_df: pd.DataFrame, horizon: int, test_error_term: int, window: int):
        """Initial and train an moving average model.

        Parameters
        ----------
        train_data: `pd.DataFrame`
            Data to train our autoregressive model on
        test_error_terms: `list`
            A list of error terms to pass to moving average model

        Returns
        ------
        trained_ma_models: `list`
            A list of trained moving average models with each differing by the moving average value we provide

        """

        # trained_ma_models = []
        # for test_error_terms_idx in range(len(test_error_terms)):
        #     test_error_term = test_error_terms[test_error_terms_idx]
        #     print("MA(", test_error_term, ")")
        #
        #     ma_model = ARIMA(train_data_df, order=(0, 0, test_error_term))
        #     trained_ma_model = ma_model.fit()
        #     print(trained_ma_model.summary())
        #     print()
        #     trained_ma_models.append(trained_ma_model)
        # train_len = len(train_data_df)
        total_len = train_data_df + horizon
        pred_MA = []

        for i in range(train_data_df, total_len, window):
            ma_model = SARIMAX(df[:i], order=(0,0,test_error_term))
            trained_ma_model = ma_model.fit(disp=False)
            # book way
            predictions = trained_ma_model.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)

        return trained_ma_model, pred_MA

    def predict_ma(self, trained_ma_model, train_data_df, horizon, window: int) -> list:
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

        # model_prediction = trained_ma_model.predict(start=start, end=end, dynamic=False)
        # predictions.append(model_prediction)

        # return predictions

        # start = len(train_data_df)
        # end = start + len(test_data_df) - 1
        #
        # predictions = []

        # predictions = []
        # for trained_ma_models_idx in range(len(trained_ma_models)):
        #     trained_ma_model = trained_ma_models[trained_ma_models_idx]
        #     print("MA(", trained_ma_model, ")")
        #     model_prediction = trained_ma_model.predict(start=start, end=end, dynamic=False)
        #     predictions.append(model_prediction)
        #
        # return predictions
        total_len = train_data_df + horizon
        pred_MA = []

        for i in range(train_data_df, total_len, window):
            predictions = trained_ma_model.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)

        return pred_MA

# Need to rebuild and verify
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
        trained_arma_models: `ARMA models`
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
    # Need to rebuild and verify
    def eval_mse(true_labels: pd.DataFrame, predictions: list, per_element=True):
        """Calculate the mean squared error"""
        if per_element == True:

            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]
                mse = mean_squared_error(true_labels, prediction)
                print('Test MSE: %.3f' % mse)
        else:
            mse = mean_squared_error(true_labels, predictions)
            print('Test MSE: %.3f' % mse)

    # Need to rebuild and verify
    def eval_rmse(true_labels: np.array, predictions: np.array, per_element=True):
        """Calculate the root mean squared error"""
        if per_element == True:
            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]
                rmse = sqrt(mean_squared_error(true_labels, prediction))
                print('Test RMSE: %.3f' % rmse)
        else:
            rmse = sqrt(mean_squared_error(true_labels, predictions))
            print('Test RMSE: %.3f' % rmse)


    def plot_forecast(train_data_df: pd.DataFrame, test_data_df: pd.DataFrame, predictions: np.array, per_element=True):
        """Plots the forecast of each model respectively on the same plot.

        Parameters
        ----------
        train_data_df: `pd.DataFrame`
            The data we used to train our model(s)

        test_data_df: `pd.DataFrame`
            The actual forecasts

        predictions: `list`
            The predicted forecasts
        """

        if per_element == True:
            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]

                plt.figure(figsize=(18, 4))
                plt.xlabel("Observations")
                plt.ylabel("Values")
                plt.title("Forecast")

                # Plotting the training data
                train_dates = train_data_df.index
                train_values = train_data_df.values
                plt.plot(train_dates, train_values, color='blue', label='Training Data', linewidth=1)

                # Plotting the actual test data
                test_dates = test_data_df.index
                test_values = test_data_df.values
                plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

                # Plotting the forecasted values
                plt.plot(test_dates, prediction, color='red', label='Predicted Forecasts', linewidth=1)
        else:
            plt.figure(figsize=(18, 4))
            plt.xlabel("Observations")
            plt.ylabel("Values")
            plt.title(f"Forecast")

            # Plotting the training data
            train_dates = train_data_df.index
            train_values = train_data_df.values
            plt.plot(train_dates, train_values, color='blue', label='Training Data', linewidth=1)

            # Plotting the actual test data
            test_dates = test_data_df.index
            test_values = test_data_df.values
            plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

            # Plotting the forecasted values
            plt.plot(test_dates, predictions, color='red', label='Predicted Forecasts', linewidth=1)

        matplotx.line_labels()
        plt.show()

    def plot_forecast_only(test_data_df: pd.DataFrame, predictions: np.array, per_element=True):
        """Plots the forecast of each model respectively on the same plot.

        Parameters
        ----------
        test_data_df: `pd.DataFrame`
            The actual forecasts

        predictions: `list`
            The predicted forecasts
        """

        if per_element == True:
            for predictions_idx in range(len(predictions)):
                prediction = predictions[predictions_idx]

                plt.figure(figsize=(18, 4))
                plt.xlabel("Observations")
                plt.ylabel("Values")
                plt.title("Forecast")

                # Plotting the actual test data
                test_dates = test_data_df.index
                test_values = test_data_df.values
                plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

                # Plotting the forecasted values
                plt.plot(test_dates, prediction, color='red', label='Predicted Forecasts', linewidth=2)
        else:
            plt.figure(figsize=(18, 4))
            plt.xlabel("Observations")
            plt.ylabel("Values")
            plt.title("Forecast")

            # Plotting the actual test data
            test_dates = test_data_df.index
            test_values = test_data_df.values
            plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

            # Plotting the forecasted values
            plt.plot(test_dates, predictions, color='red', label='Predicted Forecasts', linewidth=2)

        matplotx.line_labels()
        plt.show()
