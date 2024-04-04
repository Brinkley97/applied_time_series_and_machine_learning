import matplotx
import torch

import numpy as np
import pandas as pd

import torch.nn as nn
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

    def forward(self):
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
# verified
class AR(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive model

    Verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    """
    def __name__(self):
        return "AR"

    def train_ar_model(self, train_data_df: pd.DataFrame, threshold_lags: list):
        """Initial and train an autoregressive model.

        Verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

        Parameters
        ----------
        train_data_df: `pd.DataFrame`
            Data to train our autoregressive model on

        threshold_lags: `list`
            A list of lag values that are over a threshold to pass to autoregressive model

        Returns
        ------
        trained_ar_model: `statsmodel AutoReg model`
            A single trained autoregressive models with each differing by lag value

        """

        ar_model = AutoReg(train_data_df, lags=threshold_lags)
        trained_ar_model = ar_model.fit()

        return trained_ar_model

    def predict(self, trained_ar_model, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool, lag_to_test: int = None) -> np.array:
        """Make predictions with trained autoregressive moving average model.

        Parameters
        ----------
        trained_arma_model: AR models
            Trained autoregressive models

        historical_data_df: `pd.DataFrame`
            The data we used to train our model(s)

        y_true_predictions_df: `pd.DataFrame`
            The actual forecasts

        retrain: `bool`
            False --- predicts values for future time points without updating or retraining the model with new data. It simply uses the existing trained model to make predictions.
            True --- predicts values for future time points with updating or retraining the model with new data.  It involves retraining the model using a subset of historical data and possibly other parameters, then making predictions based on the updated model.

        Returns
        -------
        model_predictions: `np.array`
            A list of predictions

        """

        if retrain == False:
            # Example: Days 1 - 30 with forecast of 7 days
            historical_dates = list(historical_data_df.index) # 1 - 23
            y_true_dates = list(y_true_predictions_df.index) # 24 - 30
            print(f"Predictions for dates {y_true_dates}")

            start = len(historical_dates) # 23
            end = start + len(y_true_dates) - 1 # 23 + 7 - 1 - 29

            # verify y_true_dates
            all_dates = historical_dates + y_true_dates # 1 - 30
            prediction_dates = list(all_dates[len(historical_dates):]) # 24 - 30
            # print(f"Predictions for dates {prediction_dates}")

            model_predictions = trained_ar_model.predict(start=start, end=end, dynamic=False)
            return model_predictions

        elif retrain == True:
            start_retrain_idx = len(historical_data_df) - lag_to_test
            history = historical_data_df[start_retrain_idx:].values.tolist()
            for i in range(len(history)):
                history[i] = np.array(history[i])
            test = y_true_predictions_df.values

            predictions = list()
            for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length - lag_to_test, length)]
                coef = trained_ar_model.params

                yhat = coef[0]
                for d in range(lag_to_test):
                    yhat += coef[d+1] * lag[lag_to_test-d-1]
                    obs = test[t]
                predictions.append(yhat)
                history.append(obs)
                print('predicted=%f, expected=%f' % (yhat, obs))

            return predictions
        else:
            print(f"{retrain} is NOT a valid name for retrian")

# Need to rebuild and verify
class MA(Model):
    def __name__(self):
        return "MA"

    def train_predict_ma_model(self, ts_df: pd.DataFrame, training_data_len: int, testing_data_len: int, window_len: int, test_error_term: int) -> list:
        """Initial, train, and predict using the moving average model per https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/CH04/CH04.ipynb

        Parameters
        ----------
        ts_df: `pd.DataFrame`
            Data to train our moving average model
        training_data_len: `pd.DataFrame`
            Length of training data
        testing_data_len: `pd.DataFrame`
            Length of testing data
        window_len: `int`
            Length of sliding window for our moving average model
        test_error_terms: `int`
            A single error term to pass to our moving average model. Formally known as q for MA(q)

        Returns
        ------
        predicted_forecasts: `list`
            A list of predicted forecasts

        """
        predicted_forecasts = []
        total_len = training_data_len + testing_data_len

        for i in range(training_data_len, total_len, window_len):

            ma_model = SARIMAX(ts_df[:i], order=(0, 0, test_error_term))
            trained_ma_model = ma_model.fit(disp=False)
            predictions = trained_ma_model.get_prediction(0, i + window_len - 1)
            oos_pred = predictions.predicted_mean.iloc[-window_len:]
            predicted_forecasts.extend(oos_pred)

        return predicted_forecasts

class ARMA(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive moving average model.

    Modifying AR code bc we verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    """
    def __name__(self):
        return "ARMA"

    def train_arma_model(self, train_data_df: pd.DataFrame, lag_p: int, error_q: int):
        """Initial and train an autoregressive moving average model.

        Parameters
        ----------
        train_data_df: `pd.DataFrame`
            Data to train our autoregressive model on

        lag_p: `int`
            Lag value from Partial Autocorrelation plot

        error_q: `int`
            Error value from Autocorrelation plot

        Returns
        ------
        trained_arma_model: `statsmodel AutoReg model`
            A single trained autoregressive moving average model

        """

        arma_model = ARIMA(train_data_df, order=(lag_p, 0, error_q), trend="n")
        trained_arma_model = arma_model.fit()

        return trained_arma_model

    def predict(self, trained_arma_model, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool, lag_to_test: int = None) -> np.array:
        """Make predictions with trained autoregressive moving average model.

        Parameters
        ----------
        trained_arma_model: AR models
            Trained autoregressive models

        historical_data_df: `pd.DataFrame`
            The data we used to train our model(s)

        y_true_predictions_df: `pd.DataFrame`
            The actual forecasts

        retrain: `bool`
            False --- predicts values for future time points without updating or retraining the model with new data. It simply uses the existing trained model to make predictions.
            True --- predicts values for future time points with updating or retraining the model with new data.  It involves retraining the model using a subset of historical data and possibly other parameters, then making predictions based on the updated model.

        Returns
        -------
        model_predictions: `np.array`
            A list of predictions

        """
        
        if retrain == False:
            # Example: Days 1 - 30 with forecast of 7 days
            historical_dates = list(historical_data_df.index) # 1 - 23
            y_true_dates = list(y_true_predictions_df.index) # 24 - 30
            print(f"Predictions for dates {y_true_dates}")

            start = len(historical_dates) # 23
            end = start + len(y_true_dates) - 1 # 23 + 7 - 1 - 29

            # verify y_true_dates
            all_dates = historical_dates + y_true_dates # 1 - 30
            prediction_dates = list(all_dates[len(historical_dates):]) # 24 - 30
            # print(f"Predictions for dates {prediction_dates}")

            model_predictions = trained_arma_model.predict(start=start, end=end, dynamic=False)
            return model_predictions

        elif retrain == True:
            start_retrain_idx = len(historical_data_df) - lag_to_test
            history = historical_data_df[start_retrain_idx:].values.tolist()
            for i in range(len(history)):
                history[i] = np.array(history[i])
            test = y_true_predictions_df.values

            predictions = list()
            for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length - lag_to_test, length)]
                coef = trained_arma_model.params

                yhat = coef[0]
                for d in range(lag_to_test):
                    yhat += coef[d+1] * lag[lag_to_test-d-1]
                    obs = test[t]
                predictions.append(yhat)
                history.append(obs)
                print('predicted=%f, expected=%f' % (yhat, obs))

            return predictions
        else:
            print(f"{retrain} is NOT a valid name for retrian")

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
        """Make predictions with trained autoregressive integrated moving average models on the .

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

class MLP(nn.Module):
    def __name__(self):
        return "MLP"


    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def augment_data(uts_observations, prior_observations: int, forecasting_step: int) -> pd.DataFrame:
        """Splits a given UTS into multiple input rows where each input row has a specified number of timestamps and the output is a single timestamp.

        Parameters:
        uts_observations -- 1D np array (of UTS data to transform to SML data with size  b rows/length x 1 dimension)
        prior_observations -- py int (of all observations before we get to where we want to start making the predictions)
        forecasting_step -- py int (of how far out to forecast, 1 only the next timestamp, 2 the next two timestamps, ... n the next n timestamps)

        Return:
        agg.values -- np array (of new sml data)
        """

        df = pd.DataFrame(uts_observations)
        cols = list()

        lag_col_names = []
        count_lag = 0
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
        X_train = uts_sml_df.iloc[:, 0: -1]

        # last column
        y_train = uts_sml_df.iloc[:, [-1]]
        return uts_sml_df, X_train, y_train

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@dataclass
class EvaluationMetric:
    """Investigate the philosphy/design behind typing in python.

    https://realpython.com/python-type-checking/
    """
    # Need to rebuild and verify
    def eval_mse(true_predictions_df: pd.DataFrame, model_predictions: np.array, per_element: bool):
        """Calculate the mean squared error

        Verifed with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
        """
        true_predictions = true_predictions_df.values

        if per_element == True:
            for predictions_idx in range(len(model_predictions)):
                prediction = model_predictions[predictions_idx]
                true_prediction = true_predictions[predictions_idx]
                print('predicted=%f, expected=%f' % (prediction, true_prediction))
                mse = mean_squared_error(true_predictions, model_predictions)
                print('Test MSE: %.3f' % mse)
        else:
            mse = mean_squared_error(true_predictions, model_predictions)
            print('Test MSE: %.3f' % mse)

    def eval_rmse(true_predictions_df: pd.DataFrame, model_predictions: np.array, per_element: bool):
        """Calculate the root mean squared error

        Verifed with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
        """
        true_predictions = true_predictions_df.values

        if per_element == True:
            for predictions_idx in range(len(model_predictions)):
                prediction = model_predictions[predictions_idx]
                true_prediction = true_predictions[predictions_idx]
                print('predicted=%f, expected=%f' % (prediction, true_prediction))
                mse = sqrt(mean_squared_error(true_predictions, model_predictions))
                print('Test RMSE: %.3f' % mse)
        else:
            mse = sqrt(mean_squared_error(true_predictions, model_predictions))
            print('Test RMSE: %.3f' % mse)


    def eval_mae(true_predictions_df: pd.DataFrame, model_predictions: np.array, per_element: bool):
        """Calculate the mean absolute error

        Verifed with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error
        """
        true_predictions = true_predictions_df.values

        if per_element == True:
            for predictions_idx in range(len(model_predictions)):
                prediction = model_predictions[predictions_idx]
                true_prediction = true_predictions[predictions_idx]
                print('predicted=%f, expected=%f' % (prediction, true_prediction))
                mae = mean_absolute_error(true_predictions, model_predictions)
                print('Test MAE: %.3f' % mae)
        else:
            mae = mean_absolute_error(true_predictions, model_predictions)
            print('Test MAE: %.3f' % mae)

    def eval_mape(true_predictions_df: pd.DataFrame, model_predictions: np.array, per_element: bool):
        """Calculate the mean absolute percentage error

        Verifed with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error
        """
        true_predictions = true_predictions_df.values

        if per_element == True:
            for predictions_idx in range(len(model_predictions)):
                prediction = model_predictions[predictions_idx]
                true_prediction = true_predictions[predictions_idx]
                print('predicted=%f, expected=%f' % (prediction, true_prediction))
                mape = mean_absolute_percentage_error(true_predictions, model_predictions)
                print('Test MAPE: %.3f' % mape)
        else:
            mape = mean_absolute_percentage_error(true_predictions, model_predictions)
            print('Test MAPE: %.3f' % mape)

    # Need to rebuild and verify
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


    def plot_predictions(true_predictions_df: pd.DataFrame, model_predictions: np.array, lag: int):
        """Plots the in-sample prediction of each model respectively on the same plot.

        Verifed with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
        """

        true_predictions = true_predictions_df.values

        plt.figure(figsize=(20, 4))
        plt.xlabel("Observations")
        plt.ylabel("Values")

        plt.plot(true_predictions, color='blue', label='Actual Forecasts', linewidth=3)
        plt.plot(model_predictions, color='red', label='Predicted Forecasts', linewidth=2)

        matplotx.line_labels()
        plt.show()
