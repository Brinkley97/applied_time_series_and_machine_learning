"""
Detravious Jamari Brinkley (aka FitToCode)

Factory Pattern: https://refactoring.guru/design-patterns/factory-method/python/example#lang-features
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from abc import ABC
from math import sqrt
from dataclasses import dataclass

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# from constants import Number, TimeSeriesData
# from time_series import UnivariateTimeSeries

from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error, mean_absolute_percentage_error

# Define the abstract base class
@dataclass
class Model(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    def __name__(self):
        pass

    def __init__(self, train_type_name: str, lag_p: int = None, error_q: int = None, integrated_d = None):
        self.train_type_name = self.__name__() + train_type_name
        self.lag_p = lag_p
        self.error_q = error_q
        self.integrated_d = integrated_d

    def make_predictions(self, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame) -> np.array:
        """Make predictions with trained autoregressive moving average model.

        Parameters:
        -----------
        historical_data_df: `pd.DataFrame`
            The data we used to train our model(s)

        y_true_predictions_df: `pd.DataFrame`
            The actual forecasts

        retrain: `bool`
            False --- predicts values for future time points without updating or retraining the model with new data. It simply uses the existing trained model to make predictions.
            True --- predicts values for future time points with updating or retraining the model with new data.  It involves retraining the model using a subset of historical data and possibly other parameters, then making predictions based on the updated model.

        Returns:
        --------
        model_predictions: `np.array`
            A list of predictions

        """

        start_forecasting = len(historical_data_df)  # 24
        end_forecasting = start_forecasting + len(y_true_predictions_df) - 1  # 24 + 7 - 1 = 30

        # Generate predictions for the specified date range [24, 25, 26, 27, 28, 29, 30]
        model_predictions = self.model.predict(start=start_forecasting, end=end_forecasting, dynamic=False)
        return model_predictions
                    
    def model_predictions_to_df(self, y_true_predictions_df, model_predictions):
        all_predictions_df = y_true_predictions_df.copy()
        all_predictions_df[self.train_type_name] = model_predictions
        return all_predictions_df    
        
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

        Parameters:
        -----------
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
class AR_Model(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive model

    Verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    """
    def __name__(self):
        return "AR Model"

    def train(self, train_data_df: pd.DataFrame):
        """Initial and train an autoregressive model.

        Parameters:
        -----------
        train_data_df: `pd.DataFrame`
            Data to train our autoregressive model on

        lag: `int`
            A list of lag values that are over a threshold to pass to autoregressive model

        """
        self.model = AutoReg(train_data_df, lags=self.lag_p).fit()

    def summary(self):
        """Return the summary of the autoregressive model."""
        return self.model.summary()
    
class MA_Model(Model):
    """A class used to initialize, train, and forecast predictions with our moving average model."""

    def __name__(self):
        return "MA Model"

    def summary(self):
        """Return the summary of the autoregressive model."""
        return self.model.summary()
    
    def train(self, train_data_df: pd.DataFrame):
        """Initial and train a moving average model.

        Parameters:
        -----------
        train_data_df: `pd.DataFrame`
            Data to train our moving average model on
        error: `int`
            The error term

        Returns:
        --------
        trained_ma_model: `statsmodels ARIMA model`
            A trained moving average model
        """
        self.model = ARIMA(train_data_df, order=(0, 0, self.error_q)).fit()

class ARMA_Model(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive moving average model.

    Modifying AR code bc we verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    """
    def __name__(self):
        return "ARMA Model"

    def summary(self):
        """Return the summary of the autoregressive model."""
        return self.model.summary()
    
    def train(self, train_data_df: pd.DataFrame):
        """Initial and train an autoregressive moving average model.

        Parameters:
        -----------
        train_data_df: `pd.DataFrame`
            Data to train our autoregressive model on

        lag_p: `int`
            Lag value from Partial Autocorrelation plot

        error_q: `int`
            Error value from Autocorrelation plot

        Returns:
        --------
        trained_arma_model: `statsmodel ARIMA model`
            A single trained autoregressive moving average model

        """

        self.model = ARIMA(train_data_df, order=(self.lag_p, 0, self.error_q), trend="n").fit()

class ARIMA_Model(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive integrated moving average model.

    """
    def __name__(self):
        return "ARIMA Model"

    def train(self, train_data_df: pd.DataFrame):
        """Initial and train an autoregressive integrated moving average model.

        Parameters:
        -----------
        train_data_df: `pd.DataFrame`
            Data to train our autoregressive moving average model

        lag_p: `int`
            Lag value from Partial Autocorrelation plot

        integrated_d: `int`
            Differenced term

        error_q: `int`
            Error value from Autocorrelation plot

        Returns:
        --------
        trained_arma_model: `statsmodel ARIMA model`
            A single trained autoregressive integrated moving average model

        """

        self.model = ARIMA(train_data_df, order=(self.lag_p, self.integrated_d, self.error_q), trend="n").fit()


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

def augment_retrain_predictions(input_to_augment):
    concatenated_output = np.concatenate(input_to_augment)
    # Format the output
    formatted_output = np.array(concatenated_output)
    return formatted_output
    
# Need to rebuild and verify
# class MA(Model):
#     def __name__(self):
#         return "MA"

    # def train_predict_ma_model(self, ts_df: pd.DataFrame, training_data_len: int, testing_data_len: int, window_len: int, test_error_term: int) -> list:
    #     """Initial, train, and predict using the moving average model per https://github.com/marcopeix/TimeSeriesForecastingInPython/blob/master/CH04/CH04.ipynb

    #     Parameters:
    #     -----------
    #     ts_df: `pd.DataFrame`
    #         Data to train our moving average model
    #     training_data_len: `pd.DataFrame`
    #         Length of training data
    #     testing_data_len: `pd.DataFrame`
    #         Length of testing data
    #     window_len: `int`
    #         Length of sliding window for our moving average model
    #     test_error_terms: `int`
    #         A single error term to pass to our moving average model. Formally known as q for MA(q)

    #     Returns:
    #     --------
    #     predicted_forecasts: `list`
    #         A list of predicted forecasts

    #     """
    #     predicted_forecasts = []
    #     total_len = training_data_len + testing_data_len

    #     for i in range(training_data_len, total_len, window_len):

    #         ma_model = SARIMAX(ts_df[:i], order=(0, 0, test_error_term))
    #         trained_ma_model = ma_model.fit(disp=False)
    #         predictions = trained_ma_model.get_prediction(0, i + window_len - 1)
    #         oos_pred = predictions.predicted_mean.iloc[-window_len:]
    #         predicted_forecasts.extend(oos_pred)

    #     return predicted_forecasts

    # class ARIMA_model_old(Model):
    # def __name__(self):
    #     return "ARIMA"

    # def train_arima_model(self, train_data: np.array, test_lag_term: int, integrated: int, test_error_term: int) -> list:
    #     """Initial and train an autoregressive integrated moving average model.

    #     Parameters:
    #     -----------
    #     train_data: `np.array`
    #         Data to train our autoregressive model on
    #     test_lags: `list`
    #         A list of lag values to pass to autoregressive model
    #     test_error_terms: `list`
    #         A list of error terms to pass to moving average model
    #     integrated: `int`
    #         An integer value to difference the TS

    #     Returns:
    #     --------
    #     trained_arima_models: `list`
    #         A list of trained autoregressive integrated moving average models

    #     """
    #     trained_arima_models = []

    #     arima_model = ARIMA(train_data, order=(test_lag_term, integrated, test_error_term))
    #     trained_arima_model = arima_model.fit()
    #     print(trained_arima_model.summary())
    #     trained_arima_models.append(trained_arima_model)

    #     return trained_arima_models

    # def predict(self, trained_arima_models, go: int, stop: int) -> np.array:
    #     """Make predictions with trained autoregressive integrated moving average models on the .

    #     Parameters:
    #     -----------
    #     trained_arma_models: `ARMA models`
    #         Trained autoregressive moving average models
    #     len_historical_data: `np.array`
    #         The length of our historical data
    #     train: `np.array`
    #         The training data
    #     test: `np.array`
    #         The testing data

    #     Returns:
    #     --------
    #     predictions: `list`
    #         A list of predictions for each autoregressive integrated moving average model

    #     """

    #     predictions = []

    #     for trained_arima_models_idx in range(len(trained_arima_models)):
    #         trained_arima_model = trained_arima_models[trained_arima_models_idx]
    #         print("ARIMA(", trained_arima_model, ")")
    #         model_prediction = trained_arima_model.predict(start=go, end=stop, dynamic=False)
    #         predictions.append(model_prediction)

    #     return predictions