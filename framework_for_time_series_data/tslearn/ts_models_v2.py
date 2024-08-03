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

    def train(self):
        pass

    def predict(self):
        pass

    def summary(self):
        pass

    def __name__(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        pass

    def get_prediction_indices(self, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame):
        """Get start and end indices for predictions based on historical and true prediction data.

        Parameters:
        -----------
        historical_data_df: `pd.DataFrame`
            The historical data used to train our model(s)

        y_true_predictions_df: `pd.DataFrame`
            The actual forecasts

        Returns:
        --------
        start: `int`
            The start index for predictions

        end: `int`
            The end index for predictions

        y_true_dates: `list`
            The dates for the true predictions
        """
        historical_dates = list(historical_data_df.index)
        y_true_dates = list(y_true_predictions_df.index)
        start = len(historical_dates)
        end = start + len(y_true_dates) - 1
        return start, end, y_true_dates

    def ensure_datetime_index(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a datetime index with a frequency."""
        if not isinstance(data_df.index, pd.DatetimeIndex) or data_df.index.freq is None:
            data_df.index = pd.date_range(start=data_df.index[0], periods=len(data_df), freq='D')
        return data_df

    def make_predictions(self, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool) -> np.array:
        """Make predictions with trained model.

        Parameters:
        -----------
        historical_data_df: `pd.DataFrame`
            Data used to train the model.
        y_true_predictions_df: `pd.DataFrame`
            Data for which predictions are made.
        retrain: `bool`
            If False, uses the existing model for predictions.
            If True, retrains the model on the historical data and then makes predictions.

        Returns:
        --------
        model_predictions: `np.array`
            Array of predictions.
        """
        historical_data_df = self.ensure_datetime_index(historical_data_df)
        y_true_predictions_df = self.ensure_datetime_index(y_true_predictions_df)

        if not retrain:
            # Use existing model for predictions
            historical_dates = list(historical_data_df.index)
            y_true_dates = list(y_true_predictions_df.index)
            start = len(historical_dates)
            end = start + len(y_true_dates) - 1
            model_predictions = self.model.predict(start=start, end=end, dynamic=False)
            return model_predictions

        else:
            # Retrain the model using the historical data
            start_retrain_idx = len(historical_data_df) - self.error_q
            
            # Prepare the historical data for retraining
            history = historical_data_df[start_retrain_idx:].values.tolist()
            history_array = []
            for data_point in history:
                history_array.append(np.array(data_point))
            
            # Prepare the true values for predictions
            test_values = y_true_predictions_df.values
            
            predictions = []
            
            # Iterate over each test value to make predictions
            for test_index in range(len(test_values)):
                history_length = len(history_array)
                
                # Extract the lagged values for the prediction
                lags = []
                for i in range(history_length - self.error_q, history_length):
                    lags.append(history_array[i])
                
                # Retrieve model coefficients
                coefficients = self.model.params
                
                # Calculate the predicted value
                predicted_value = coefficients[0]
                for error_term_index in range(self.error_q):
                    # Retrieve the coefficient for the current lagged value
                    coefficient = coefficients[error_term_index + 1]
                    
                    # Compute the index to access the correct lagged value
                    lag_index = -(error_term_index + 1)
                    
                    # Retrieve the lagged value from the history
                    lagged_value = lags[lag_index]
                    
                    # Update the predicted value by adding the contribution of the current lagged value
                    predicted_value += coefficient * lagged_value
                                
                # Append the actual value to the history and store the prediction
                actual_value = test_values[test_index]
                predictions.append(predicted_value)
                history_array.append(actual_value)
            
        return predictions

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
        return "AR_Model"
    
    def __init__(self, lag_p: int):
        super().__init__()
        self.lag_p = lag_p

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

    def make_predictions(self, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool) -> np.array:
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
        # Call the parent class's make_predictions method
        return super().make_predictions(historical_data_df, y_true_predictions_df, retrain)

class MA_Model(Model):
    """A class used to initialize, train, and forecast predictions with our moving average model."""

    def __name__(self):
        return "MA_Model"
    
    def __init__(self, error_q):
        super().__init__()
        self.error_q = error_q

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

    def make_predictions(self, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool) -> np.array:
        """Make predictions with a trained moving average model.

        Parameters:
        -----------
        historical_data_df: `pd.DataFrame`
            The data used to train the model
        y_true_predictions_df: `pd.DataFrame`
            The actual forecasts
        retrain: `bool`
            False - predict values without updating or retraining the model
            True - predict values with updating or retraining the model

        Returns:
        --------
        model_predictions: `np.array`
            A list of predictions
        """
        # Call the parent class's make_predictions method
        return super().make_predictions(historical_data_df, y_true_predictions_df, retrain)

class ARMA(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive moving average model.

    Modifying AR code bc we verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    """
    def __name__(self):
        return "ARMA"
    
    def __init__(self, lag_p, error_q):
        self.lag_p = lag_p
        self.error_q = error_q

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

    def make_predictions(self, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool) -> np.array:
        """Make predictions with trained autoregressive moving average model.

        Parameters:
        -----------
        trained_arma_model: AR models
            Trained autoregressive moving average model

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
        # Call the parent class's make_predictions method
        return super().make_predictions(historical_data_df, y_true_predictions_df, retrain)

class ARIMA_model(Model):
    """A class used to initialize, train, and forecast predictions with our autoregressive integrated moving average model.

    Modifying AR code bc we verified with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

    """
    def __name__(self):
        return "ARMA"

    def train_arima_model(self, train_data_df: pd.DataFrame, lag_p: int, integrated_d : int, error_q: int):
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

        arima_model = ARIMA(train_data_df, order=(lag_p, integrated_d, error_q), trend="n")
        trained_arima_model = arima_model.fit()

        return trained_arima_model

    def make_predictions(self, trained_arima_model, historical_data_df: pd.DataFrame, y_true_predictions_df: pd.DataFrame, retrain: bool, lag_to_test: int = None) -> np.array:
        """Make predictions with trained autoregressive integrated moving average model.

        Parameters:
        -----------
        trained_arima_model: ARIMA models
            Trained autoregressive integrated moving average model

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
        # Call the parent class's make_predictions method
        return super().make_predictions(historical_data_df, y_true_predictions_df, retrain)

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
