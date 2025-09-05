"""
Detravious Jamari Brinkley (aka FitToCode)

Factory Pattern: https://refactoring.guru/design-patterns/factory-method/python/example#lang-features
"""

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

# @dataclass
# class PlotData:

#     # Need to rebuild and verify
#     def plot_forecast(train_data_df: pd.DataFrame, test_data_df: pd.DataFrame, predictions: np.array, per_element=True):
#         """Plots the forecast of each model respectively on the same plot.

#         Parameters
#         ----------
#         train_data_df: `pd.DataFrame`
#             The data we used to train our model(s)

#         test_data_df: `pd.DataFrame`
#             The actual forecasts

#         predictions: `list`
#             The predicted forecasts
#         """

#         if per_element == True:
#             for predictions_idx in range(len(predictions)):
#                 prediction = predictions[predictions_idx]

#                 plt.figure(figsize=(18, 4))
#                 plt.xlabel("Observations")
#                 plt.ylabel("Values")
#                 plt.title("Forecast")

#                 # Plotting the training data
#                 train_dates = train_data_df.index
#                 train_values = train_data_df.values
#                 plt.plot(train_dates, train_values, color='blue', label='Training Data', linewidth=1)

#                 # Plotting the actual test data
#                 test_dates = test_data_df.index
#                 test_values = test_data_df.values
#                 plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

#                 # Plotting the forecasted values
#                 plt.plot(test_dates, prediction, color='red', label='Predicted Forecasts', linewidth=1)
#         else:
#             plt.figure(figsize=(18, 4))
#             plt.xlabel("Observations")
#             plt.ylabel("Values")
#             plt.title(f"Forecast")

#             # Plotting the training data
#             train_dates = train_data_df.index
#             train_values = train_data_df.values
#             plt.plot(train_dates, train_values, color='blue', label='Training Data', linewidth=1)

#             # Plotting the actual test data
#             test_dates = test_data_df.index
#             test_values = test_data_df.values
#             plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

#             # Plotting the forecasted values
#             plt.plot(test_dates, predictions, color='red', label='Predicted Forecasts', linewidth=1)

#         matplotx.line_labels()
#         plt.show()

#     def plot_forecast_only(test_data_df: pd.DataFrame, predictions: np.array, per_element=True):
#         """Plots the forecast of each model respectively on the same plot.

#         Parameters
#         ----------
#         test_data_df: `pd.DataFrame`
#             The actual forecasts

#         predictions: `list`
#             The predicted forecasts
#         """

#         if per_element == True:
#             for predictions_idx in range(len(predictions)):
#                 prediction = predictions[predictions_idx]

#                 plt.figure(figsize=(18, 4))
#                 plt.xlabel("Observations")
#                 plt.ylabel("Values")
#                 plt.title("Forecast")

#                 # Plotting the actual test data
#                 test_dates = test_data_df.index
#                 test_values = test_data_df.values
#                 plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

#                 # # Plotting the forecasted values
#                 plt.plot(test_dates[predictions_idx], prediction, color='red', label='Predicted Forecasts', linewidth=2)
                
#         else:
#             plt.figure(figsize=(18, 4))
#             plt.xlabel("Observations")
#             plt.ylabel("Values")
#             plt.title("Forecast")

#             # Plotting the actual test data
#             test_dates = test_data_df.index
#             test_values = test_data_df.values
#             plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

#             # Plotting the forecasted values
#             plt.plot(test_dates, predictions, color='red', label='Predicted Forecasts', linewidth=2)

#         matplotx.line_labels()
#         plt.show()


#     def plot_predictions(true_predictions_df: pd.DataFrame, model_predictions: np.array):
#         """Plots the in-sample prediction of each model respectively on the same plot.

#         Verifed with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#         """

#         true_predictions = true_predictions_df.values

#         plt.figure(figsize=(20, 4))
#         plt.xlabel("Observations")
#         plt.ylabel("Values")

#         plt.plot(true_predictions, color='blue', label='True Values', linewidth=3)
#         plt.plot(model_predictions, color='red', label='Predicted Values', linewidth=2)      

#         matplotx.line_labels()
#         plt.show()

import torch

import numpy as np
import pandas as pd

import torch.nn as nn
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class PlotData:
    """
    Data structure for plot data.

    Attributes
    ----------
    epoch_count: `List[int]`
        List of integers representing epochs.

    train_loss_values_df: `pd.DataFrame`
        DataFrame of training loss values.

    test_loss_values_df: `pd.DataFrame`
        DataFrame of test loss values.
    """
    epoch_count: List[int]
    train_loss_values_df: pd.DataFrame
    test_loss_values_df: pd.DataFrame

class Plotter(ABC):
    @abstractmethod
    def plot(self):
        pass

class InterpolatePlotter(Plotter):
    def __init__(self, true_predictions_df: pd.DataFrame = None, predictions_dict: dict = None):
        self.true_predictions_df = true_predictions_df
        self.predictions_dict = predictions_dict

    def plot_in_sample_predictions(self, scatter_type: bool):
        """
        Plots the in-sample predictions.

        Need to verify with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
        """
        if self.true_predictions_df is not None and self.predictions_dict:
            true_predictions = self.true_predictions_df.values

            plt.figure(figsize=(14, 8))
            plt.xlabel("Observations")
            plt.ylabel("Values")

            if scatter_type:
                plt.scatter(range(len(true_predictions)), true_predictions, color='blue', label='True Values', s=10)

                for label, predictions in self.predictions_dict.items():
                    plt.scatter(range(len(predictions)), predictions, label=label, s=10)
            else:
                plt.plot(true_predictions, color='blue', label='True Values', linewidth=3)

                for label, predictions in self.predictions_dict.items():
                    plt.plot(predictions, label=label, linewidth=2)

            plt.legend()
            plt.show()

    def plot_ts_training_and_testing_data(self,
                                          train_data_df: pd.DataFrame, 
                                          test_data_df: pd.DataFrame,
                                          scatter_type: bool, 
                                          predictions_df: pd.DataFrame = None):
        """
        Plots training data, test data, and compares predictions.
        """

        plt.figure(figsize=(7, 7))
        plt.xlabel("Observations")
        plt.ylabel("Values")

        train_idx = train_data_df.index
        train_values = train_data_df.iloc[:, 0].values
        test_idx = test_data_df.index
        test_values = test_data_df.iloc[:, 0].values

        if scatter_type:
            plt.scatter(train_idx, train_values, c="b", s=4, label="Training data")
            plt.scatter(test_data_df.index, test_data_df.iloc[:, 0].values, c="g", s=10, label="Testing data")

            if predictions_df is not None:
                prediction_values = predictions_df.values
                plt.scatter(test_data_df.values, prediction_values, c="r", s=4, label="Predictions")

        else:
            plt.plot(train_idx, train_values, c="b", label="Training data")
            plt.plot(test_idx, test_values, c="g", label="Testing data")
            
            if predictions_df is not None:
                    prediction_values = predictions_df.values
                    plt.plot(test_idx, prediction_values, c="r", label="Predictions")
            
        plt.legend(prop={"size": 14})
        plt.show()
    
    def plot_ml_training_and_testing_data(self, train_data_df: pd.DataFrame, train_labels_df: pd.DataFrame, test_data_df: pd.DataFrame, test_labels_df: pd.DataFrame, predictions=None):
        """
        Plots training data, test data, and compares predictions.
        """
        plt.figure(figsize=(7, 7))
        plt.scatter(train_data_df.values, train_labels_df.values, c="b", s=4, label="Training data")
        plt.scatter(test_data_df.values, test_labels_df.values, c="g", s=10, label="Testing data")

        if predictions is not None:
            plt.scatter(test_data_df.values, predictions, c="r", s=4, label="Predictions")
        
        plt.legend(prop={"size": 14})
        plt.show()

    def plot(self):
        """
        Default plot method to satisfy the abstract base class requirement.
        """
        self.plot_in_sample_predictions()

class ExtrapolatePlotter(Plotter):
    def __init__(self, test_data_df: pd.DataFrame, predictions: np.array, train_data_df: pd.DataFrame = None, per_element=True):
        self.train_data_df = train_data_df
        self.test_data_df = test_data_df
        self.predictions = predictions
        self.per_element = per_element

    def plot(self):
        """
        Plots the out-sample forecasts. Optionally includes training data if provided.
        """
        if self.per_element:
            for predictions_idx in range(len(self.predictions)):
                prediction = self.predictions[predictions_idx]

                plt.figure(figsize=(7, 7))
                plt.xlabel("Observations")
                plt.ylabel("Values")
                plt.title("Forecast")

                # Optionally plot training data
                if self.train_data_df is not None:
                    train_dates = self.train_data_df.index
                    train_values = self.train_data_df.values
                    plt.plot(train_dates, train_values, color='blue', label='Training Data', linewidth=1)

                # Plotting the actual test data
                test_dates = self.test_data_df.index
                test_values = self.test_data_df.values
                plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

                # Plotting the forecasted values
                plt.plot(test_dates, prediction, color='red', label='Predicted Forecasts', linewidth=1)
        else:
            plt.figure(figsize=(7, 7))
            plt.xlabel("Observations")
            plt.ylabel("Values")
            plt.title("Forecast")

            # Optionally plot training data
            if self.train_data_df is not None:
                train_dates = self.train_data_df.index
                train_values = self.train_data_df.values
                plt.plot(train_dates, train_values, color='blue', label='Training Data', linewidth=1)

            # Plotting the actual test data
            test_dates = self.test_data_df.index
            test_values = self.test_data_df.values
            plt.plot(test_dates, test_values, color='green', label='Actual Forecasts', linewidth=4)

            # Plotting the forecasted values
            plt.plot(test_dates, self.predictions, color='red', label='Predicted Forecasts', linewidth=1)

        plt.legend()
        plt.show()

class PlotFactory:
    @staticmethod
    def create_plotter(plot_type: str, **kwargs) -> Plotter:
        if plot_type == 'interpolate':
            return InterpolatePlotter(kwargs.get('true_predictions_df'), kwargs.get('model_predictions'))
        elif plot_type == 'extrapolate':
            return ExtrapolatePlotter(kwargs['test_data_df'], kwargs['predictions'], kwargs.get('train_data_df'), kwargs.get('per_element', True))
        elif plot_type == 'loss_curve':
            return LossCurvePlotter(PlotData(**kwargs))
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def plot_bandpass(base_ts, 
                      limit: int,
                      data_units_xaxis,
                      data_units_yaxis,
                      pre_or_post: str,
                      data_name: str = 'ECG'
                      ):
        """Plot the peaks.

        Parameters
        ---------- 
        base_ts: str
            The uts/mts before/after bandpass.
            If before, same uts/mts passed in bandpass_filter method
            If after, get the uts/mts returned from the bandpass_filter method
        limit: int
            At most to plot to reduce data
        data_units_xaxis: str
            What to name x-axis
        data_units_yaxis: str
            What to name y-axis
        pre_or_post:
            See base_ts to determine if this is before/after bandpass.
        data_name: str
            What type of data
        
        Return
        ------
        None
            Only/Go ahead show plots

        Notes
        -----
        See time_series.py | UnivariateTimeSeries | bandpass_filter()
        See time_series.py | UnivariateTimeSeries | detect_peak()
            For ECG data, after bandpass_filter(), call detect_peak()
        """
        ts = base_ts.get_as_df().to_numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        if limit:
            plt.plot(ts[:limit], label=f'{data_name} {pre_or_post} Bandpass')
        else:
            plt.plot(ts, label=f'{data_name} {pre_or_post} Bandpass')  
        plt.title('Bandpass Filter')
        plt.xlabel(data_units_xaxis)
        plt.ylabel(data_units_yaxis)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_peaks(base_ts, 
                   peaks: dict,
                   detection_name: str, 
                   sampling_rate: float,
                   data_units_xaxis: str,
                   data_units_yaxis: str,
                   data_name: str = 'ECG'
                   ):
        """Plot the peaks.

        Parameters
        ---------- 
        base_ts: str
            The uts/mts before peaks.
            Same uts/mts passed as peaks method
        peaks: dict
            The peak_algo_name : peak_values
        detection_name: str
            Select the detection name (peak_algo_name) that relates to the type of peak detection algo we want to implement.
        sampling_rate: float
            The sampling frequency of the signal. 
            See update_with_sampling_rate() and avg_down_sample() for explanations.
        data_units_xaxis: str
            What to name x-axis
        data_units_yaxis: str
            What to name y-axis
        data_name: str
            What type of data
        
        Return
        ------
        None
            Only/Go ahead show plots

        Notes
        -----
        See time_series.py | UnivariateTimeSeries | detect_peak()
        See time_series.py | UnivariateTimeSeries | bandpass_filter()
            Can run bandpass_filter() before detect_peak()
        Some of these are used for ecg health data
        Follow: https://www.samproell.io/posts/signal/ecg-library-comparison/#benchmark-results
        """
        import wfdb  # pip install wfdb

        # less fancy: plt.plot(ecg_signal); plt.plot(rpeaks, ecg_signal[rpeaks], "x")
        wfdb.plot_items(
            base_ts.values,
            [peaks[detection_name]],
            fs=sampling_rate,
            sig_name=[data_name],
            sig_units=[data_units_yaxis],
            time_units=data_units_xaxis,
            return_fig=True,
            ann_style="o",
        )
        
class LossCurvePlotter:
    def __init__(self, plot_data: PlotData):
        """
        Initializes the LossCurvePlotter with plot data.

        Parameters
        ----------
        plot_data: `PlotData`
            Object containing epoch count, training loss values, and test loss values.
        """
        self.epoch_count = plot_data.epoch_count
        self.train_loss_values_df = plot_data.train_loss_values_df
        self.test_loss_values_df = plot_data.test_loss_values_df

    def plot(self):
        """
        Plots the training and testing loss curves.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.epoch_count, self.train_loss_values_df, label="Train loss")
        plt.plot(self.epoch_count, self.test_loss_values_df, label="Test loss")
        plt.title("Train & Test Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()



