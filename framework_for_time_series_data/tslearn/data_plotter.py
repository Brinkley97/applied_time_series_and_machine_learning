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
    def __init__(self, true_predictions_df: pd.DataFrame = None, model_predictions: np.array = None):
        self.true_predictions_df = true_predictions_df
        self.model_predictions = model_predictions

    def plot_in_sample_predictions(self):
        """
        Plots the in-sample predictions.

        Need to verify with https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
        """
        if self.true_predictions_df is not None and self.model_predictions is not None:
            true_predictions = self.true_predictions_df.values

            plt.figure(figsize=(7, 7))
            plt.xlabel("Observations")
            plt.ylabel("Values")

            plt.scatter(range(len(true_predictions)), true_predictions, color='blue', label='True Values', s=10)
            plt.scatter(range(len(self.model_predictions)), self.model_predictions, color='red', label='Predicted Values', s=10)

            plt.legend()
            plt.show()
        else:
            print("In-sample predictions or true predictions are not available.")

    def plot_training_and_testing_data(self, train_data_df: pd.DataFrame, train_labels_df: pd.DataFrame, test_data_df: pd.DataFrame, test_labels_df: pd.DataFrame, predictions=None):
        """
        Plots training data, test data, and compares predictions.
        """
        plt.figure(figsize=(7, 7))
        plt.scatter(train_data_df.values, train_labels_df.values, c="b", s=4, label="Training data")
        plt.scatter(test_data_df.values, test_labels_df.values, c="g", s=4, label="Testing data")

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