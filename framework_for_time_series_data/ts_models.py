import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from abc import ABC
from typing import List
from abc import abstractmethod
from dataclasses import dataclass

from statsmodels.tsa.ar_model import AutoReg

from constants import Number, TimeSeriesData
from time_series import UnivariateTimeSeries

# Define the abstract base class
@dataclass
class Model(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """
    # data: pd.DataFrame


    def augment_data(self):
        pass


class AR(Model):
    def __name__(self):
        return "AR"

    def train_ar_model(self, train_data, lag):
        ar_model = AutoReg(train_data, lags=lag)
        train_ar_model = ar_model.fit()
        train_ar_model.summary()

        return train_ar_model

    def predict(self):
        return "AR Predict"
