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

class MLP(nn.Module):
    def __name__(self):
        return "MLP"

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)
        return fc2_out