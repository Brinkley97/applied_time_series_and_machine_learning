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

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)
        return fc2_out
    
    def train(self, input_size: int, hidden_size: int, output_size: int, config: list):
        """Train the MLP for #epoch

        """
        criterion, optimizer = config

        X = torch.randn(hidden_size, input_size)  # Example input data
        y = torch.randn(hidden_size, output_size)  # Example target data

        epochs = 2000
        for epoch in range(epochs):
            # Forward pass
            outputs = self(X)
            loss = criterion(outputs, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
    
    def predict(self, data_df: pd.DataFrame, input_size: int):
        """
        """

        x_input_df = data_df.iloc[[-1], -input_size:].copy()
        x_input_tensor = torch.tensor(x_input_df.values, dtype=torch.float32)
        print("x_input_tensor shape:", x_input_tensor.shape, x_input_tensor)

        yhat = self(x_input_tensor)  # Perform forward pass
        
        # Extract predicted values for each sample in the batch
        predicted_values = yhat.squeeze(dim=1).tolist()  # Convert tensor to list of predicted values
        print("Predicted Outputs:", predicted_values)
        
        return predicted_values
