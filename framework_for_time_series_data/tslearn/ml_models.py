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
    
    def train(self, X, y, config: list):
        """Train the MLP for #epoch

        Parameters
        ----------
        X: `pd.DataFrame` 
            Input data tensor
        y: `pd.DataFrame`
            Target data tensor
        config: `py list`
            criterion: `torch.nn.Module`
                Loss criterion
            optimize: `torch.optim.Optimizer`
                Optimization algorithm
            epochs: `int` 
                Number of training epochs


        """
        X_train = torch.tensor(X.values, dtype=torch.float32)
        y_train = torch.tensor(y.values, dtype=torch.float32)

        criterion, optimizer, epochs = config

        for epoch in range(epochs):
            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward() # calculate the gradients
            optimizer.step() # update the weights
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
    
    def predict(self, data_tensor: pd.DataFrame, input_size: int):
        """
        data_tensor: `torch.Tensor`
            Test data
        """
        
        yhat = self(data_tensor)  # Perform forward pass
        
        # Extract predicted values for each sample in the batch
        predicted_values = yhat.squeeze(dim=1).tolist()  # Convert tensor to list of predicted values
        print("Predicted Outputs:", predicted_values)
        
        return predicted_values