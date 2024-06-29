import torch

import pandas as pd

import torch.nn as nn
import matplotlib.pyplot as plt

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

# Define the abstract base class
@dataclass
class ML_MODELS(ABC, nn.Module):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    def __name__(self):
        return "ML MODEL BASE"
    
    @abstractmethod
    def forward():
        pass
    
    def train_model(self, X: pd.DataFrame, y: pd.DataFrame, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
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
        """

        X_train = torch.tensor(X, requires_grad=True, dtype=torch.float32)
        y_train = torch.tensor(y, requires_grad=True, dtype=torch.float32)

        # Set model to training mode
        self.train()  # Set model to training model which sets all parameters that require gradients to require gradients

        # 1. Forward pass
        y_preds = self.forward(X_train)

        # 2. Loss
        train_loss = loss_fn(y_preds, y_train)

        # 3. Optimizer zero grad to erase or to zero out gradiens to between 0 - 1
        # Get a fresh start every epoch instead of increasing every time... 1, 2, 3, ...
        optimizer.zero_grad()

        # 4. Backward pass- Backpropagation
        train_loss.backward()

        # 5. Steps the optimizers (perform gradient descent)
        optimizer.step()

        print(self.state_dict())

class MLP(ML_MODELS):
    def __name__(self):
        return "MLP"

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
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
        X_train = torch.tensor(X.values, require_grad=True, dtype=torch.float32)
        y_train = torch.tensor(y.values, require_grad=True, dtype=torch.float32)

        criterion, optimizer, epochs = config

        for epoch in range(epochs):
            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward() # calculate the gradients
            optimizer.step() # update the weights
            
            if (epoch+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
    
    def predict(self, X_test_df: pd.DataFrame, input_size: int):
        """
        data_tensor: `torch.Tensor`
            Test data
        """
        data_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
        yhat = self(data_tensor)  # Perform forward pass
        
        # Extract predicted values for each sample in the batch
        predicted_values = yhat.squeeze(dim=1).tolist()  # Convert tensor to list of predicted values
        print("Predicted Outputs:", predicted_values)
        
        return predicted_values

class CNN(nn.Module):
    def __name__(self):
        return "CNN"
    
    def __init__(self, previous_steps: int, hidden_size: int, N_filters: int, kernel_size: int, activation_type: str, n_variables: int, pool_size: int, forecast_steps: int):
        self.fc1 = nn.Linear(previous_steps, hidden_size)
        self.conv_1d = nn.Conv1D(filter=N_filters, kernel_size=kernel_size, activation_type=activation_type)
        self.max_pool = nn.MaxPool1d(pool_size)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(hidden_size, forecast_steps)
        self.fc3 = nn.Linear()

        def forward(self, x):
            fc1_out = self.fc1(x)
            conv_1d_out = self.conv_1d(fc1_out)
            pool_out = self.max_pool(conv_1d_out)
            flatten_out = self.flatten(pool_out)
            fc2_out = self.fc2(flatten_out)
            fc3_out = self.fc3(fc2_out)
            return fc3_out