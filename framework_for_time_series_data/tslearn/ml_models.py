"""
Detravious Jamari Brinkley (aka FitToCode)

Factory Pattern: https://refactoring.guru/design-patterns/factory-method/python/example#lang-features
"""

import os
import torch

import pandas as pd

import tkinter as tk
import torch.nn as nn
import matplotlib.pyplot as plt

from abc import ABC
from typing import List
from torchviz import make_dot
from abc import abstractmethod
from IPython.display import Image

from tkinter.filedialog import askopenfilenames, askdirectory


# TSLearn
from data_loader import create_file_version
# from time_series import df_to_tensor


# Define the abstract base class
class Model(ABC, nn.Module):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented; if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    def __name__(self):
        return "ML MODEL BASE"
    
    def __init__(self):
        super().__init__()

    @abstractmethod # Method is required in sub classes as it'll differ per sub class
    def forward_pass(self, x):
        pass
    
    # Method is NOT required in sub classes as it's the same for all sub classes
    def train_model(self, X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, config: list) -> tuple[list]:
        """Train all models #epoch

        Parameters
        ----------
        X_train_df: `pd.DataFrame` 
            Input data tensor

        y_train_df: `pd.DataFrame`
            Target data tensor

        config: `py list`
            criterion: `torch.nn.Module`
                Loss criterion

            optimize: `torch.optim.Optimizer`
                Optimization algorithm
        
        Return
        ------
        tuple[list]: 
            train_y_preds: The model's train predictions
            train_loss: The loss evaluation metric
        """

        X_train = torch.tensor(X_train_df.values, requires_grad=True, dtype=torch.float32)
        y_train = torch.tensor(y_train_df.values, requires_grad=True, dtype=torch.float32)

        loss_fn, optimizer = config

        # Set model to training mode
        self.train()  # Set model to training model which sets all parameters that require gradients to require gradients

        # 1. Forward pass
        train_y_preds = self.forward_pass(X_train)

        # 2. Loss
        train_loss = loss_fn(train_y_preds, y_train)

        # 3. Optimizer zero grad to erase or to zero out gradiens to between 0 - 1
        # Get a fresh start every epoch instead of increasing every time... 1, 2, 3, ...
        optimizer.zero_grad()

        # 4. Backward pass- Backpropagation
        train_loss.backward()

        # 5. Steps the optimizers (perform gradient descent)
        optimizer.step()
    
        return train_y_preds, train_loss

    def interpolate_predictions(self, X_test_df: pd.DataFrame, y_test_df: pd.DataFrame, config: list) -> tuple[list]:
        """Perform interpolation to predict values within the existing range of data points (so test data), thus predict in-sample values.

        Parameters
        ----------
        X_train_df: `pd.DataFrame` 
            Input data tensor

        y_train_df: `pd.DataFrame`
            Target data tensor

        config: `py list`
            criterion: `torch.nn.Module`
                Loss criterion

            optimize: `torch.optim.Optimizer`
                Optimization algorithm
        
        Return
        ------
        Tuple[list]: 
            test_y_preds: The model's test predictions
            test_loss: The loss evaluation metric
        """

        X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
        y_test = torch.tensor(y_test_df.values, dtype=torch.float32)

        loss_fn, _ = config

        ### Testing
        self.eval() # turns off gradient tracking to make code faster as we're NOT saving gradients

        # Predictions
        with torch.inference_mode():

            # 1. Foward pass
            test_y_preds = self.forward_pass(X_test)

            # 2. Calculate test loss
            test_loss = loss_fn(test_y_preds, y_test)
        
        return test_y_preds, test_loss
    
    def extrapolate_forecasts(self, X_test_df: pd.DataFrame) -> list:
        """Perform extrapolation to predict values beyond the existing range of data points (so no test data), thus forecast out-sample values.

        Parameters
        ----------
        X_train_df: `pd.DataFrame` 
            Input data tensor
        
        Returns
        -------
        test_y_preds: `list`
                The model's forecasts
        """

        X_test = torch.tensor(X_test_df.values, dtype=torch.float32)

        ### Testing
        self.eval() # turns off gradient tracking to make code faster as we're NOT saving gradients

        # Predictions
        with torch.inference_mode():

            # 1. Foward pass
            test_y_preds = self.forward_pass(X_test)
        
        return test_y_preds

class MLP(Model):
    def __name__(self):
        return "Multi-Layer Perceptron Model"

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward_pass(self, x):
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)
        return fc2_out
        
class LinearRegressionModel(Model, nn.Module):
    def __name__(self):
        return "Linear Regression Model"
    
    def __init__(self, stabilizer: int):
        super(LinearRegressionModel, self).__init__()
        # Create random seed because we initialize randomly and want to stablize our random values
        # stablize as in keep random #s same; remove manual_seed() and model params will change
        # helps with reproducing works
        torch.manual_seed(stabilizer)

        # Randomly initialize our learnable model parameters
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward_pass(self, input_x: torch.Tensor) -> torch.Tensor:
        y = self.weights * input_x + self.bias
        return y
        
class CNN(Model, nn.Module):
    def __name__(self):
        return "CNN"
    
    def __init__(self, previous_steps: int, hidden_size: int, N_filters: int, kernel_size: int, activation_type: str, n_variables: int, pool_size: int, forecast_steps: int):
        super(CNN, self).__init__()
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

class ModelFactory:
    model_mapping = {
            'mlp': MLP,
            'lr': LinearRegressionModel,
            'cnn': CNN
        }
    
    def create_model(self, select_model: str, **kwargs) -> nn.Module:
        """Create a PyTorch model
        
        Parameters
        ----------
        select_model: `str`
            Select or input a key (ie: mlp) from model_mapping
        
        Returns
        -------
        self.model_mapping[select_model](**kwargs): `nn.Module`
            The model of interests or
            An error if pass in model that doesn't exists
        """
        if select_model in self.model_mapping:
            return self.model_mapping[select_model](**kwargs)
        else:
            raise ValueError(f"Unknown model type: {select_model}. Pass in one of the following model types: {list(self.model_mapping.keys())}")
    
    @staticmethod # This will allow us to save model (by directly creating an instance of the model) without requiring create_model().
    def save_model_and_graph(model: nn.Module, model_name: str, example_input:torch.Tensor):
        """Save a PyTorch model and the graph of the model
        
        Parameters
        ----------
        model: `nn.Module`
            The actual model
        model_name: `str`
            The file name
        
        Function
        --------
        create_file_version() -> str
            Create a new file if file with same name exists 
        
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        _, ext = os.path.splitext(model_name)
        if ext in [".pt", ".pth"]:
            print("Where to save model? Select the folder.")
            save_directory = askdirectory(title="Select Directory to Save Model and Image") + "/"

            if save_directory:
                # Save the model
                model_save_path = save_directory + model_name
                # print("model save path: ", model_save_path)
                updated_model_save_path = create_file_version(model_save_path)
                # print("updated_model_save_path: ", updated_model_save_path)
                torch.save(obj=model.state_dict(), f=updated_model_save_path)
                # print(f"Model saved successfully at: {updated_model_save_path}")

                # Visualize and save the model graph
                with torch.inference_mode():
                    output = model.forward_pass(example_input)
                dot = make_dot(output, params=dict(model.named_parameters()))
                updated_model_name, _ = os.path.splitext(model_name)
                image_path = os.path.join(save_directory, updated_model_name) + '_image'
                updated_image_path = create_file_version(image_path)
                dot.format = 'png'
                dot.render(updated_image_path)
                # print(f"Model graph image saved to {updated_image_path}.png")
                
                # Show the saved graph image
                display(Image(filename=f'{updated_image_path}.png'))
            else:
                print("Save directory not selected. Model and image not saved.")

        else:
            print(f"Cannot save model: the extension '{ext}' is incorrect. Should be '.pt' or '.pth'.")

    @staticmethod # This will allow us to save model (by directly creating an instance of the model) without requiring create_model().
    def load_model(model: nn.Module) -> nn.Module:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        files = askopenfilenames(title="Select file to load")
        model_file = files[0]
        state_dict = torch.load(f=model_file)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys when loading the model: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading the model: {unexpected_keys}")
        return model
    
    @staticmethod
    def compare_models(models: list[nn.Module], input_data_df: pd.DataFrame) -> bool:
        """Compare predictions of multiple models that are the same (as in save model, load model, compare predictions of these models to ensure same) on the given input data.
        
        Parameters
        ----------
        models: `list` 
            List of models to compare.
            
        input_data_df: `pd.DataFrame` 
            The input data for making predictions. Usually it'll be the test_data
        
        Returns
        -------
        are_equal: `bool`
            True if all models produce the same predictions, False otherwise.
        """
        if len(models) < 2:
            raise ValueError("At least two models are required for comparison.")
        
        # Convert pd.DF to torch.tensor
        input_data = torch.tensor(input_data_df.values, requires_grad=False, dtype=torch.float32)


        # Append all predictions to the same list (without reseting list)
        predictions = []
        for model in models:
            model.eval()
            with torch.inference_mode():
                predictions.append(model.forward_pass(input_data))
        
        are_equal = torch.equal(predictions[0], predictions[1]) # Compare element-wise of only first two model
        # If are_equal returns True, then compare the remaining predictions to the first model
        # Already compared the first two models (0 and 1, so start at 2)
        for i in range(2, len(predictions)):
            are_equal &= torch.equal(predictions[i], predictions[0]) # Auto break if are_equal returns False
        
        if are_equal:
            print(f"All models ({models}) produce the same predictions.")
        else:
            print(f"Models ({models}) produce different predictions.")

