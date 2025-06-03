"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import os
import openai
import pathlib
import torch
import ipdb

import pandas as pd

from groq import Groq
from tqdm import tqdm
from typing import Dict, List

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer


# from log_files import LogData
# from data_processing import DataProcessing
load_dotenv()  # Load environment variables from .env file

class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    def __init__(self):
        """In the init method (also called constructor), initialize our class with variables or attributes."""
        # Create instance variables or attributes
        # Standardized model parameters
        self.temperature = 0.3
        self.top_p = 0.9
        self.model_name = None
   
    def map_platform_to_api(self, platform_name: str):
        """
        Parameter:
        ----------
        platform_name : `str`
            Platform to use for generations.
        
        Returns:
        --------
        api_key : `str`
            The api key of specified platform.

        """
        platform_to_api_mappings = {
            "GROQ_CLOUD" : os.getenv('GROQ_CLOUD_API_KEY'), # https://console.groq.com/docs/models
            "NAVI_GATOR" : os.getenv('NAVI_GATOR_API_KEY'), # https://it.ufl.edu/ai/navigator-toolkit/
            "HUGGING_FACE": os.getenv('HUGGING_FACE_API_KEY') # https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
        }

        api_key = platform_to_api_mappings.get(platform_name)
        
        if api_key is None:
            raise ValueError("API_KEY environment variable not set")
        
        return api_key
    
    @classmethod        
    def create_instance(self, model_name):

        # Groq models
        if model_name == 'llama-3.3-70b-versatile':
            return LlamaVersatileTextGenerationModel()
        elif model_name == 'llama-3.1-8b-instant':
            return LlamaInstantTextGenerationModel()
        elif model_name == 'llama3-70b-8192':
            return Llama70B8192TextGenerationModel()
        elif model_name == 'llama3-8b-8192':
            return Llama8B8192TextGenerationModel()
        else:
            raise ValueError(f"Unknown class name: {model_name}")

    def assistant(self, content: str) -> Dict:
        """Create an assistant message.
        
        Parameters:
        -----------
        content: `str`
            The content of the assistant message.
        
        Returns:
        --------
        Dict
            A dictionary representing the assistant message.
        """

        return {"role": "assistant", "content": content}
    
    def user(self, content: str) -> Dict:
        """Create a user message.
        
        Parameters:
        -----------
        content : `str`
            The content of the user message.
        
        Returns:
        --------
        Dict
            A dictionary representing the user message.
        """

        return {"role": "user", "content": content}
    
    def chat_completion(self, messages: List[Dict]) -> str:
        """Generate a chat completion response.
        
        Parameters:
        -----------
        messages: `List[Dict]`
            A list of dictionaries representing the chat history.
        
        model: `str`
            The name of the model to use.
        
        temperature: `float`
            Sampling temperature.
        
        top_p: `float`
            Nucleus sampling parameter.
        
        Returns:
        --------
        `str`
            The generated chat completion response.
        """

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
    
    def generate_predictions(self, prompt_template: str, label: str, domain: str, batch_id: int) -> pd.DataFrame:
        """Generate a completion response and return as a DataFrame.

        Parameters:
        -----------
        prompt_template: `str`
            The prompt template to generate a prediction prompt or a non-prediction prompt.
        
        label: `str`
            The prediction label for the prediction. Either 0 (non-prediction) or 1 (prediction).
        
        domain: `str`
            The domain of the prediction. As of now, the domains are finance, weather, health, and policy.

        template_number: `int`
            The template number to use for the prediction. For non-prediction prompts, the template number is 0 and for prediction prompts, the template number is 1 to 5.
        
        Returns:
        --------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
        """
        # Generate the raw prediction text
        # print(f"\n  prompt_template: \n{prompt_template}\n\n")
        raw_text = self.chat_completion([self.user(prompt_template)])
        # print(f"    {self.model_name} + {domain} generates: {raw_text}")
        print(f"generates:\n{raw_text}")
        
        
        # Parse the raw text into structured data (assuming a consistent format)
        predictions = []
        for line in raw_text.split("\n"):
            if line.strip():  # Skip empty lines
                predictions.append(line.strip())
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions, columns=['Base Sentence'])
        df['Sentence Label'] = label
        df['Domain'] = domain
        df['Model Name'] = self.model_name
        df['API Name'] = self.api_name
        df['Batch ID'] = batch_id
        # print()
        # print(df)
        # ipdb.set_trace()
        # print()
        return df

    def log_batch_df(self, reformat_batch_predictions_df, sentence_label):
        print("Start logging batch")

        base_path = pathlib.Path(__file__).parent.resolve()
        
        prediction_files = None
        if sentence_label == 0:
            prediction_files = "observation"
        elif sentence_label == 1:
            prediction_files = "prediction"
        else:
            print("Invalid sentence label. It should be 0 (non-prediction) or 1 (prediction).")
            quit()

        log_file_path = f"data/{prediction_files}_logs"
        log_directory = os.path.join(base_path, log_file_path)
        print(f"log_directory: {log_directory}")
        
        n = 1
        save_batch_directory = os.path.join(log_directory, f"batch_{n}-{prediction_files}")
    
        while os.path.exists(save_batch_directory):
            n += 1
            save_batch_directory = os.path.join(log_directory, f"batch_{n}-{prediction_files}")

        os.makedirs(save_batch_directory)
        save_batch_name = f"batch_{n}-info.log"
        save_from_df_name = f"batch_{n}-from_df.csv"
        save_from_csv_name = f"batch_{n}-from_csv.log"
    
        logger = LogData(base_path, log_file_path, save_batch_directory, save_batch_name)
        logger.dataframe_to_csv(reformat_batch_predictions_df, save_from_df_name)
        logger.csv_to_log(save_from_df_name, save_from_csv_name)

    def batch_generate_data(self, N_batches, text_generation_models, domains, prompt_outputs, sentence_label):
        """Generate a completion response and return as a DataFrame.

        Parameters:
        -----------
        N_batches: `int`
            The number of batches
        
        text_generation_models: `list`
            The models to use to generate data
        
        domains: `str`
            The domain of the prediction. As of now, the domains are finance, weather, health, and policy.

        prompt_outputs: `dict`
            Langchain dictionary to map the domain to which prompt to use for generation.

        sentence_label: `int`
            The prediction label for the prediction. Either 0 (non-prediction) or 1 (prediction).

        Returns:
        --------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
    """

        all_batches_df = []   
        for batch_idx in tqdm(range(N_batches)):
            print(f"===================================== Batch {batch_idx} ===============================================")

            batch_dfs = [] # Reset when starting a new batch

            for domain in domains:
                # print(f"    Domain: {domain}")            
                for text_generation_model in text_generation_models:
 
                    print(f"{domain} --- {text_generation_model.__name__()} --- {text_generation_model.api_name}")

                    prompt_output = prompt_outputs[domain]
                    model_df = text_generation_model.generate_predictions(prompt_output, label=sentence_label, domain=domain, batch_id=batch_idx)

                    batch_dfs.append(model_df)
                    batch_predictions_df = DataProcessing.concat_dfs(batch_dfs)
                    reformat_batch_predictions_df = DataProcessing.reformat_df_with_template_number(batch_predictions_df, col_name="Base Sentence")
                print()

                # print(f"NEW DOMAIN: {domain}")
            # ipdb.set_trace()

            self.log_batch_df(reformat_batch_predictions_df, sentence_label)
            # print(reformat_batch_predictions_df)

            # Extend the main DataFrame list with the batch DataFrames
            all_batches_df.append(reformat_batch_predictions_df)
        # print(all_batches_df)
        updated_all_batches_df = DataProcessing.concat_dfs(all_batches_df)
        return updated_all_batches_df    

    def __name__(self):
        pass

class LlamaVersatileTextGenerationModel(TextGenerationModelFactory):    
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()
    
    def __name__(self):
        return "llama-3.3-70b-versatile"

class LlamaInstantTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama-3.1-8b-instant"

    def __name__(self):
        return "llama-3.1-8b-instant"

class Llama70B8192TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama3-70b-8192"

    def __name__(self):
        return "llama3-70b-8192"

class Llama8B8192TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama3-8b-8192"
    
    def __name__(self):
        return "llama3-8b-8192"