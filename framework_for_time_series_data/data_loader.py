import numpy as np
import pandas as pd
import yfinance as yf

from time_series import UnivariateTimeSeries

def build_airline_passenger_uts() -> UnivariateTimeSeries:
    # Get air passenger data and build our UTS
    data_df = pd.read_csv("../datasets/AirPassengers.csv")

    return UnivariateTimeSeries(
        time_col="month",
        time_values=data_df["Month"].values,
        values_cols="passengers_count",
        values=data_df["#Passengers"].values
    )

def build_stock_uts(stock_symbol: str, stock_name: str, start_date: str, end_date: str) -> UnivariateTimeSeries:
    stock_df = yf.download(stock_symbol, start=start_date, end=end_date)

    return UnivariateTimeSeries(
        time_col="Date",
        time_values=stock_df.index,
        values_cols="Open",
        values=stock_df["Open"].values
    )

def build_any_time_series_uts(file, time_col_name: str, value_col_name: str) -> UnivariateTimeSeries:
    with open(file, 'rb') as f:
        series = np.load(f)
        data_df = pd.DataFrame(series, columns=[value_col_name])
        # print(data_df)

        return UnivariateTimeSeries(
            time_col=time_col_name,
            time_values=data_df.index,
            values_cols=value_col_name,
            values=data_df[value_col_name].values
        )
