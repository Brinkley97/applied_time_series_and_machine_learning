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
