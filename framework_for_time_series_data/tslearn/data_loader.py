import os.path

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

from time_series import UnivariateTimeSeries

def build_airline_passenger_uts() -> UnivariateTimeSeries:
    # Get air passenger data and build our UTS
    data_df = pd.read_csv("../../datasets/AirPassengers.csv")

    return UnivariateTimeSeries(
        time_col="month",
        time_values=data_df["Month"].values,
        values_cols="passengers_count",
        values=data_df["#Passengers"].values
    )

def build_stock_uts(stock_symbol: str, stock_name: str, independent_variable: str, start_date: str, end_date: str, frequency: str) -> UnivariateTimeSeries:
    """Get the statistics of the univariate time series data.

    Parameters
    ----------
    independent_variable: `str`
        The single variable of interests. Use "Close" price listing to evaluate end of day or overall market sentiment. How much can I make at the end of X day?

    frequency: `str`
        The yfinance library supports various intervals for downloading financial data. Some are '1d': Daily, '1wk': Weekly, '1mo': Monthly , '5m': 5 minutes, '15m': 15 minutes, '30m': 30 minutes, '1h': 1 hour, '90d': 3 months (approximated)

    end_date: `str`
        The last date we want. This is exclusive (ie: I want 2023-06-21 in my data, I must set this param to 2023-06-22).

    Returns
    -------
    `UnivariateTimeSeries`
    """

    stock_df = yf.download(stock_symbol, start=start_date, end=end_date, interval=frequency)

    return UnivariateTimeSeries(
        time_col="Date",
        time_values=stock_df.index,
        values_cols=independent_variable,
        values=stock_df[independent_variable].values
    )

def build_air_temperature_uts() -> UnivariateTimeSeries:
    data_df = pd.read_csv("../../datasets/daily-min-temperatures.csv")

    return UnivariateTimeSeries(
        time_col="Date",
        time_values=data_df["Date"],
        values_cols="Temp",
        values=data_df["Temp"].values
    )

def build_sunspots_uts() -> UnivariateTimeSeries:
    # data_df = pd.read_csv("../datasets/daily-min-temperatures.csv")
    print(sm.datasets.sunspots.NOTE)
    data_df = sm.datasets.sunspots.load_pandas().data
    data_df.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
    data_df.index.freq = data_df.index.inferred_freq


    return UnivariateTimeSeries(
        time_col="YEAR",
        time_values=data_df["YEAR"],
        values_cols="SUNACTIVITY",
        values=data_df["SUNACTIVITY"].values
    )

def build_website_traffic_uts() -> UnivariateTimeSeries:
    # Get website traffic data and build our UTS
    data_df = pd.read_csv("../datasets/website_data.csv")
    data_df["Timestamp"] = data_df.index
    data_df.set_index("Timestamp", inplace=True)

    return UnivariateTimeSeries(
        time_col=data_df.index.name,
        time_values=data_df.index,
        values_cols="traffic",
        values=data_df["traffic"].values
    )

def build_bitcoin_uts() -> UnivariateTimeSeries:
    # Get website traffic data and build our UTS
    data_df = pd.read_csv("../../datasets/nlp_ts/bitcoin_2017_to_2023.csv")
    data_df['Date with tmestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['timestamp'] = data_df['Date with tmestamp'].dt.date
    data_df = data_df.rename(columns={"timestamp": "Timestamp"})
    data_df.set_index("Timestamp", inplace=True)



    return UnivariateTimeSeries(
        time_col=data_df.index.name,
        time_values=data_df.index.values,
        values_cols="close",
        values=data_df["close"].values
    )

def build_any_univariate_time_series(path_to_file: str) -> UnivariateTimeSeries:
    file_extension = os.path.splitext(path_to_file)[1]

    # add support for if csv file
    # if file_extension == ".csv":
    #     data_df = pd.read_csv(path_to_file)

    # if data_csv = pd.read_csv(path_to_file)
    # update: if index is originally 0, change to 1. Maybe return both.
    if file_extension == ".npy":
        with open(path_to_file, 'rb') as f:
            series = np.load(f)
            data_df = pd.DataFrame(series, columns=['Observations'])
            data_df['Timestamp'] = data_df.index
            data_df.set_index('Timestamp', inplace=True)

        return UnivariateTimeSeries(
            time_col=data_df.index.name,
            time_values=data_df.index,
            values_cols="Observations",
            values=data_df["Observations"].values
        )

    if file_extension == ".csv":
        data_csv = pd.read_csv(path_to_file)
        columns = data_csv.columns
        series = data_csv.values

        number_of_observations, number_of_columns = np.shape(data_csv)

        if number_of_columns == 1:
            data_df = pd.DataFrame(series, columns=['Observations'])
            data_df['Timestamp'] = data_df.index
            data_df.set_index('Timestamp', inplace=True)

        elif number_of_columns == 2:

            data_df = pd.DataFrame(series, columns=['Timestamp', 'Observations'])
            # data_df['Timestamp'] = data_df.index
            # data_df['Timestamp'] = data_csv.columns[0]
            data_df.set_index('Timestamp', inplace=True)

        return UnivariateTimeSeries(
            time_col=data_df.index.name,
            time_values=data_df.index,
            values_cols="Observations",
            values=data_df["Observations"].values
        )


    else:
        print("File extension not supported yet. Contact me at dbrinkle@usc.edu so I can add support for this file extension.")
