import requests
import pandas as pd
import os

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_single_ticker(ticker):
    url = ("https://www.alphavantage.co/query"
           "?function=TIME_SERIES_DAILY"
           f"&symbol={ticker}&apikey={API_KEY}")

    r = requests.get(url)
    data = r.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Could not fetch data for {ticker}: {data}")

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index"
    ).astype(float)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "date"

    df["ticker"] = ticker
    return df.reset_index()

def fetch_data(tickers):
    dfs = [fetch_single_ticker(t) for t in tickers]
    return pd.concat(dfs, ignore_index=True)

