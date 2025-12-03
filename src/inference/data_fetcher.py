import os
import requests
import json
import pandas as pd

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

#Fetch historical time series data for tickers
def fetch_single_ticker(ticker: str) -> pd.DataFrame:
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}&apikey={API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid response for ticker {ticker}: {data}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert to Chronos schema
    df = df.reset_index().rename(columns={
        "index": "timestamp",
        "4. close": "target"
    })

    df["item_id"] = ticker

    return df[["timestamp", "target", "item_id"]]

# Fetch and concatenate multiple tickers into one DataFrame
def fetch_data(tickers):
    frames = []
    for t in tickers:
        print(f"Fetching: {t}")
        frames.append(fetch_single_ticker(t))

    return pd.concat(frames, ignore_index=True)

#Local testing
RAW_DIR = "data/raw"

# Download data from alpha vantage locally to data/raw for local testing
def load_local_daily(ticker: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{ticker}_daily.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No local dataset found for {ticker}: expected {path}"
        )

    with open(path, "r") as f:
        data = json.load(f)

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Invalid local JSON format for {ticker}")

    ts = data["Time Series (Daily)"]

    df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df.reset_index().rename(columns={
        "index": "timestamp",
        "4. close": "target"
    })

    df["item_id"] = ticker

    return df[["timestamp", "target", "item_id"]]


