import os
import time
import json
import requests

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "TSLA", "META", "JPM", "BAC", "XOM"
]

API_KEY = "R4UQU7GI73HAJDLL" # May need to be modified since API can get lost
BASE_URL = "https://www.alphavantage.co/query"

SAVE_DIR = "data/raw/"

os.makedirs(SAVE_DIR, exist_ok=True)


def fetch_daily_series(ticker: str):
    """Fetch daily price data from AlphaVantage for one ticker."""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": API_KEY,
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    # Handle rate limits
    if "Note" in data:
        print(f"Rate limit hit. Waiting 60 seconds...")
        time.sleep(60)
        return fetch_daily_series(ticker)

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected response for {ticker}: {data}")

    return data


def save_json(ticker: str, data: dict):
    path = os.path.join(SAVE_DIR, f"{ticker}_daily.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {ticker} → {path}")


def main():
    if not API_KEY:
        raise ValueError("No API key found. Set ALPHA_VANTAGE_API_KEY")

    print("----------Downloading Demo Dataset------------")
    print(f"Tickers: {', '.join(TICKERS)}\n")

    for ticker in TICKERS:
        save_path = os.path.join(SAVE_DIR, f"{ticker}_daily.json")

        if os.path.exists(save_path):
            print(f"Skipping {ticker} (already exists)")
            continue

        print(f"Fetching {ticker}...")
        data = fetch_daily_series(ticker)
        save_json(ticker, data)

        # AlphaVantage limit: 5 calls / minute
        print("Waiting 12 seconds to respect rate limit...\n")
        time.sleep(12)

    print("\nDone! Demo dataset ready.")


if __name__ == "__main__":
    main()

