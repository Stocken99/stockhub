from data.alpha_vantage_client import AlphaVantageClient
import json
import pandas as pd 
def main():
    client = AlphaVantageClient(config_path="config/alpha_vantage.yaml")
    ticker = "AAPL"
    print("Testing TIME_SERIES_DAILY...")
    data = client.fetch("TIME_SERIES_DAILY", ticker=ticker, outputsize="compact")
    print(json.dumps(data, indent=4)[:500]) #Printa första 500
#    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
#    df.to_csv(f"{ticker}_timeseries.csv")

    print("\nTesting NEWS_SENTIMENT...")
    data = client.fetch(
        "NEWS_SENTIMENT",
        ticker="AAPL",
        time_from="20250101T0000",
        time_to="20250102T0000"
    )
    print(json.dumps(data, indent=4)[:500])

    print("\nTesting SMA...")
    data = client.fetch("SMA", ticker="AAPL")
    print(json.dumps(data, indent=4)[:500])

if __name__ == "__main__":
    main()
