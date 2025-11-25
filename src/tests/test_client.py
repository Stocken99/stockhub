from ..data.alpha_vantage_client import AlphaVantageClient
import json

def main():
    client = AlphaVantageClient(config_path="config/alpha_vantage.yaml")
    
    print("Testing TIME_SERIES_DAILY...")
    data = client.fetch("TIME_SERIES_DAILY", ticker="AAPL")
    print(json.dumps(data, indent=4)[:500]) #Printa första 500

    print("\nTesting NEWS_SENTIMENT...")
    data = client.fetch(
        "NEWS_SENTIMENT",
        ticker="AAPL",
        time_from="19990101T0000",
        time_to="20250102T0000"
    )
    print(json.dumps(data, indent=4)[:500])

    print("\nTesting SMA...")
    data = client.fetch("SMA", ticker="AAPL")
    print(json.dumps(data, indent=4)[:500])

if __name__ == "__main__":
    main()
