import pandas as pd
import json
import yaml
import os


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def extract_timeseries(path):
    data = load_json(path)
    ts = data["Time Series (Daily)"]

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df = df.sort_index()
    df.index.name = "date"
    return df.reset_index()


def extract_indicator(path, key, value_name):
    data = load_json(path)
    d = data[key]

    df = pd.DataFrame.from_dict(d, orient="index")
    df.index = pd.to_datetime(df.index)
    df[value_name] = df[value_name].astype(float)
    df = df.sort_index()
    df.index.name = "date"
    return df.reset_index()


def extract_bbands(path):
    data = load_json(path)
    d = data["Technical Analysis: BBANDS"]

    df = pd.DataFrame.from_dict(d, orient="index")
    df.index = pd.to_datetime(df.index)

    df = df.astype(float)
    df = df.sort_index()
    df.index.name = "date"
    return df.reset_index()


def extract_news(path, ticker="AAPL"):
    data = load_json(path)
    articles = data["feed"]

    df = pd.DataFrame(articles)
    df["time_published"] = pd.to_datetime(df["time_published"])
    df["date"] = df["time_published"].dt.date

    def get_score(row, key):
        for item in row["ticker_sentiment"]:
            if item["ticker"] == ticker:
                return float(item[key])
        return None

    df["sentiment"] = df.apply(lambda r: get_score(r, "ticker_sentiment_score"), axis=1)
    df["relevance"] = df.apply(lambda r: get_score(r, "relevance_score"), axis=1)

    df_daily = df.groupby("date").agg({
        "sentiment": "mean",
        "relevance": "mean"
    }).reset_index()

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    return df_daily


def preprocess(config_path="config/preprocess.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ticker = cfg["ticker"]
    output_csv = cfg["output_dataset"]

    df_prices = extract_timeseries(paths["time_series_daily"])
    df_sma = extract_indicator(paths["sma"], "Technical Analysis: SMA", "SMA")
    df_rsi = extract_indicator(paths["rsi"], "Technical Analysis: RSI", "RSI")
    df_obv = extract_indicator(paths["obv"], "Technical Analysis: OBV", "OBV")
    df_bbands = extract_bbands(paths["bbands"])
    df_news = extract_news(paths["news"], ticker=ticker)

    # Merge
    df = df_prices.merge(df_sma, on="date", how="outer")
    df = df.merge(df_rsi, on="date", how="outer")
    df = df.merge(df_obv, on="date", how="outer")
    df = df.merge(df_bbands, on="date", how="outer")
    df = df.merge(df_news, on="date", how="outer")

    df = df.sort_values("date")
    df = df.fillna(method="ffill")

    df.to_csv(output_csv, index=False)
    print(f"Preprocessing done! Saved: {output_csv}")
    return output_csv


if __name__ == "__main__":
    preprocess()



