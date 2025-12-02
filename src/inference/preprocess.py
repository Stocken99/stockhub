import pandas as pd

def preprocess(df_raw):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["date"])
    df["target"] = df["4. close"].astype(float)
    df["item_id"] = "AAPL"
    df = df[["timestamp", "item_id", "target"]].sort_values("timestamp")

    return df

