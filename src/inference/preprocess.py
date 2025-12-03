import pandas as pd

#Get required chronos columns
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["item_id", "timestamp"])
    df = df.reset_index(drop=True)
    return df

def preprocess_for_chronos(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Ensure full daily frequency (important for Chronos)
    df = df.asfreq("1D")

    # Fill missing days (weekends/holidays)
    df["target"] = df["target"].ffill()

    # Add item_id
    df["item_id"] = ticker

    df = df.reset_index()

    return df[["timestamp", "target", "item_id"]]

