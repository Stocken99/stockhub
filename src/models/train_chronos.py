# src/models/train_chronos.py

import pandas as pd
import os
from autogluon.timeseries import TimeSeriesPredictor
import yaml
import numpy as np

# ----------------------------------
# Load config
# ----------------------------------
def load_config(path="config/train_chronos.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ----------------------------------
# Data prep
# ----------------------------------
def prepare_data(csv_path, prediction_length):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    df = df[["date", "4. close"]].rename(
        columns={"date": "timestamp", "4. close": "target"}
    )
    df["item_id"] = "AAPL"
    df = df.sort_values("timestamp")

    train_df = df.iloc[:-prediction_length]
    test_df = df.iloc[-prediction_length:]

    return train_df, test_df

# ----------------------------------
# Training function (NO MLFLOW)
# ----------------------------------
def train_chronos_model(cfg, processed_csv_path=None):

    freq = cfg["params"].get("freq", "1D")

    if processed_csv_path is not None:
        cfg["data"]["processed_csv"] = processed_csv_path

    # Load dataset
    train_df, test_df = prepare_data(
        cfg["data"]["processed_csv"],
        cfg["params"]["prediction_length"]
    )

    # Ensure output folder exists
    model_dir = "models/chronos_model/"
    os.makedirs(model_dir, exist_ok=True)

    # Create predictor
    predictor = TimeSeriesPredictor(
        prediction_length=cfg["params"]["prediction_length"],
        target="target",
        eval_metric="RMSE",
        path=model_dir,
        freq=freq
    )

    predictor.fit(train_df, hyperparameters={"Chronos": {}})
    predictor.save()

    # Predict for evaluation
    predictions = predictor.predict(train_df)

    merged = predictions.merge(
        test_df[["timestamp", "target", "item_id"]],
        on=["timestamp", "item_id"],
        how="left"
    )

    # Compute metrics
    merged["sq_error"] = (merged["mean"] - merged["target"]) ** 2
    mse = merged["sq_error"].mean()
    mae = (merged["mean"] - merged["target"]).abs().mean()
    rmse = np.sqrt(mse)

    out_path = cfg["outputs"]["prediction_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("Training finished. MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    

    metrics = {"mse": mse, "rmse": rmse, "mae": mae}

    return metrics, out_path, model_dir



