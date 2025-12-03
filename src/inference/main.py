from fastapi import FastAPI
import pandas as pd
from src.inference.model_loader import load_model
from src.inference.data_fetcher import load_local_daily, fetch_single_ticker
from src.inference.preprocess import preprocess_for_chronos
from src.inference.schemas import PredictionRequest

app = FastAPI()

model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

#Helper function for convertion of predictions
def preds_to_records(preds):
    # Case 1: preds is a single DataFrame
    if isinstance(preds, pd.DataFrame):
        return preds.reset_index().to_dict(orient="records")

    # Case 2: preds is a dict {item_id: df}
    if isinstance(preds, dict):
        all_rows = []
        for df in preds.values():
            all_rows.extend(df.reset_index().to_dict(orient="records"))
        return all_rows

    # Case 3: Autogluon TimeSeriesDataFrame
    if hasattr(preds, "to_pandas"):
        return preds.to_pandas().reset_index().to_dict(orient="records")

    # Last fallback
    return pd.DataFrame(preds).reset_index().to_dict(orient="records")

@app.post("/predict")
def predict(request: PredictionRequest):
    results = []

    for ticker in request.tickers:
        # Fetch data locally
        df_raw = load_local_daily(ticker)

        # Fetch data through API
        #df_raw = fetch_single_ticker(ticker)

        # Prepare for Chronos format
        df_chronos = preprocess_for_chronos(df_raw, ticker)

        # Run prediction
        preds = model.predict(df_chronos)

        results.append({
            "ticker": ticker,
            "history": df_chronos.to_dict(orient="records"),
            #"forecast": preds.reset_index().to_dict(orient="records")
            "forecast": preds_to_records(preds)
        })

    return {"results": results}

