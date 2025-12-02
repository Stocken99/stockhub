from fastapi import FastAPI
from src.inference.model_loader import load_model
from src.inference.data_fetcher import fetch_data
from src.inference.preprocess import preprocess
from src.inference.schemas import PredictionRequest, PredictionResponse

app = FastAPI()

# Load model once when the API starts
model = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 1. Fetch raw stock data
    df_raw = fetch_data(request.tickers)

    # 2. Preprocess for inference
    df_preprocessed = preprocess(df_raw)

    # 3. Make prediction
    preds = model.predict(df_preprocessed)

    # Convert AutoGluon TimeSeriesDataFrame to pandas
    preds_df = preds.reset_index()

    # Convert pandas to list of dict entries
    preds_json = preds_df.to_dict(orient="records")

    return PredictionResponse(
        tickers=request.tickers,
        predictions=preds_json,
    )
