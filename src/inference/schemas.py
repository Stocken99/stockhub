from pydantic import BaseModel
from typing import List, Any

class PredictionRequest(BaseModel):
    tickers: List[str]

class PredictionResponse(BaseModel):
    tickers: List[str]
    predictions: Any
