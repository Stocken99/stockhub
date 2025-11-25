# Price Movement Prediction of Financial Assets

## Authors Group Members
| Name             | GitHub Username  | 
| ---------------- | ---------------- | 
| Mohamad Alkhaled | Mohamadalkhaled  | 
| Felix Stockinger | Stocken99        | 
| Lukas Ydkvist    | Lukasydkvist | 

## 1. Motivation / Introduction

Predicting and understanding the stock market remains one of the most complex and widely researched topics in modern data science. There are lot of factors that influence the market like numerical indicators such as price and volume, but also textual like news, trends, and sentiments. Traditional methods often fail to capture these non-linear relationships which makes this topic a great target for advanced AI systems.

Gathering and analyzing all data in terms of context, correlations, and actionable decisions can be time consuming.

Our project aims to address this real-world problem by offering an AI-assisted platform that allows users to gain a deeper understanding of selected stock tickers. By combining pre-trained models for sentiment analysis and time-series forecasting, the application will combine numerical and textual data to generate informative summaries and actionable proposals.

This application will democratize access to advanced financial insights that many non-professional investor might not have available. Users will recieve transparent, data-driven, trustworthy recommendations, in one of the most complex and unpredictable domains.



## 2. Pre-trained Model / Method
We are using the **AutoGluon Chronos** time series model for forecasting stock prices. 

- **Inputs / Features**: Historical stock prices (Open, High, Low, Close), SMA, RSI, OBV, Bollinger Bands, and news sentiment scores.  
- **Target Variable**: Closing price (`4. close`).  
- **Metrics**: RMSE (Root Mean Squared Error), squared error per time step.  

The model is trained using a preprocessed dataset combining price history, technical indicators, and news sentiment. Hyperparameters like number of layers, hidden size, dropout, learning rate, and batch size can be tuned to improve performance.

## 3. Experiment and Dataset

We fetch data from **Alpha Vantage API**, including:

- Daily stock time series data for selected tickers  
- Technical indicators: SMA, RSI, OBV, Bollinger Bands  
- News and sentiment scores  

The data is processed by:

1. Merging price history, technical indicators, and news sentiment into one dataset  
2. Sorting by date  
3. Forward-filling missing values  

The final dataset is saved as `data/processed/processed_dataset.csv` and used to train the AutoGluon Chronos model.

### Initial Experiment Results
After running the AutoGluon Chronos model on the processed dataset:

- The model predicts the closing price (`4. close`) with reasonable accuracy.  
- The `mean` column shows the predicted price, and columns `0.1`–`0.9` represent prediction intervals. The actual prices mostly fall within these intervals.  
- Predictions capture short-term trends in the stock price.  
- All predictions are saved to `data/processed/predictions.csv` for further analysis.  

**MLflow Screenshots:**  
Here screenshots from MLFlow of model evaluation metrics
![Predictions Plot](data\images/mse.png)
![MLflow Metrics](data\images/metric1.png)
![MLflow Metrics](data\images/metric2.png)

## 4. Build / Running Instructions
### Using Docker
# Build and start all services
docker-compose up --build

## 5. Initial Ideas for Model Deployment and Inference Serving

A web interface could let users select stock tickers and view predicted prices along with confidence intervals. This provides an easy way to explore forecasts and trends. Frameworks like **Streamlit** could be used to build this interactive frontend.
