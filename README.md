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
We will use two different kinds of models, an LSTM and Cronos-2 model. An LSTM is a type of recurrent neural network or RNN designed to mitigate the vanishing gradient problem. The main advantage of the LSTM is the low requirements of computing power to train it. The Cronos-2 model is a state-of-the-art 120M-parameter, encoder-only time series foundation model specialized for zero-shot forecasting. It is a pre-trained model trained on a combination of real-world and synthetic data. The model used in the final product will be the one that achieves the highest result during the development of the application.

Cronos-2 model: [amazon/chronos-2 · Hugging Face](https://huggingface.co/amazon/chronos-2)
## 3. Dataset

We are planning to use the API provided by Alpha Vantage to collect our datasets. The main data we will utilize from this site is the time series of different stocks with a daily granularity with the corresponding technical indicators provided such as SMA, RSI, with others. The other main type of data we will fetch from this site is the news of different stocks.

https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo

The above is an example of a query to fetch the daily time series data for the IBM stock with the demo API-key.


[Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
