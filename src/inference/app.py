import streamlit as st
import requests
import pandas as pd

st.title("Stockhub Forecasting Demo")

tickers_input = st.text_input("Enter stock tickers:", "AAPL, MSFT")

if st.button("Predict"):
    tickers = [t.strip() for t in tickers_input.split(",")]

    response = requests.post(
        "http://localhost:8000/predict",
        json={"tickers": tickers}
    )

    if response.status_code != 200:
        st.error("Error: " + response.text)
    else:
        data = response.json()
        st.subheader("Predictions")
        st.write(pd.DataFrame(data["predictions"]))

