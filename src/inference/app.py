import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stockhub Forecasting Demo", layout="wide")
st.title("Stockhub Forecasting Demo")

tickers_input = st.text_input("Enter stock tickers:", "AAPL, MSFT")

if st.button("Predict"):
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    with st.spinner("Fetching predictions..."):
        response = requests.post(
            "http://localhost:8000/predict",
            json={"tickers": tickers}
        )

    if response.status_code != 200:
        st.error("Error: " + response.text)
        st.stop()

    results = response.json()["results"]

    st.subheader("Model Predictions")

    for result in results:
        ticker = result["ticker"]

        st.markdown(f"## {ticker}")

        df_hist = pd.DataFrame(result["history"])
        df_pred = pd.DataFrame(result["forecast"])

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["target"],
            name="Historical",
            mode="lines",
            line=dict(color="blue")
        ))

        fig.add_trace(go.Scatter(
            x=df_pred["timestamp"],
            y=df_pred["mean"],
            name="Forecast",
            mode="lines",
            line=dict(color="orange", dash="dash")
        ))

        fig.update_layout(
            title=f"{ticker} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show forecast table
        st.dataframe(df_pred)



