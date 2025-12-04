import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Stockhub Forecasting Demo", layout="wide")

# Sidebar for input
st.sidebar.header("Stockhub Forecasting")
tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT")
prediction_length = st.sidebar.slider("Prediction Days", min_value=1, max_value=30, value=30)

st.title("Stockhub Forecasting Dashboard")

if st.sidebar.button("Predict"):
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

    for result in results:
        ticker = result["ticker"]
        st.markdown(f"### {ticker}")

        df_hist = pd.DataFrame(result["history"])
        df_pred = pd.DataFrame(result["forecast"])

        # Plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["target"],
            name="Historical",
            mode="lines",
            line=dict(color="blue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_pred["timestamp"],
            y=df_pred["mean"],
            name="Forecast",
            mode="lines",
            line=dict(color="orange", dash="dash", width=2)
        ))

        fig.update_layout(
            title=f"{ticker} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",  # dark theme for style
            legend=dict(x=0, y=1)
        )

        # Use container & expander
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Forecast Table"):
                st.dataframe(df_pred)



