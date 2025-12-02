import models.LSTM_model as lstm_py
import data.yf_fetcher as yff

def start_lstm(run = True): # False unless called upon
    if run == True:
        data = yff.yf_data() # just using temporary yfinance data
        lstm_py.run_lstm(data)

start_lstm()