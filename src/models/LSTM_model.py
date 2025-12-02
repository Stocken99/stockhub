import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_sequences(data, seq_len = 32,future_pred = 1):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + future_pred]) # just one step ahead i+seq_len:i + seq_len
    return np.array(X), np.array(y)
     

def run_lstm(data): 
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1,1))
    future_pred=30
    batch_X, y = create_sequences(scaled_data, future_pred)

    print(batch_X.shape)
    lstm = Sequential([
        LSTM(32,return_sequences = False, input_shape=(32,1)), 
        Dense(future_pred)
        ])
    lstm.compile(optimizer = "adam", loss = "mse")
    lstm.summary()
    lstm.fit(batch_X,y,epochs = 1, batch_size = 32, verbose = 1)

    preds = lstm.predict(batch_X)
    preds = scaler.inverse_transform(preds)
    print(preds) #.flatten()
    import matplotlib.pyplot as plt

    # Assuming `data` is your original pandas Series of Close prices
    # and `preds` is the output from your LSTM, shape (num_sequences, future_pred)
    future_pred = preds.shape[1]

    # Get last prediction vector
    last_pred = preds[-1]
    print(last_pred)
    # Create x-axis for plotting
    historical_len = len(data)
    forecast_x = np.arange(historical_len, historical_len + future_pred)
    #plt.figure(figsize=(12,6))
    plt.plot(np.arange(historical_len), data[:historical_len].values)#, label="Historical Stock Prices"
    plt.plot(forecast_x, last_pred, label="LSTM Forecast", color='red', marker='o')
    plt.xlabel("Time step")
    plt.ylabel("Stock Price")
    plt.title("Stock Price and LSTM Forecast")
    plt.legend()
    plt.grid(True)
    plt.show()
