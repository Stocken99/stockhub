import yfinance as yf
import pandas as pd
from curl_cffi import Session
import time
#import pandas_ta_classic

session = Session(impersonate="chrome")

#def download_data_full():
#    failed_list = []
#    period = "max"
#    
#    for i in lists.list_tickers_omx30:
#        data = yf.download(tickers = i, interval = granuality, period=period, session=session)
#    
#        if (not data.empty):
#            print("Saving to csv:")
#            data.to_csv(f"omx_30/{i}_yf_{granuality}.csv")
#            time.sleep(0.2) # sleep to to not gate ratelimited
#        else:
#            print("Data could not be downloaded")
#            failed_list.append(i)
#            
#    if(len(failed_list)>0):
#        print("Failed dowloads: ", failed_list)


def download_data_limited(granuality = "1d", ticker = "AAPL", period = "3mo"): # download and save
      
        if (f"src/data/stocks_yf/{ticker}_{granuality}_{period}" != True):
            data = yf.download(tickers=ticker, interval= granuality, period=period, session=session)
        if (not data.empty):
            data.to_csv(f"src/data/stocks_yf/{ticker}_{granuality}_{period}.csv")
        else:
            print(f"Data for {ticker} could not be downloaded")
        #time.sleep(0.2) # sleep to to not gate ratelimited

def yf_clean_data(granuality, ticker, period): #Pre-processes a single stock
    #df = pd.read_csv(f"src/data/stocks_yf/{ticker}_{granuality}_{period}.csv")
    df = pd.read_csv(f"src/data/stocks_yf/SWED-A.ST_yf_1d.csv")
    df = df.rename(columns={"Price" : "Date"})
    df = df.drop([0,1])
    #df = df[df["Date"] >= start_date]2025-11-19
    #df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format = True)
    df["Close"]= pd.to_numeric(df["Close"])
    df = df["Close"]
    #print(df)
    return df


def yf_data():
    granuality = "1d"
    ticker = "AAPL"
    period = "3mo"
    download_data_limited(granuality, ticker, period)
    df = yf_clean_data(granuality, ticker, period)
    print(df)
    return(df)
     
#download_data_limited()
#clock = time.localtime()
#str_time = f"{clock.tm_year}-{clock.tm_mon}-{clock.tm_wday}"
#str_time = pd.to_datetime(str_time)
#print(str_time)


