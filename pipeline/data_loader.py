import yfinance as yf
import pandas as pd
from datetime import datetime

def load_or_update_price_csv(tickers, path):
    try:
        existing = pd.read_csv(path, parse_dates=['Date'])
        start_date = existing['Date'].max() + pd.Timedelta(days=1)
    except:
        existing = pd.DataFrame()
        start_date = "2010-01-01"

    end_date = datetime.today().strftime('%Y-%m-%d')
    df_new = yf.download(tickers, start=start_date, end=end_date)['Close']
    if df_new.empty:
        return existing

    df_new = df_new.pct_change().dropna().reset_index()
    updated = pd.concat([existing, df_new], ignore_index=True)
    updated.drop_duplicates(subset='Date').to_csv(path, index=False)
    return updated