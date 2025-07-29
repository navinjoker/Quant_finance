import pandas as pd
import numpy as np

def create_sequences_from_csv(csv_path, window=30):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    data = df.drop(columns=['Date']).values

    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)