# 🧬 Low-Level Design (LLD)

## 📌 Project: Hybrid Deep Learning Model for Financial Time Series Forecasting

---

## 🔧 Module: `pipeline/data_loader.py`

### Function: `load_or_update_price_csv(tickers, path)`
- **Input**: List of tickers, path to price.csv
- **Logic**:
  - Read existing CSV
  - Determine start date from last available row
  - Download new price data using yFinance
  - Calculate daily returns
  - Append, deduplicate, and save updated file
- **Output**: Updated DataFrame

---

## 🔧 Module: `pipeline/preprocessing.py`

### Function: `create_sequences_from_csv(csv_path, window)`
- **Input**: Path to return matrix CSV, window size
- **Logic**:
  - Drop 'Date' column
  - For each time index `t`, form sequence `X[t:t+window]` and label `y[t+window]`
- **Output**: `(X, y)` pair as NumPy arrays

---

## 🔧 Module: `pipeline/model.py`

### Function: `build_model(hp, input_shape)`
- **Input**: Keras tuner `hp`, shape `(window, n_assets)`
- **Architecture**:
  - LSTM (units tunable)
  - Transformer block with:
    - Multi-Head Attention
    - Feed-forward layers
  - GlobalAvgPooling → Dense → Dropout → Output
- **Output**: Compiled Keras model

---

## 🔧 Module: `pipeline/train.py`

### Function: `train_model(...)`
- **Logic**:
  - Uses Keras-Tuner `RandomSearch`
  - Objective: `val_loss`
  - 50 trials, 10 epochs each
  - EarlyStopping used
- **Output**: Saves best model to `models/`

---

## 🔧 Module: `pipeline/predict.py`

### Function: `predict_latest(X_seq, model_path)`
- Loads Keras model
- Predicts returns using latest sequence
- Returns predicted returns

---

## 🔧 Module: `evaluation.py`

### Class: `Evaluator`
- `sharpe_ratio(returns)`
- `max_drawdown(cum_returns)`
- `annualized_return(returns)`
- `annualized_volatility(returns)`

---

## 🔧 Module: `portfolio.py`

### Class: `PortfolioAllocator`
- `equal_weight(n_assets)`
- `mean_variance(predicted_returns, cov_matrix)`
- `risk_parity(cov_matrix)`

---

## 🔧 Airflow DAG: `retrain_pipeline.py`

- DAG: `weekly_retrain_pipeline`
- **Schedule**: `@weekly`
- **Steps**:
  - Load and append new data
  - Create sequences
  - Train model with tuning
  - Save updated model
- **Features**:
  - `catchup=True` → handles backfills

---

## 🔧 Streamlit App: `streamlit_app.py`

- Reads last 30 days from `price.csv`
- Predicts next returns using saved model
- Computes portfolio weights
- Calculates metrics using `Evaluator`
- Visualizes:
  - Predicted next returns
  - Cumulative backtest returns
  - Residual histogram
- Allows user to select strategy

---

## 📁 I/O Summary

| Component        | Input                        | Output                        |
|------------------|------------------------------|-------------------------------|
| Data Loader      | tickers, existing CSV        | updated `price.csv`           |
| Preprocessor     | CSV, window                  | X, y sequences                |
| Model            | X_train, y_train             | `hybrid_lstm_transformer_model.h5`            |
| Predictor        | Latest X                     | Next-step returns             |
| Evaluator        | Returns                      | Metrics + Plots               |
| Streamlit        | `price.csv`, model           | Live prediction UI            |