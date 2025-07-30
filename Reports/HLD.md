# 🧩 High-Level Design (HLD)

## 📌 Project: Hybrid Deep Learning Model for Financial Time Series Forecasting

---

## 🎯 Objective

To build an end-to-end, modular, and automated **ML pipeline** that:

- Predicts asset returns using a hybrid **LSTM + Transformer** model.
- Allocates portfolios using financial strategies (Equal Weight, Mean-Variance, Risk-Parity).
- Automates **weekly retraining** using **Apache Airflow**.
- Visualizes **portfolio performance** via a **Streamlit dashboard**.
- Uses data from **Yahoo Finance**, updates it incrementally, and supports backfill.

---

## 🗂️ Folder Structure

```
quant_finance_pipeline/
├── pipeline/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   ├── portfolio.py
│   ├── metrics.py
│   └── __init__.py
├── data/
│   └── price.csv
├── models/
│   └── hybrid_lstm_transformer_model.h5
├── airflow/
│   └── dags/
│       └── retrain_pipeline.py
├── reports/
│   ├── HLD.md
│   ├── LLD.md
│   └── Hybrid_DL_Timeseries_report.pdf
└── streamlit_app.py
```

---

## 🛠️ Components

### 📈 Data Loader (`pipeline/data_loader.py`)
- Loads asset prices from Yahoo Finance
- Computes daily returns
- Incrementally updates `price.csv`

### 🧹 Preprocessing (`pipeline/preprocessing.py`)
- Converts return matrix into `(X, y)` sequences for modeling
- Sliding window approach (default: 30 days)

### 🧠 Model (`pipeline/model.py`)
- Hybrid deep learning architecture:
  - LSTM for temporal pattern learning
  - Transformer for attention across time
- Fully tunable via Keras-Tuner

### 🔁 Training (`pipeline/train.py`)
- Uses `RandomSearch` from `keras-tuner`
- Saves best model to disk
- Logs to `keras_tuner_logs/`

### 📊 Evaluation & Metrics
- `evaluation.py`: Sharpe Ratio, Max Drawdown, Annualized Return, Volatility, Directional Accuracy
- `metrics.py`: MSE, R², residual plots, cumulative returns
- `portfolio.py`: Equal Weight, Mean-Variance, Risk-Parity strategies

---

## ⏱ Automation: Apache Airflow

### `airflow/dags/retrain_pipeline.py`
- Runs weekly (`@weekly`)
- Updates price data
- Prepares sequences
- Retrains and tunes model
- Saves updated model
- `catchup=True` for backfilling

---

## 💻 Visualization: Streamlit

### `streamlit_app.py`
- Loads the last 30-day window from `price.csv`
- Predicts next asset returns
- Allocates portfolio
- Computes performance metrics
- Plots cumulative return and residuals
- Allows allocation strategy selection

---

## ⚙️ Environment

- **Python**: `3.10.13`
- **Key Libraries**:
  - TensorFlow / Keras
  - Keras-Tuner
  - Pandas, NumPy, Matplotlib
  - yFinance
  - Streamlit
  - Apache Airflow

---

## 🔐 Extensibility

- Add more tickers easily
- Extend retraining frequency or rolling windows
- Plug in RL agents or VaR-style risk models
- Serve model as REST API or batch job
