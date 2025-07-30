
# 📊 Hybrid Deep Learning Model for Financial Time Series Forecasting

## 🧠 Overview

This project implements a **hybrid deep learning architecture** combining **LSTM** (Long Short-Term Memory) and **Transformer** blocks to predict weekly returns of multiple financial assets. It uses a modular, automated pipeline built with **Airflow** and a lightweight **Streamlit UI** for visualization and DAG triggering.

---

## 🧾 Project Objective

- Predict next-period (weekly) returns for a basket of financial ETFs.
- Integrate a time series model into an automated, production-ready pipeline.
- Achieve meaningful predictions for use in portfolio strategies, directional signals, or macroeconomic insight.

---

## 📈 Data

- **Source**: Yahoo Finance via `yfinance`
- **Assets**: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, HYG, LQD
- **Preprocessing**:
  - Missing value handling
  - `pct_change()` to remove non-stationarity (partially)
  - Windowing with look-back = 30 time steps
  - Scaling using `StandardScaler`

---

## 🏗️ Model Architecture

- **Input Shape**: `(batch_size, 30, 10)` — 30 weeks history of 10 assets
- **Stack**:
  - LSTM layers (capture sequence dependencies)
  - Transformer Encoder block (capture context and attention across sequences)
  - Dense output layer for multi-output regression
- **Loss**: MSE
- **Optimizer**: Adam
- **Metrics**: MSE, Directional Accuracy, Cumulative Return

---

## 🔧 Hyperparameter Tuning

- **Tool**: `keras-tuner`
- **Search Space**:
  - LSTM units
  - Attention heads
  - Dropout rate
  - Learning rate
- **Limitations**: Due to resource constraints, only 95 combinations over 35 epochs were tested.
- **Observation**: Validation and training loss converged in most trials, suggesting scope for deeper tuning.

---

## 📊 Evaluation Metrics

- **Directional Accuracy** (percent of up/down correctly predicted)
- **Annualized Return & Volatility**
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Cumulative Return** over test window

---

## 🛠️ Airflow Automation

Two DAGs automate the ML lifecycle:

- **`train_ai_finance_model`**: Weekly retraining of model with updated data
- **`predict_ai_finance_model`**: Daily forecasts based on latest model

Each DAG performs:
- Data load
- Preprocessing
- Model training/loading
- Forecast generation
- Result saving and metric tracking

---

## 🌐 Streamlit Dashboard

Provides a simple UI for:
- Triggering DAGs
- Viewing logs
- Optionally visualizing results and performance trends

---

## 🧪 Limitations & Future Work

- Current setup is a **boilerplate architecture**; many extensions possible:
  - Advanced tuning (more combinations, epochs, batch size, embedding dimensions)
  - Improved preprocessing using ADF test insights for better stationarity control
  - Integration with macro indicators or sentiment data
  - Enhanced architectures (TFT, Informer, TCNs)
  - Model interpretability via attention visualization or SHAP
  - Strategy backtesting with predicted signals

> 💡 **Note**: This is a foundational version. ADF test results will guide deeper preprocessing like differencing or specific asset-wise transformations.

---

## 📁 Project Structure

```bash
quant_finance_pipeline/
├── dags/
│   ├── train_ai_finance_model.py
│   └── predict_ai_finance_model.py
├── models/
│   └── best_model.h5
├── data/
│   └── asset_data.csv
├── streamlit_app/
│   └── dashboard.py
├── notebooks/
│   └── development_notebook.ipynb
├── utils/
│   └── preprocessing.py
│   └── metrics.py
└── README.md



🧰 Requirements
Python 3.9+

TensorFlow

Keras

Keras-Tuner

Pandas

NumPy

scikit-learn

yfinance

# Airflow (installing with constraints)
#pip install "apache-airflow==2.7.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.8.txt"

Streamlit


🚀 Getting Started
bash
Always show details

Copy
# Create environment
python -m venv fin_env
source fin_env/bin/activate

# Install dependencies
pip install "apache-airflow==2.7.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.8.txt"
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py

# Run Airflow
airflow webserver -p 8080
airflow scheduler


📬 Contact
For further enhancements, collaboration, or deployment support, feel free to reach out. Navin Dwivedy +917307795703
Sr. Data Scientist / AI /ML Enthusiast
