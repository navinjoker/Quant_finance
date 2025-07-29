import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from pipeline.evaluation import Evaluator
from pipeline.portfolio import PortfolioAllocator
from pipeline.metrics import directional_accuracy
import os
from pipeline.preprocessing import create_sequences_from_csv
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
# -----------------------------
# 📥 Config
# -----------------------------
MODEL_PATH = "C:/Users/Navin/Desktop/Finance Final/quant_finance_pipeline/models/hybrid_lstm_transformer_model.h5"
DATA_PATH = "C:/Users/Navin/Desktop/Finance Final/quant_finance_pipeline/data/price.csv"
WINDOW = 30  # sequence length used in training

# -----------------------------
# 📊 App Layout
# -----------------------------
st.title("📈 Auto-Predict Portfolio Dashboard")

strategy = st.selectbox("Select Portfolio Allocation Strategy", ["Equal Weight", "Mean-Variance", "Risk-Parity"])

# -----------------------------
# 📤 Predict Next Step
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH)
    tickers = df.columns  
    latest_data = df.values[-WINDOW:]

    model = load_model(MODEL_PATH)
    input_seq = np.expand_dims(latest_data, axis=0)  # (1, window, n_assets)

    pred_return = model.predict(input_seq)[0]  # (n_assets,)

    st.subheader("📈 Predicted Next Returns")
    pred_df = pd.DataFrame(pred_return.reshape(1, -1), columns=tickers)
    st.dataframe(pred_df.style.format("{:.4f}"))

    # -----------------------------
    # 🧠 Portfolio Allocation
    # -----------------------------
    evaluator = Evaluator()
    allocator = PortfolioAllocator()
    cov_matrix = np.cov(df.values[-WINDOW:].T)

    if strategy == "Equal Weight":
        weights = allocator.equal_weight(len(pred_return))
    elif strategy == "Mean-Variance":
        weights = allocator.mean_variance(pred_return, cov_matrix)
    else:
        weights = allocator.risk_parity(cov_matrix)

    port_ret = np.dot(pred_return, weights)

    # -----------------------------
    # 🧠 Historical Benchmark
    # -----------------------------
    y_true = df.values[-(WINDOW+1):-1]
    true_returns = np.dot(y_true, weights)

    # -----------------------------
    # 📈 Metrics
    # -----------------------------
    st.subheader("📊 Portfolio Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{evaluator.sharpe_ratio(true_returns):.2f}")
    col2.metric("Annual Return", f"{evaluator.annualized_return(true_returns)*100:.2f}%")
    col3.metric("Volatility", f"{evaluator.annualized_volatility(true_returns)*100:.2f}%")

    col4, col5 = st.columns(2)
    col4.metric("Max Drawdown", f"{evaluator.max_drawdown(np.cumsum(true_returns))*100:.2f}%")
    col5.metric("Directional Accuracy", f"{directional_accuracy(true_returns, [port_ret]*len(true_returns))*100:.2f}%")

    # -----------------------------
    # 📊 Visual
    # -----------------------------
    st.subheader("📉 Simulated Recent Cumulative Return")
    plt.figure(figsize=(10, 4))
    cum = np.cumsum(true_returns)
    plt.plot(cum, label="Backtested")
    plt.axhline(y=cum[-1]+port_ret, color='r', linestyle='--', label="Predicted Next Return")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
