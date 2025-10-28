import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from joblib import load
import plotly.graph_objects as go

# Load model and scaler
model = load("volatility_model.pkl")
scaler = load("scaler.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Stock Volatility Predictor", layout="wide")
st.title("📊 Stock Volatility Prediction App")
st.markdown("Predict next-day volatility using an ML model trained on historical stock data.")

# User input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "TCS.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-01"))

# Fetch stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty:
    st.error("No data found. Please check symbol or dates.")
    st.stop()

# Feature engineering
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=5).std() * np.sqrt(252)
data['MA_5'] = data['Close'].rolling(5).mean()
data['MA_10'] = data['Close'].rolling(10).mean()
data['MA_20'] = data['Close'].rolling(20).mean()
data['Volume_Change'] = data['Volume'].pct_change()
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['Vol_Lag1'] = data['Volatility'].shift(1)
data['Vol_Lag2'] = data['Volatility'].shift(2)
data.dropna(inplace=True)

# Prepare features
X_new = data[['Return','MA_5','MA_10','MA_20','Volume_Change','Lag1','Lag2','Vol_Lag1','Vol_Lag2']]
X_scaled = scaler.transform(X_new)
data['Predicted Volatility'] = model.predict(X_scaled)

# Display chart
fig = go.Figure()
fig.add_trace(go.Scatter(y=data['Volatility'], mode='lines', name='Actual Volatility', line=dict(color='blue')))
fig.add_trace(go.Scatter(y=data['Predicted Volatility'], mode='lines', name='Predicted Volatility', line=dict(color='orange', dash='dash')))
fig.update_layout(title=f"{stock_symbol} Volatility Prediction", xaxis_title="Days", yaxis_title="Volatility")
st.plotly_chart(fig, use_container_width=True)

# Next-day prediction
last_row = X_new.iloc[-1:]
scaled_last = scaler.transform(last_row)
next_day_pred = model.predict(scaled_last)[0]
st.subheader(f"📈 Predicted Next-Day Volatility for {stock_symbol}: {next_day_pred:.4f}")

st.dataframe(data.tail(10))
