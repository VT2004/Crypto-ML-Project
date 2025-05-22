# app.py
import streamlit as st
import streamlit as st
import numpy as np
import joblib

# Load your trained model and scaler
model = joblib.load('liquidity_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🔮 Crypto Liquidity Predictor")
st.markdown("Enter cryptocurrency market data to predict liquidity ratio.")

# Input fields
price = st.number_input("Price ($)", min_value=0.0, format="%.6f")
volatility = st.number_input("Volatility", min_value=0.0, format="%.6f")
h1 = st.number_input("1h % Change", format="%.6f")
h24 = st.number_input("24h % Change", format="%.6f")
d7 = st.number_input("7d % Change", format="%.6f")

if st.button("Predict Liquidity Ratio"):
    # Prepare input and scale
    input_data = np.array([[price, volatility, h1, h24, d7]])
    input_scaled = scaler.transform(input_data)
    
    # Predict log liquidity and reverse log transform
    log_pred = model.predict(input_scaled)
    pred = np.expm1(log_pred[0])  # Reverse log1p
    
st.success(f"📉 Predicted Liquidity Ratio: **{pred:.4f}**")
