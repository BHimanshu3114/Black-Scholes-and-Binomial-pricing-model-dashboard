import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import log, sqrt, exp  # Make sure to import these

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes & Binomial Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}
.metric-call {
    background-color: #90ee90;
    color: black;
    margin-right: 10px;
    border-radius: 10px;
}
.metric-put {
    background-color: #ffcccb;
    color: black;
    border-radius: 10px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Black-Scholes Model Class
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (
            log(self.current_price / self.strike)
            + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity
        ) / (self.volatility * sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        )
        put_price = (
            self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        return call_price, put_price


# Binomial Model Class
class BinomialModel:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate, steps):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.steps = steps

    def calculate_prices(self):
        dt = self.time_to_maturity / self.steps
        u = exp(self.volatility * sqrt(dt))
        d = 1 / u
        p = (exp(self.interest_rate * dt) - d) / (u - d)

        stock_prices = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = self.current_price * (u ** (i - j)) * (d ** j)

        call_prices = np.zeros((self.steps + 1, self.steps + 1))
        put_prices = np.zeros((self.steps + 1, self.steps + 1))

        for j in range(self.steps + 1):
            call_prices[j, self.steps] = max(0, stock_prices[j, self.steps] - self.strike)
            put_prices[j, self.steps] = max(0, self.strike - stock_prices[j, self.steps])

        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                call_prices[j, i] = exp(-self.interest_rate * dt) * (
                    p * call_prices[j, i + 1] + (1 - p) * call_prices[j + 1, i + 1]
                )
                put_prices[j, i] = exp(-self.interest_rate * dt) * (
                    p * put_prices[j, i + 1] + (1 - p) * put_prices[j + 1, i + 1]
                )

        self.call_price = call_prices[0, 0]
        self.put_price = put_prices[0, 0]
        return self.call_price, self.put_price


# Sidebar Inputs
st.sidebar.title("📊 Option Pricing Models")
st.sidebar.markdown("Choose inputs for the pricing models:")
current_price = st.sidebar.number_input("Current Asset Price", value=100.0)
strike = st.sidebar.number_input("Strike Price", value=100.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.2)
interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)

# Model selection
pricing_model = st.sidebar.radio("Choose Pricing Model", ["Black-Scholes", "Binomial Model"])

if pricing_model == "Black-Scholes":
    model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = model.calculate_prices()
    model_name = "Black-Scholes"
else:
    steps = st.sidebar.slider("Number of Steps (Binomial Model)", min_value=1, max_value=500, value=100)
    model = BinomialModel(time_to_maturity, strike, current_price, volatility, interest_rate, steps)
    call_price, put_price = model.calculate_prices()
    model_name = "Binomial"

# Display Option Prices
st.title(f"Option Prices - {model_name} Model")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value ({model_name})</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value ({model_name})</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
