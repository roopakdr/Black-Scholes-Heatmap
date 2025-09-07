import os
import math
import pandas as pd
import streamlit as st
import altair as alt
from scipy.stats import norm
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"



def black_scholes(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

def get_payoff_diagram(K, option_price, option_type):
    prices = list(range(0, int(K * 2)))
    if option_type == "Call":
        payoff = [max(p - K, 0) - option_price for p in prices]
    else:
        payoff = [max(K - p, 0) - option_price for p in prices]

    df = pd.DataFrame({
        "Stock Price at Expiration": prices,
        "Payoff": payoff
    })
    return df

# Streamlit app
st.set_page_config(page_title="Black-Scholes Calculator", layout="centered")
st.title("📈 Black-Scholes Option Pricing Calculator")

with st.form("bs_form"):
    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Stock Price (S)", value=100.0, min_value=0.01)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01)
        T = st.number_input("Time to Maturity (T, in years)", value=1.0, min_value=0.01)

    with col2:
        r = st.number_input("Risk-Free Rate (r)", value=0.05, format="%.4f")
        sigma = st.number_input("Volatility (σ)", value=0.2, format="%.4f", min_value=0.0001)

    submitted = st.form_submit_button("Calculate")

if submitted:
    call_price, put_price = black_scholes(S, K, T, r, sigma)

    st.success(f"💰 Call Option Price: **${call_price:.4f}**")
    st.success(f"💼 Put Option Price:  **${put_price:.4f}**")

    st.subheader("📊 Payoff at Expiration")

    # Select which payoff to show
    option_type = st.selectbox("Choose option type to display payoff", ["Call", "Put"])
    option_price = call_price if option_type == "Call" else put_price

    df_payoff = get_payoff_diagram(K, option_price, option_type)

    chart = alt.Chart(df_payoff).mark_line().encode(
        x=alt.X("Stock Price at Expiration", title="Stock Price"),
        y=alt.Y("Payoff", title="Profit / Loss"),
        tooltip=["Stock Price at Expiration", "Payoff"]
    ).properties(
        width=600,
        height=400,
        title=f"{option_type} Option Payoff at Expiration"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
