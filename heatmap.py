import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes pricing function
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.set_page_config(page_title="Option Return Heatmap (Matplotlib)", layout="wide")
st.title("📊 Option Return Heatmap with Sliders & 'Hover' Info (Matplotlib)")

option_type = st.radio("Option Type", ["Call", "Put"])
K = st.slider("Strike Price", 50.0, 150.0, 100.0, 1.0)
r = st.slider("Risk-Free Rate (%)", 0.0, 0.10, 0.05, 0.005)
sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
premium_paid = st.slider("Premium Paid", 1.0, 50.0, 10.0, 0.5)

spot_min, spot_max = st.slider("Spot Price Range", 50.0, 200.0, (90.0, 110.0), 1.0)
t_min, t_max = st.slider("Maturity Range (Years)", 0.01, 2.0, (0.1, 0.3), 0.01)

spot_prices = np.linspace(spot_min, spot_max, 50)
maturities = np.linspace(t_min, t_max, 50)

# Calculate % Return matrix
percent_returns = np.empty((len(maturities), len(spot_prices)))
for i, T in enumerate(maturities):
    for j, S in enumerate(spot_prices):
        option_price = black_scholes_price(S, K, T, r, sigma, option_type.lower())
        percent_returns[i, j] = ((option_price - premium_paid) / premium_paid) * 100

# Find break-even points (within ±1%)
break_even_indices = np.argwhere((percent_returns >= -1) & (percent_returns <= 1))

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.get_cmap('RdYlGn')
im = ax.imshow(percent_returns, aspect='auto', origin='lower',
               extent=[spot_min, spot_max, t_min, t_max],
               vmin=-100, vmax=200,
               cmap=cmap)

# Break-even markers
if break_even_indices.size > 0:
    be_spots = spot_prices[break_even_indices[:,1]]
    be_mats = maturities[break_even_indices[:,0]]
    ax.scatter(be_spots, be_mats, marker='x', color='black', label='Break-even (±1%)')

ax.set_xlabel("Spot Price")
ax.set_ylabel("Time to Maturity (Years)")
ax.set_title(f"Option % Return Heatmap ({option_type})")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("% Return")

ax.legend()

# Simulate hover: select spot and maturity values with sliders
st.sidebar.header("Check % Return at Specific Point")
spot_point = st.sidebar.slider("Spot Price", spot_min, spot_max, float((spot_min+spot_max)/2), 0.1)
maturity_point = st.sidebar.slider("Maturity (Years)", t_min, t_max, float((t_min+t_max)/2), 0.01)

# Find closest indices
closest_spot_idx = (np.abs(spot_prices - spot_point)).argmin()
closest_mat_idx = (np.abs(maturities - maturity_point)).argmin()
point_return = percent_returns[closest_mat_idx, closest_spot_idx]

st.sidebar.markdown(f"### % Return at Spot={spot_point:.2f}, T={maturity_point:.3f} yrs:")
st.sidebar.markdown(f"**{point_return:.2f}%**")

st.pyplot(fig)
