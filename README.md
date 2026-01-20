# Advanced Black–Scholes Options P/L Dashboard

An interactive Streamlit app for visualizing Black–Scholes option prices, P/L heatmaps, Greeks, and basic risk metrics using Plotly. [page:1]

## Features

- **P/L** heatmap for call and put options across stock price and time-to-expiry ranges. [page:1]
- Point analysis showing option price, P/L, and percentage return at a specific stock price and days-to-expiry. [page:1]
- Greeks dashboard (Delta, Gamma, Theta, Vega, Rho) with both point metrics and heatmaps over price and time. [page:1]
- Risk metrics tab with a simple Value at Risk (VaR) estimate and Monte Carlo–based probability of achieving a target return. [page:1]
- Scenario analysis for predefined market regimes (Bull, Bear, High Vol, Low Vol, Market Crash) with P/L and return visualization. [page:1]


Install dependencies:

bash
pip install -r requirements.txt
or, if you prefer to install manually:

bash
pip install streamlit numpy pandas plotly scipy matplotlib
These packages cover Streamlit for the UI, NumPy/Pandas for data handling, SciPy for the normal distribution, and Plotly/Matplotlib for visualization. [page:1]

Usage
Run the dashboard with:

streamlit run heatmap.py
Then open the local URL printed in the terminal (usually http://localhost:8501) in your browser. [page:1]

How It Works: 
Uses the Black–Scholes formula with safeguards for edge cases (for example, when time to expiry is near zero) to compute option prices. [page:1]

Computes Greeks analytically and exposes them as both scalar metrics and heatmaps over a grid of stock prices and times. [page:1]

Simulates stock prices via geometric Brownian motion to estimate probabilities of hitting a target return over a specified horizon. [page:1]

Provides predefined market scenarios by shocking spot price and volatility, then recomputing option value, P/L, and return. [page:1]

Inputs and Controls
The sidebar lets you configure: [page:1]

Option type: call or put

Current stock price and strike

Risk-free rate and implied volatility (annualized)

Premium paid for the option

Stock price range

time-to-expiry range (days)

Tabs provide: [page:1]

P/L Heatmap: percent return grid with break-even markers

Greeks Analysis: metrics and per-Greek heatmaps

Risk Metrics: VaR and Monte Carlo probability metrics

Scenario Analysis: table and bar chart of scenario returns

Disclaimers:
This project is for educational and research purposes only and does not constitute financial, investment, trading, or legal advice.

The Black–Scholes model and all simulations here rely on simplifying assumptions that may not hold in real markets; outputs can be materially inaccurate or incomplete.

Past or simulated performance is not indicative of future results. Use any outputs at your own risk.

Always consult a qualified financial professional before making trading or investment decisions.
