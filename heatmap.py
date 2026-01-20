import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Black-Scholes functions with Greeks
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Enhanced Black-Scholes pricing with edge case handling"""
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(price, 0)  # Ensure non-negative prices

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def calculate_implied_volatility(market_price, S, K, T, r, option_type='call', max_iterations=100):
    """Calculate implied volatility using Newton-Raphson method"""
    if T <= 0:
        return 0
    
    # Initial guess
    sigma = 0.2
    tolerance = 1e-6
    
    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = S * norm.pdf((np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))) * np.sqrt(T)
        
        if abs(vega) < 1e-10:
            break
            
        diff = market_price - price
        if abs(diff) < tolerance:
            return sigma
            
        sigma = sigma + diff / vega
        sigma = max(sigma, 0.001)  # Ensure positive volatility
        
    return sigma

# Page configuration
st.set_page_config(
    page_title="Advanced Options P/L Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Black-Scholes Options P/L Dashboard</h1>
    <p>Interactive heatmaps, Greeks analysis, and risk metrics for options trading</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("🎯 Option Parameters")
    
    # Option basics
    option_type = st.selectbox("Option Type", ["Call", "Put"], help="Select call or put option")
    
    col1, col2 = st.columns(2)
    with col1:
        S0 = st.number_input("Current Stock Price", min_value=1.0, value=100.0, step=1.0)
        K = st.number_input("Strike Price", min_value=1.0, value=100.0, step=1.0)
    with col2:
        r = st.slider("Risk-Free Rate (%)", 0.0, 15.0, 5.0, 0.1) / 100
        sigma = st.slider("Volatility (%)", 1.0, 100.0, 20.0, 1.0) / 100
    
    premium_paid = st.number_input("Premium Paid", min_value=0.1, value=5.0, step=0.1)
    
    st.subheader("📊 Analysis Ranges")
    stock_range = st.slider("Stock Price Range (%)", -50, 100, (-20, 20), 5)
    time_range = st.slider("Time to Expiry (Days)", 1, 365, (7, 90), 1)
    
    # Calculate absolute ranges
    stock_min = S0 * (1 + stock_range[0]/100)
    stock_max = S0 * (1 + stock_range[1]/100)
    t_min = time_range[0] / 365
    t_max = time_range[1] / 365

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["P/L Heatmap", "Greeks Analysis", "Risk Metrics", "Scenario Analysis"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Generate data for heatmap
        n_stocks = 50
        n_times = 40
        stock_prices = np.linspace(stock_min, stock_max, n_stocks)
        time_to_expiry = np.linspace(t_min, t_max, n_times)
        
        # Calculate P/L matrix
        pnl_matrix = np.zeros((n_times, n_stocks))
        percent_return_matrix = np.zeros((n_times, n_stocks))
        
        for i, T in enumerate(time_to_expiry):
            for j, S in enumerate(stock_prices):
                option_price = black_scholes_price(S, K, T, r, sigma, option_type.lower())
                pnl = option_price - premium_paid
                pnl_matrix[i, j] = pnl
                percent_return_matrix[i, j] = (pnl / premium_paid) * 100 if premium_paid > 0 else 0
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=percent_return_matrix,
            x=stock_prices,
            y=time_to_expiry * 365,  # Convert to days for display
            colorscale='RdYlGn',
            zmid=0,
            hovertemplate='<b>Stock Price:</b> $%{x:.2f}<br>' +
                         '<b>Days to Expiry:</b> %{y:.0f}<br>' +
                         '<b>Return:</b> %{z:.1f}%<br>' +
                         '<extra></extra>',
            colorbar=dict(title="Return (%)")
        ))
        
        # Add break-even line
        break_even_stocks = []
        break_even_times = []
        for i, T in enumerate(time_to_expiry):
            for j, S in enumerate(stock_prices):
                if abs(percent_return_matrix[i, j]) <= 2:  # Within 2% of break-even
                    break_even_stocks.append(S)
                    break_even_times.append(T * 365)
        
        if break_even_stocks:
            fig.add_trace(go.Scatter(
                x=break_even_stocks,
                y=break_even_times,
                mode='markers',
                marker=dict(color='black', symbol='x', size=8),
                name='Break-even Points',
                hovertemplate='<b>Break-even Point</b><br>' +
                             'Stock: $%{x:.2f}<br>' +
                             'Days: %{y:.0f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"{option_type} Option P/L Heatmap - Strike: ${K:.0f}",
            xaxis_title="Stock Price ($)",
            yaxis_title="Days to Expiry",
            height=500,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Point Analysis")
        
        # Point selection
        selected_stock = st.number_input(
            "Stock Price", 
            min_value=float(stock_min), 
            max_value=float(stock_max), 
            value=S0, 
            step=0.5
        )
        
        selected_days = st.number_input(
            "Days to Expiry", 
            min_value=1, 
            max_value=365, 
            value=30, 
            step=1
        )
        
        selected_T = selected_days / 365
        
        # Calculate metrics for selected point
        option_price = black_scholes_price(selected_stock, K, selected_T, r, sigma, option_type.lower())
        pnl = option_price - premium_paid
        percent_return = (pnl / premium_paid) * 100 if premium_paid > 0 else 0
        
        # Display metrics
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Option Price</h4>
            <h2>${option_price:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>P/L</h4>
            <h2 style="color: {'green' if pnl >= 0 else 'red'}">${pnl:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Return</h4>
            <h2 style="color: {'green' if percent_return >= 0 else 'red'}">{percent_return:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.subheader("Greeks Analysis Dashboard")
    
    # Greeks calculation for current parameters
    greeks = calculate_greeks(S0, K, 30/365, r, sigma, option_type.lower())
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Delta (Δ)", f"{greeks['delta']:.3f}", help="Price sensitivity to underlying stock")
    with col2:
        st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}", help="Delta sensitivity to underlying stock")
    with col3:
        st.metric("Theta (Θ)", f"{greeks['theta']:.3f}", help="Time decay per day")
    with col4:
        st.metric("Vega (ν)", f"{greeks['vega']:.3f}", help="Volatility sensitivity")
    with col5:
        st.metric("Rho (ρ)", f"{greeks['rho']:.3f}", help="Interest rate sensitivity")
    
    # Greeks heatmaps
    greek_type = st.selectbox("Select Greek for Heatmap", ["Delta", "Gamma", "Theta", "Vega"])
    
    # Calculate selected Greek matrix
    greek_matrix = np.zeros((n_times, n_stocks))
    greek_key = greek_type.lower()
    
    for i, T in enumerate(time_to_expiry):
        for j, S in enumerate(stock_prices):
            greeks_point = calculate_greeks(S, K, T, r, sigma, option_type.lower())
            greek_matrix[i, j] = greeks_point[greek_key]
    
    # Create Greek heatmap
    fig_greek = go.Figure(data=go.Heatmap(
        z=greek_matrix,
        x=stock_prices,
        y=time_to_expiry * 365,
        colorscale='Viridis',
        hovertemplate=f'<b>Stock Price:</b> $%{{x:.2f}}<br>' +
                     '<b>Days to Expiry:</b> %{y:.0f}<br>' +
                     f'<b>{greek_type}:</b> %{{z:.4f}}<br>' +
                     '<extra></extra>',
        colorbar=dict(title=greek_type)
    ))
    
    fig_greek.update_layout(
        title=f"{greek_type} Heatmap",
        xaxis_title="Stock Price ($)",
        yaxis_title="Days to Expiry",
        height=400
    )
    
    st.plotly_chart(fig_greek, use_container_width=True)

with tab3:
    st.subheader("💰 Risk Metrics & Portfolio Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Value at Risk (VaR)")
        
        # VaR calculation parameters
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
        holding_period = st.number_input("Holding Period (Days)", 1, 30, 1)
        
        # Calculate VaR
        current_price = black_scholes_price(S0, K, 30/365, r, sigma, option_type.lower())
        daily_vol = sigma / np.sqrt(252)  # Convert annual to daily volatility
        z_score = norm.ppf(confidence_level)
        
        var_amount = current_price * z_score * daily_vol * np.sqrt(holding_period)
        
        st.metric(f"VaR ({confidence_level*100:.0f}%)", f"${var_amount:.2f}")
        st.write(f"With {confidence_level*100:.0f}% confidence, the maximum loss over {holding_period} day(s) will not exceed ${var_amount:.2f}")
    
    with col2:
        st.subheader("Probability Analysis")
        
        # Probability of profit
        target_return = st.slider("Target Return (%)", -50, 200, 0)
        target_price = premium_paid * (1 + target_return/100)
        
        # Calculate probability using Black-Scholes
        if target_return == 0:
            prob_text = "Break-even probability"
        elif target_return > 0:
            prob_text = f"Probability of {target_return}%+ return"
        else:
            prob_text = f"Probability of {target_return}% loss or worse"
        
        st.write(f"**{prob_text}:** Calculating...")
        
        # Monte Carlo simulation for probability
        n_simulations = 10000
        T_target = 30/365  # 30 days
        
        final_stocks = S0 * np.exp((r - 0.5 * sigma**2) * T_target + 
                                 sigma * np.sqrt(T_target) * np.random.standard_normal(n_simulations))
        
        final_option_prices = np.array([
            black_scholes_price(S, K, 0.001, r, sigma, option_type.lower()) 
            for S in final_stocks
        ])
        
        prob_target = np.mean(final_option_prices >= target_price) * 100
        st.metric(prob_text, f"{prob_target:.1f}%")

with tab4:
    st.subheader("📊 Scenario Analysis")
    
    # Scenario parameters
    scenarios = {
        "Bull Market": {"stock_change": 0.15, "vol_change": -0.05},
        "Bear Market": {"stock_change": -0.20, "vol_change": 0.10},
        "High Volatility": {"stock_change": 0.0, "vol_change": 0.15},
        "Low Volatility": {"stock_change": 0.0, "vol_change": -0.10},
        "Market Crash": {"stock_change": -0.30, "vol_change": 0.25}
    }
    
    scenario_results = []
    base_price = black_scholes_price(S0, K, 30/365, r, sigma, option_type.lower())
    
    for scenario_name, changes in scenarios.items():
        new_stock = S0 * (1 + changes["stock_change"])
        new_vol = max(sigma + changes["vol_change"], 0.01)
        new_price = black_scholes_price(new_stock, K, 30/365, r, new_vol, option_type.lower())
        
        pnl = new_price - premium_paid
        return_pct = (pnl / premium_paid) * 100 if premium_paid > 0 else 0
        
        scenario_results.append({
            "Scenario": scenario_name,
            "Stock Change": f"{changes['stock_change']*100:+.0f}%",
            "Vol Change": f"{changes['vol_change']*100:+.0f}%",
            "New Option Price": f"${new_price:.2f}",
            "P/L": f"${pnl:.2f}",
            "Return": f"{return_pct:.1f}%"
        })
    
    df_scenarios = pd.DataFrame(scenario_results)
    st.dataframe(df_scenarios, use_container_width=True)
    
    # Scenario visualization
    fig_scenario = go.Figure()
    
    returns = [float(result["Return"].replace('%', '')) for result in scenario_results]
    scenarios_names = [result["Scenario"] for result in scenario_results]
    
    colors = ['green' if r >= 0 else 'red' for r in returns]
    
    fig_scenario.add_trace(go.Bar(
        x=scenarios_names,
        y=returns,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in returns],
        textposition='auto'
    ))
    
    fig_scenario.update_layout(
        title="Scenario Analysis - Expected Returns",
        xaxis_title="Market Scenario",
        yaxis_title="Expected Return (%)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_scenario, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Black-Scholes Options Analysis Dashboard</p>
</div>
""", unsafe_allow_html=True)
