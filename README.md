# Advanced Black-Scholes Options P/L Dashboard

An interactive web application for analyzing options trading strategies using the Black-Scholes model. This dashboard provides comprehensive profit/loss analysis, Greeks calculations, risk metrics, and scenario testing for call and put options.

## Features

### P/L Heatmap
- Interactive 2D heatmap showing profit/loss across different stock prices and time periods
- Break-even point visualization
- Percentage return calculations
- Real-time point analysis with customizable parameters

### Greeks Analysis
- Real-time calculation of all major Greeks:
  - **Delta (Δ)**: Price sensitivity to underlying stock movement
  - **Gamma (Γ)**: Rate of change of Delta
  - **Theta (Θ)**: Time decay per day
  - **Vega (ν)**: Sensitivity to volatility changes
  - **Rho (ρ)**: Sensitivity to interest rate changes
- Interactive heatmaps for each Greek across different market conditions

### Risk Metrics
- **Value at Risk (VaR)**: Calculate potential losses at various confidence levels
- **Probability Analysis**: Monte Carlo simulation for target return probabilities
- Customizable holding periods and confidence intervals

### Scenario Analysis
- Pre-defined market scenarios:
  - Bull Market
  - Bear Market
  - High Volatility
  - Low Volatility
  - Market Crash
- Visual comparison of expected returns across scenarios

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Libraries
```bash
pip install streamlit numpy pandas plotly scipy matplotlib
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.17.0
scipy>=1.11.0
matplotlib>=3.7.0
```

## Usage

### Running the Dashboard
1. Navigate to the project directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. The dashboard will open in your default web browser at `http://localhost:8501`

### Basic Configuration

#### Option Parameters (Sidebar)
- **Option Type**: Select Call or Put
- **Current Stock Price**: The current price of the underlying asset
- **Strike Price**: The option's strike price
- **Risk-Free Rate**: Annual risk-free interest rate (%)
- **Volatility**: Annual implied volatility (%)
- **Premium Paid**: The premium you paid for the option

#### Analysis Ranges
- **Stock Price Range**: Define the range of stock prices to analyze (% from current)
- **Time to Expiry**: Set the range of time periods to analyze (in days)

## Understanding the Dashboard

### P/L Heatmap Tab
The heatmap visualizes your option's profit/loss across different scenarios:
- **X-axis**: Stock prices
- **Y-axis**: Days until expiration
- **Color**: Return percentage (green = profit, red = loss)
- **Black markers**: Break-even points

**Point Analysis Panel** allows you to examine specific scenarios with exact P/L calculations.

### Greeks Analysis Tab
Monitor how your option's value changes with market conditions:
- View all Greeks at a glance
- Select individual Greeks for detailed heatmap analysis
- Understand risk exposure and hedging opportunities

### Risk Metrics Tab
Quantify your risk exposure:
- **VaR Calculation**: Understand maximum potential loss at your chosen confidence level
- **Probability Analysis**: See the likelihood of achieving your target return

### Scenario Analysis Tab
Test your position against predefined market conditions:
- Compare returns across different market scenarios
- Understand best and worst-case outcomes
- Visual bar chart for easy comparison

## Mathematical Background

### Black-Scholes Formula
The dashboard uses the Black-Scholes model for European options pricing:

**Call Option:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
- `d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)`
- `d₂ = d₁ - σ√T`

### Greeks Calculations
- **Delta**: ∂V/∂S (first derivative with respect to stock price)
- **Gamma**: ∂²V/∂S² (second derivative with respect to stock price)
- **Theta**: -∂V/∂t (time decay)
- **Vega**: ∂V/∂σ (volatility sensitivity)
- **Rho**: ∂V/∂r (interest rate sensitivity)

## Important Notes

### Assumptions
- European-style options (can only be exercised at expiration)
- No dividends on the underlying stock
- Constant volatility and interest rates
- Efficient markets with no transaction costs
- Log-normal distribution of stock prices

### Limitations
- Real market conditions may differ from Black-Scholes assumptions
- Does not account for American-style early exercise
- Dividends are not considered
- Transaction costs and slippage are excluded
- Implied volatility may not remain constant

## Use Cases

1. **Options Strategy Evaluation**: Compare potential outcomes before entering a trade
2. **Risk Assessment**: Understand maximum loss and probability of profit
3. **Portfolio Hedging**: Use Greeks to balance portfolio risk
4. **Educational Tool**: Learn how options pricing works in different scenarios
5. **Backtesting**: Analyze historical "what-if" scenarios

## Contributing

Contributions are welcome! Some areas for enhancement:
- Add support for American options
- Implement dividend adjustments
- Add multi-leg strategy analysis (spreads, straddles, etc.)
- Include real-time data integration
- Add options chains visualization

## License

This project is for educational and analytical purposes.

## Support

For issues or questions:
- Review the parameter descriptions in the sidebar
- Hover over metrics for additional information
- Adjust ranges if heatmaps appear empty or unclear

## Future Enhancements

- [ ] Multi-leg options strategies
- [ ] Real-time market data integration
- [ ] Historical volatility analysis
- [ ] Options chain visualization
- [ ] Portfolio-level analysis
- [ ] Export functionality for reports
- [ ] Custom scenario builder
- [ ] Dividend adjustment support

---

**Disclaimer**: This tool is for educational and analytical purposes only. It should not be considered financial advice. Always consult with a qualified financial advisor before making investment decisions.
