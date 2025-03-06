import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch historical stock data
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Example portfolio
start_date = '2020-01-01'
end_date = '2024-01-01'

# Download data and check columns
data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)
print(data.head())  # Debugging step

# Use 'Close' instead of 'Adj Close' if auto_adjust=True
if 'Adj Close' in data.columns:
    prices = data['Adj Close']
elif 'Close' in data.columns:
    prices = data['Close']
else:
    raise ValueError("Neither 'Adj Close' nor 'Close' found in data!")

# Calculate daily returns
returns = prices.pct_change().dropna()

# Portfolio simulation parameters
num_simulations = 10000
weights = np.random.dirichlet(np.ones(len(stocks)), num_simulations)  # Random weights
initial_investment = 1_000_000  # $1M

# Portfolio expected returns (Annualized)
expected_returns = np.dot(weights, returns.mean() * 252)  

# Portfolio volatility (Annualized)
cov_matrix = returns.cov() * 252  # Annualized covariance matrix
portfolio_volatility = np.sqrt(np.einsum('ij,ji->i', weights, np.dot(cov_matrix, weights.T)))  

# Assuming a risk-free rate of 2%
risk_free_rate = 0.02
sharpe_ratios = (expected_returns - risk_free_rate) / portfolio_volatility

# Simulating possible portfolio outcomes
simulated_results = pd.DataFrame({'Return': expected_returns, 'Risk': portfolio_volatility, 'Sharpe Ratio': sharpe_ratios})

# Plot Monte Carlo simulation
plt.figure(figsize=(10, 6))
scatter = plt.scatter(simulated_results['Risk'], simulated_results['Return'], c=simulated_results['Sharpe Ratio'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Simulation for Portfolio Optimization')
plt.show()
