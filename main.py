# all the necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from hmmlearn.hmm import GaussianHMM
import seaborn as sns
import networkx as nx

# read in the data

all = pd.read_csv("crypto-price-data.csv", parse_dates=["Date"])
all.set_index("Date", inplace=True)
all.sort_index(inplace=True)
log_returns = np.log(all / all.shift(1)).dropna()

# MVO application

cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD','SOL-USD','LINK-USD','BNB-USD','MATIC-USD']
# 'BTC-USD', 'ETH-USD', 'XRP-USD','SOL-USD','LINK-USD','BNB-USD','MATIC-USD'

mean_returns = log_returns[cryptos].mean()
cov_matrix = log_returns[cryptos].cov()

num_portfolios = 10000

rf_rate = 0 # placeholder can change based on 10y treasury bond returns

results = np.zeros((3, num_portfolios))

portfolio_weights = np.zeros((num_portfolios, len(cryptos)))

for i in range(num_portfolios):
    # randomly generate portfolio weights
    weights = np.random.random(len(cryptos))
    weights /= np.sum(weights)  # normalize to sum to 1
    
    # calculate portfolio returns and volatility
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Store the results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = (portfolio_return - rf_rate) / portfolio_volatility  # Sharpe Ratio
    portfolio_weights[i, :] = weights  # Store the portfolio weights

results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'])

# find the portfolio with the highest Sharpe ratio (optimal portfolio)
max_sharpe_idx = results_df['Sharpe'].idxmax()
max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]

# get the optimal portfolio weights
optimal_weights = portfolio_weights[max_sharpe_idx]

# print the optimal portfolio weights for each cryptocurrency
print("Optimal Portfolio Weights (for Max Sharpe Ratio Portfolio):")
for i, crypto in enumerate(cryptos):
    print(f"{crypto}: {optimal_weights[i]:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier with Cryptocurrencies')

# plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'], marker='*', color='red', s=200, label="Max Sharpe Portfolio")

plt.legend(loc='best')

plt.savefig("efficient-frontier.pdf", bbox_inches="tight")

plt.show()





