from main import *

# read in all of the data from crypto-price-data.csv

all = pd.read_csv("crypto-price-data.csv", parse_dates=["Date"])

all.set_index("Date", inplace=True)

all.sort_index(inplace=True)

# log all the data such that it can be compared

log_returns = np.log(all / all.shift(1)).dropna()

# plot all of the data

log_returns.plot(figsize=(12,6),title = "Cryptocurrency Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend(loc="best")
plt.ylim(-0.6, 0.6)
plt.grid(True)
plt.show()

# plot an individual cryptocurrency

crypto = "BTC-USD"

plt.figure(figsize=(12, 6))
plt.plot(log_returns.index, log_returns[crypto], label=f"{crypto} Log Returns", color="b")
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.title(f"{crypto} Log Returns Over Time")
plt.legend()
plt.ylim(-0.2, 0.2)
plt.savefig("log-btc.pdf", bbox_inches="tight")
plt.grid(True)
plt.show()

# produce visualisations for the correlation

# visual 1 heatmap

correlation_matrix = log_returns.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Cryptocurrency Correlation Heatmap')
plt.show()


# visual 2 network graphs

G = nx.Graph()

for coin in log_returns.columns:
    G.add_node(coin)

for i, coin1 in enumerate(log_returns.columns):
    for j, coin2 in enumerate(log_returns.columns):
        if i < j:
            correlation_value = correlation_matrix.iloc[i, j]
            if abs(correlation_value) > 0.3:
                G.add_edge(coin1, coin2, weight=correlation_value)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')
nx.draw_networkx_labels(G, pos, font_size=12)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, alpha=0.5, edge_color=weights, edge_cmap=plt.cm.Blues)
plt.title('Cryptocurrency Correlation Network')
plt.show()

# plot of rolling volatility of all the cryptocurrencies

rolling_window = 30  

rolling_vol = log_returns.rolling(window=rolling_window).std()

plt.figure(figsize=(12, 6))
for col in rolling_vol.columns:
    plt.plot(rolling_vol.index, rolling_vol[col], label=col)

plt.title(f'{rolling_window}-Day Rolling Volatility of Cryptocurrencies')
plt.xlabel('Date')
plt.ylabel('Rolling Volatility')
plt.legend()
plt.grid(True)
# plt.savefig("rolling-volatility.pdf", bbox_inches="tight")
plt.show()



