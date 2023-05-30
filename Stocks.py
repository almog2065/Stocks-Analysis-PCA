from numpy import linalg as alg
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def Q1():
    # 1
    df = pd.read_csv('prices.csv')
    df.head(5)
    # 2
    mask = df['date'].apply(lambda x: x[:4] == '2016')
    df = df[mask]
    df = df[df['symbol'] == 'AAPL'].reset_index()
    apple_close_prices = df.close
    apple_close_prices.plot()
    plt.show()


def load_shares():
    securities = pd.read_csv('securities.csv')
    prices = pd.read_csv('prices.csv')
    mask = prices['date'].apply(lambda x: x[:4] == '2016')
    prices_df = prices[mask]
    days_cnt_df = pd.DataFrame(prices_df['symbol'].value_counts().reset_index().values, columns=['symbol', 'counter'])
    mask = days_cnt_df['counter'].apply(lambda x: x == 252)
    symbols = days_cnt_df[mask]['symbol'].unique()
    symbols_list = list(symbols)
    sectors = []
    for symbol in symbols:
        mask = securities['Ticker symbol'].apply(lambda x: x == symbol)
        df = securities[mask]
        sectors.append(df['GICS Sector'].values[0])
    symbol_mask = prices_df['symbol'].apply(lambda x: x == symbols_list[0])
    prices_by_symbol = prices_df[symbol_mask]
    prices_mat = np.array([prices_by_symbol.close.values])
    for i in range(1, len(symbols_list)):
        symbol_mask = prices_df['symbol'].apply(lambda x: x == symbols_list[i])
        prices_by_symbol = prices_df[symbol_mask]
        prices_curr = np.array([list(prices_by_symbol['close'])])
        prices_mat = np.concatenate((prices_mat, prices_curr), axis=0)
    return symbols, prices_mat, sectors


def pca_project(X, k):
    sum = X[0]
    for x in X[1:, :]:
        sum = sum + x
    avg = sum / X.shape[0]
    for i, x in enumerate(X):
        X[i] = x - avg
    U, sigma, Vt = alg.svd(X)
    k_eigen = Vt[:k, :].T
    return X @ k_eigen


def plot_sectors(proj, sectors, sectors_to_plot):
    relevant_proj = np.array([proj[0, :]])
    sector_indicator = []
    count = 0
    for i in range(len(sectors)):
        if sectors[i] in sectors_to_plot:
            relevant_proj = np.concatenate((relevant_proj, [proj[i, :]]), axis=0)
            sector_indicator.append((sectors[i], count))
            count += 1
    relevant_proj = relevant_proj[1:, :]
    for sector in sectors_to_plot:
        sector_proj = np.array([proj[0, :]])
        for x in sector_indicator:
            if x[0] == sector:
                sector_proj = np.concatenate((sector_proj, [relevant_proj[x[1], :]]), axis=0)
        sector_proj = sector_proj[1:, :]
        plt.scatter(sector_proj[:, 0], sector_proj[:, 1], label=sector)
    plt.legend()
    plt.show()

def Q5(prices):
    proj1 = pca_project(prices, 2)
    plot_sectors(proj1, sectors, ['Energy', 'Information Technology'])

def Q6(prices):
    ln_prices = np.log(prices)
    f_prices = ln_prices[:, :ln_prices.shape[1]-1]
    l_prices = ln_prices[:, 1:]
    ln_prices = l_prices - f_prices
    proj = pca_project(ln_prices, 2)
    plot_sectors(proj, sectors, ['Energy', 'Information Technology'])

def Q7(prices):
    ln_prices = np.log(prices)
    f_prices = ln_prices[:, :ln_prices.shape[1] - 1]
    l_prices = ln_prices[:, 1:]
    ln_prices = l_prices - f_prices
    proj = pca_project(ln_prices, 2)
    plot_sectors(proj, sectors, ['Financials', 'Information Technology'])
    plot_sectors(proj, sectors, ['Energy', 'Information Technology', 'Real Estate'])


def find_layer_typical(proj):
    col2 = proj[:, 1]
    return np.argsort(col2)[-1], np.argsort(col2)[0]


def Q8(prices, symbols):
    ln_prices = np.log(prices)
    f_prices = ln_prices[:, :ln_prices.shape[1] - 1]
    l_prices = ln_prices[:, 1:]
    ln_prices = l_prices - f_prices
    proj = pca_project(ln_prices, 2)
    plot_sectors(proj, sectors, set(sectors))
    max_index, min_index = find_layer_typical(proj)
    max_symbol = symbols[max_index]
    min_prices = prices[min_index, :]
    max_prices = prices[max_index, :]
    plt.plot([i for i in range(252)], min_prices, label=max_symbol)
    plt.plot([i for i in range(252)], max_prices, label="Typical")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    symbols, prices, sectors = load_shares()

    #Q5
    prices5 = prices.copy()
    Q5(prices5)

    #Q6
    Q6(prices)

    #Q7
    Q7(prices)

    #Q8
    Q8(prices, symbols)


