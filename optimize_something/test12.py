import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as spo

#  function to get path of symbol
def symbol_to_path(symbol, base_dir='../data/'):
    return os.path.join(base_dir, '{}.csv'.format(str(symbol)))

def get_data(symbols, dates):
    # Create an empty dataframe
    df = pd.DataFrame(index=dates)
    df_final = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')
        no_SPY = True
    else:
        no_SPY = False
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'],
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close':symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])
    if no_SPY:
        df.drop(columns=['SPY'], inplace=True)
    return df

def compute_daily_returns(df):
    daily_returns = df / df.shift(1) -1
    # daily_returns.iloc[0] = 0
    return daily_returns[1:]


def compute_sharpe_ratio(normed_price, allocs, rfr=0.0, sf=252.0):
    alloced_price = normed_price * allocs
    port_val = alloced_price.sum(1)

    daily_returns = compute_daily_returns(port_val)
    avg_daily_returns = daily_returns.mean()
    std_daily_returns = daily_returns.std()
    sharpe_ratio = np.sqrt(sf) * (avg_daily_returns - rfr) / std_daily_returns

    return -sharpe_ratio


def optimize_portfolio(normed_price, sharpe_ratio):
    # Generate initial portfolio allocations
    alloc_guess = tuple([1/normed_price.shape[1]]*normed_price.shape[1])
    bounds = spo.Bounds([0]*normed_price.shape[1], [1]*normed_price.shape[1])
    #linear_constraint = spo.LinearConstraint([[1, 1, 1, 1], [1], [1]])

    result = spo.minimize(sharpe_ratio, alloc_guess, args=(normed_price,), method='SLSQP',
                          bounds=bounds,
                          options={'disp':True})

    print('The optimized Sharpe ratio is:', -compute_sharpe_ratio(normed_price, result.x))

    return result.x


def test_run():
    dates = pd.date_range('2010-01-01', '2010-12-31')
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = [0.2, 0.3, 0.4, 0.1]
    df = get_data(symbols, dates)
    normed_price = df / df.iloc[0, :]
    #print(df)

    # Compute Sharpe ratio
    sharpe_ratio = -compute_sharpe_ratio(normed_price, allocs)
    print('The Sharpe ratio of the portfolio is:', sharpe_ratio)

    x = optimize_portfolio(normed_price, compute_sharpe_ratio)
    print(x)


if __name__ == '__main__':
    test_run()