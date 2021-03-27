"""
Student Name: Chen Peng
GT User ID: cpeng78
GT ID: 903646937
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import util as ut


def author():
        return 'cpeng78'

def compute_portvals(
    record,
    start_val=100000,
    commission=9.5,
    impact=0.005,
):

    # 1. Read in the price table
    dates = pd.date_range(record.index.min(), record.index.max())
    # Make the price table
    prices = ut.get_data(record.columns, dates, addSPY=False).assign(CASH=1.)
    prices.dropna(axis=0, how='any', inplace=True)

    # 2. Read in trades history
    trades = pd.DataFrame(np.zeros(prices.shape), index=prices.index, columns=prices.columns)
    trades.loc[dates[0], ['CASH']] = start_val

    trades[record != 0.] += record  # stock position change
    trades.loc[trades.iloc[:, 0] > 0, 'CASH'] -= trades.iloc[:, 0] * prices.iloc[:, 0] * (1 + impact) + commission  # buy stock cash change
    trades.loc[trades.iloc[:, 0] < 0, 'CASH'] -= trades.iloc[:, 0] * prices.iloc[:, 0] * (1 - impact) + commission  # sell stock cash change
    # print(trades)

    # 3. Compute holdings
    holdings = trades.cumsum(axis='index')

    # 4. Obtain value of each stock
    value = holdings.multiply(prices)

    # 5. Get the portfolio value
    portvals = value.sum(axis='columns')

    return portvals.to_frame(name='portvals')


def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", "../data/")
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, addSPY=True, colname="Adj Close"):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and "SPY" not in symbols:  # add SPY for reference, if absent
        symbols = ["SPY"] + list(
            symbols
        )  # handles the case where symbols is np array of 'object'

    for symbol in symbols:
        df_temp = pd.read_csv(
            symbol_to_path(symbol),
            index_col="Date",
            parse_dates=True,
            usecols=["Date", colname],
            na_values=["nan"],
        )
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == "SPY":  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./additional_orders/orders2.csv"
    sv = 1000000

    # Process orders
    symbol = "AAPL"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2010, 1, 15)
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
    trades = prices_all[[symbol, ]]  # only portfolio symbols
    trades_SPY = prices_all["SPY"]  # only SPY, for comparison later
    trades.values[:, :] = 0  # set them all to nothing
    trades.values[0, :] = 1000  # add a BUY at the start
    trades.values[2, :] = -1000  # add a SELL
    trades.values[3, :] = 1000  # add a BUY
    trades.values[5, :] = -2000  # go short from long
    trades.values[6, :] = 2000  # go long from short
    trades.values[-1, :] = -1000  # exit on the last day

    portvals = compute_portvals(trades, start_val=1000000, commission=0.0, impact=0.0)
    print(trades)
    print(portvals)


    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    daily_returns = compute_daily_returns(portvals)
    price_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date), addSPY=True)['$SPX']
    daily_returns_SPX = compute_daily_returns(price_SPX)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        portvals.iloc[-1] / portvals.iloc[0] - 1,
        daily_returns.mean(),
        daily_returns.std(),
        np.sqrt(252) * daily_returns.mean() / daily_returns.std(),
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        price_SPX.iloc[-1] / price_SPX.iloc[0] - 1,
        daily_returns_SPX.mean(),
        daily_returns_SPX.std(),
        np.sqrt(252) * daily_returns_SPX.mean() / daily_returns_SPX.std()
    ]
    # print(price_SPX, daily_returns_SPX)


    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

def compute_daily_returns(df):
    daily_returns = df / df.shift(1) - 1
    return daily_returns[1:]


if __name__ == "__main__":
    test_code()
