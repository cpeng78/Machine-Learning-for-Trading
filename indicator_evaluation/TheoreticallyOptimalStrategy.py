"""
Student Name: Chen Peng
GT User ID: cpeng78
GT ID: 903646937
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
import marketsimcode as ms


def author():
    return "cpeng78"

# TheoreticallyOptimalStrategy
def testPolicy(symbol="AAPL", sd='2010-01-01', ed='2011-12-31', sv=100000):
    dates = pd.date_range(sd, ed)
    syms = [symbol]
    start_val = sv
    prices = get_data(syms, dates).loc[:, syms]
    prices = prices / prices.iloc[0, :]
    daily_return = prices / prices.shift(1) - 1
    daily_acceleration = daily_return * daily_return.shift(1)
    #print(daily_return)
    #print(daily_acceleration)
    a1 = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    a1[daily_return > 0] = 1
    a1[1:] = a1.diff()
    a1.ix[0] = 0
    orders = prices.copy()
    orders.ix[:, :] = np.nan
    orders[(a1 > 0)] = 1000
    orders[(a1 < 0)] = -1000
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[:] = orders.diff().shift(-1)
    orders = orders.loc[(orders != 0).any(axis=1)]
    orders.ix[-1] = -orders.ix[-2] / 2
    order_list = []
    for day in orders.index:
        for sym in syms:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, 'BUY', orders.ix[day, sym]])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, 'SELL', -orders.ix[day, sym]])
    order_list = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list.set_index(order_list['Date'], inplace=True)
    order_list.drop('Date', 1, inplace=True)
    return order_list


# Benchmark Strateggy
def Benchmark(symbol="AAPL", sd='2010-01-01', ed='2011-12-31', sv=100000):
    prices = get_data([symbol], pd.date_range(sd, ed))
    Benchmark_trade = pd.DataFrame({'Date': [prices.index[0], prices.index[-1]],
                                    'Symbol': [symbol, symbol],
                                    'Order': ['BUY', 'SELL'],
                                    'Shares': [1000, 1000]})
    Benchmark_trade.set_index(Benchmark_trade['Date'], inplace=True)
    Benchmark_trade.drop('Date', 1, inplace=True)
    return Benchmark_trade

if __name__== '__main__':
    start_val = 100000
    trade1 = Benchmark()
    portval1 = ms.compute_portvals(trade1, start_val=100000, commission=0.0, impact=0.0)
    print('trade\n', trade1)
    print('portval\n', portval1)

    trade2 = testPolicy()
    portval2 = ms.compute_portvals(trade2, start_val=100000, commission=0.0, impact=0.0)
    print('trade\n', trade2)
    print('portval\n', portval2)

    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.plot(portval1 / start_val, label='Benchmark', color='tab:green')
    plt.plot(portval2 / start_val, label='Theoretically optimal portfolio', color='tab:red')
    plt.legend()
    plt.show()
    plt.close()
