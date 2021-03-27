"""
Student Name: Chen Peng
GT User ID: cpeng78
GT ID: 903646937
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import TheoreticallyOptimalStrategy as tos
import marketsimcode as ms

def author():
    return "cpeng78"

def indicator(symbol = "JPM", sd='2008-01-01', ed='2009-12-31', sv = 100000):
    dates = pd.date_range(sd, ed)
    syms = [symbol]
    start_val = sv
    prices = get_data(syms, dates).loc[:, syms]
    prices = prices / prices.iloc[0, :]

    ## Bollinger Bands value BOLL(20, 2)
    SMA = prices.rolling(20).mean()
    stdev = prices.rolling(20).std()
    bb_upper = SMA + 2 * stdev
    bb_lower = SMA - 2 * stdev
    bb_value = (prices - SMA) / (2 * stdev)
    # Create order book
    orders = prices.copy()
    orders.ix[:, :] = np.nan
    orders[(bb_value < 1) & (bb_value.shift() > 1)] = -1000
    orders[(bb_value > -1) & (bb_value.shift() < -1)] = 1000
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[:] = orders.diff().shift(-1)
    orders = orders.loc[(orders != 0).any(axis=1)]
    orders.ix[-1] = -orders.ix[-2] / 2
    order_list = []
    order_list.append([prices.index[0], syms[0], 'BUY', 0],)
    for day in orders.index:
        for sym in syms:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, 'BUY', orders.ix[day, sym]])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, 'SELL', -orders.ix[day, sym]])
    order_list = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list.set_index(order_list['Date'], inplace=True)
    order_list.drop('Date', 1, inplace=True)
    print(order_list)
    # Plot indicator
    pd.plotting.register_matplotlib_converters()
    ax1 = plt.subplot(211)
    plt.plot(prices, label='price')
    plt.title('Bollinger Bands', fontsize=12)
    plt.plot(bb_upper, label='upper Bollinger band')
    plt.plot(bb_lower, label='lower Bollinger band')
    ax1.set_ylabel('Normalized Price')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(212, sharex=ax1)
    plt.title('Bollinger Bands value BOLL[20, 2]', fontsize=12)
    plt.plot(bb_value, label='BOLL(20,2)')
    ax2.set_ylabel('BB ratio (%)')
    ax2.legend()
    plt.savefig('Fig1.png')
    # plt.show()
    plt.close()
    # Plot portfolio returns
    trade1 = tos.Benchmark(symbol="JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval1 = ms.compute_portvals(trade1, start_val=100000, commission=0.0, impact=0.0)
    trade2 = order_list
    portval2 = ms.compute_portvals(trade2, start_val=100000, commission=0.0, impact=0.0)
    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('Bollinger Bands Strategy vs Benchmark', fontsize=12)
    plt.plot(portval1 / start_val, label='Benchmark')
    plt.plot(portval2 / start_val, label='Bollinger Bands value strategy')
    plt.legend()
    # plt.show()
    plt.close()

    ## price/SMA  BIAS12
    SMA = prices.rolling(12).mean()
    BIAS12 = (prices - SMA) / SMA * 100
    # Create order book
    orders = prices.copy()
    orders.ix[:, :] = np.nan
    orders[(BIAS12 > 8)] = 1000
    orders[(BIAS12 < -8)] = -1000
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[:] = orders.diff().shift(-1)
    orders = orders.loc[(orders != 0).any(axis=1)]
    orders.ix[-1] = -orders.ix[-2] / 2
    order_list = []
    order_list.append([prices.index[0], syms[0], 'BUY', 0],)
    for day in orders.index:
        for sym in syms:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, 'BUY', orders.ix[day, sym]])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, 'SELL', -orders.ix[day, sym]])
    order_list = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list.set_index(order_list['Date'], inplace=True)
    order_list.drop('Date', 1, inplace=True)
    print(order_list)
    # Plot indicator
    fig, ax1 = plt.subplots()
    ax1.plot(prices, label='price[t]')
    ax1.plot(SMA, label='SMA(12)')
    ax1.set_ylabel('Normalized Price')
    plt.legend(loc=3)
    ax2 = ax1.twinx()
    ax2.plot(BIAS12, label='BIAS(12)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylabel('Deviation rate [%]', color='tab:green')
    plt.legend(loc=4)
    plt.title('Simple moving average and deviation rate', fontsize=12)
    # ax = plt.plot()
    # plt.plot(prices, label='price')
    # plt.ylabel('Normalized Price')
    # plt.title('Simple moving average and deviation rate', fontsize=12)
    # plt.plot(SMA, label='SMA(12)')
    # plt.plot(prices / SMA, label='BIAS(12)')
    # plt.legend()
    plt.savefig('Fig2.png')
    # plt.show()
    plt.close()
    # Plot portfolio returns
    trade1 = tos.Benchmark(symbol="JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval1 = ms.compute_portvals(trade1, start_val=100000, commission=0.0, impact=0.0)
    trade2 = order_list
    portval2 = ms.compute_portvals(trade2, start_val=100000, commission=0.0, impact=0.0)
    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('BIAS12 strategy vs Benchmark', fontsize=12)
    plt.plot(portval1 / start_val, label='Benchmark')
    plt.plot(portval2 / start_val, label='BIAS12 strategy')
    plt.legend()
    # plt.show()
    plt.close()

    ## momentum = (price(t)/price(t-N))-1  MTM12
    window = 12
    momentum = prices / prices.shift(window) - 1
    # Create order book
    orders = prices.copy()
    orders.ix[:, :] = np.nan
    orders[(momentum < 0) & (momentum.shift() > 0)] = -1000
    orders[(momentum > 0) & (momentum.shift() < 0)] = 1000
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[:] = orders.diff().shift(-1)
    orders = orders.loc[(orders != 0).any(axis=1)]
    orders.ix[-1] = -orders.ix[-2] / 2
    order_list = []
    order_list.append([prices.index[0], syms[0], 'BUY', 0], )
    for day in orders.index:
        for sym in syms:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, 'BUY', orders.ix[day, sym]])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, 'SELL', -orders.ix[day, sym]])
    order_list = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list.set_index(order_list['Date'], inplace=True)
    order_list.drop('Date', 1, inplace=True)
    print(order_list)
    # Plot indicator
    fig, ax1 = plt.subplots()
    ax1.plot(prices, '-', label='price[t]', color='tab:blue')
    ax1.plot(prices.shift(window), ':', label='price[t-N], N=12', color='tab:blue')
    ax1.set_ylabel('Normalized Price', color='tab:blue')
    plt.title('Momentum Rate', fontsize=12)
    plt.legend(loc=3)
    ax2 = ax1.twinx()
    ax2.plot(momentum, label='MR(N)=(price[t]/price[t-N])-1', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Momentum', color='tab:orange')
    plt.legend(loc=4)
    plt.savefig('Fig3.png')
    # plt.show()
    plt.close()
    # Plot portfolio returns
    trade1 = tos.Benchmark(symbol="JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval1 = ms.compute_portvals(trade1, start_val=100000, commission=0.0, impact=0.0)
    trade2 = order_list
    portval2 = ms.compute_portvals(trade2, start_val=100000, commission=0.0, impact=0.0)
    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('MTM12 strategy vs Benchmark', fontsize=12)
    plt.plot(portval1 / start_val, label='Benchmark')
    plt.plot(portval2 / start_val, label='MTM12 strategy')
    plt.legend()
    # plt.show()
    plt.close()

    # Volatility
    volatility = (prices / prices.shift(1) - 1).rolling(10).std()
    fig, ax1 = plt.subplots()
    ax1.plot(prices, '-', label='price[t]')
    ax1.plot(prices / prices.shift(1), '--', label='daily Return')
    ax1.set_ylabel('Normalized Price and Daily return')
    plt.legend(loc=3)
    ax2 = ax1.twinx()
    ax2.plot(volatility, label='volatility', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylabel('Volatility', color='tab:green')
    plt.title('10 days volatility', fontsize=12)
    plt.legend(loc=4)
    plt.savefig('Fig4.png')
    # plt.show()
    plt.close()
    # print(volatility)

    ## Moving Average Convergence and Divergence
    EMA12 = prices.ewm(span=12, adjust=False).mean()
    EMA26 = prices.ewm(span=26, adjust=False).mean()
    DIF = EMA12 - EMA26
    DEA = DIF.ewm(span=9, adjust=False).mean()
    MACD = 2 * (DIF - DEA)
    # Plot indicator
    pd.plotting.register_matplotlib_converters()
    ax1 = plt.subplot(211)
    plt.plot(prices, label='price')
    plt.title('Exponential Moving Average (EMA)', fontsize=12)
    plt.plot(EMA12, label='EMA12')
    plt.plot(EMA26, label='EMA26')
    ax1.set_ylabel('Normalized Price')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(212, sharex=ax1)
    plt.title('Moving Average Convergence / Divergence (MACD)', fontsize=12)
    plt.plot(DIF, label='DIF')
    plt.plot(DEA, label='DEA')
    plt.plot(MACD, label='MACD')
    ax2.legend()
    plt.savefig('Fig5.png')
    # plt.show()
    plt.close()
    # Create order book
    orders = prices.copy()
    orders.ix[:, :] = np.nan
    orders[(DIF > 0) & (DEA > 0) & (DIF > DIF.shift()) & (DEA > DEA.shift())] = 1000
    orders[(DIF < 0) & (DEA < 0) & (DIF < DIF.shift()) & (DEA < DEA.shift())] = -1000
    orders[(DIF > 0) & (DEA > 0) & (DIF < DIF.shift()) & (DEA < DEA.shift())] = -1000
    orders[(DIF < 0) & (DEA < 0) & (DIF > DIF.shift()) & (DEA > DEA.shift())] = 1000
    orders.ffill(inplace=True)
    orders.fillna(0, inplace=True)
    orders[:] = orders.diff().shift(-1)
    orders = orders.loc[(orders != 0).any(axis=1)]
    orders.ix[-1] = -orders.ix[-2] / 2
    order_list = []
    order_list.append([prices.index[0], syms[0], 'BUY', 0],)
    for day in orders.index:
        for sym in syms:
            if orders.ix[day, sym] > 0:
                order_list.append([day.date(), sym, 'BUY', orders.ix[day, sym]])
            elif orders.ix[day, sym] < 0:
                order_list.append([day.date(), sym, 'SELL', -orders.ix[day, sym]])
    order_list = pd.DataFrame(order_list, columns=['Date', 'Symbol', 'Order', 'Shares'])
    order_list.set_index(order_list['Date'], inplace=True)
    order_list.drop('Date', 1, inplace=True)
    print(order_list)
    # Plot portfolio returns
    trade1 = tos.Benchmark(symbol="JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval1 = ms.compute_portvals(trade1, start_val=100000, commission=0.0, impact=0.0)
    trade2 = order_list
    portval2 = ms.compute_portvals(trade2, start_val=100000, commission=0.0, impact=0.0)
    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('MACD strategy vs Benchmark', fontsize=12)
    plt.plot(portval1 / start_val, label='Benchmark')
    plt.plot(portval2 / start_val, label='Strategy using MACD')
    plt.legend()
    # plt.show()
    plt.close()






if __name__ == '__main__':
    indicator()