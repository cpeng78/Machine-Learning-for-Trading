"""
Student Name: Chen Peng (replace with your name)
GT User ID: cpeng78 (replace with your User ID)
GT ID: 903646937 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util as ut
import indicators as ind
import marketsimcode as mc


def author():
    return "cpeng78"

def testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    syms = [symbol]
    dates = pd.date_range(sd, ed)

    # Get adjusted close price
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    # prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # Get daily volume
    volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
    volume = volume_all[syms]  # only portfolio symbols
    # volume_SPY = volume_all["SPY"]  # only SPY, for comparison later

    indicators = ind.indicators(symbol, sd, ed)
    boll = indicators.BOLL(N=20, K=2).rename(columns={'BOLL': symbol})
    bias = indicators.BIAS(N=12).rename(columns={'BIAS': symbol})
    mtm = indicators.MTM(N=12).rename(columns={'MTM': symbol})
    #mtmma = indicators.MTMMA(N=12, M=6).rename(columns={'MTMMA': symbol})
    dif = indicators.DIF(N=12, M=26).rename(columns={'DIF': symbol})
    dea = indicators.DEA(N=12, M=26, span=9).rename(columns={'DEA': symbol})

    # here we manually build a set of trades
    trades = prices_all[[symbol, ]].copy()
    trades.ix[:, :] = np.nan
    # set positions all to zero

    threshold = 1.0
    trades[(boll < threshold) & (boll.shift() > threshold)] = -1000
    trades[(boll > -threshold) & (boll.shift() < -threshold)] = 1000

    threshold = 5.0
    trades[(bias > threshold)] = -1000
    trades[(bias < -threshold)] = 1000

    threshold = 0.0
    trades[(mtm < -threshold) & (mtm.shift() > -threshold)] = 1000
    trades[(mtm > threshold) & (mtm.shift() < threshold)] = -1000

    # trades[(mtm < mtmma) & (mtm.shift() > mtmma)] = 1000
    # trades[(mtm > mtmma) & (mtm.shift() < mtmma)] = -1000

    trades[(dif > 0) & (dea > 0) & (dif > dif.shift()) & (dea > dea.shift())] = -1000
    trades[(dif < 0) & (dea < 0) & (dif < dif.shift()) & (dea < dea.shift())] = 1000
    trades[(dif > 0) & (dea > 0) & (dif < dif.shift()) & (dea < dea.shift())] = 1000
    trades[(dif < 0) & (dea < 0) & (dif > dif.shift()) & (dea > dea.shift())] = -1000

    trades.ix[-1, :] = 0

    trades.ffill(inplace=True)
    trades.fillna(0, inplace=True)

    trades = trades.diff()
    trades.fillna(0, inplace=True)

    # tmp_csum = 0.0
    # for date, trade in trades.iterrows():
    #     tmp_csum += trade.iloc[0]
    #     if ((trade.iloc[0] != 0) and (trade.abs().iloc[0] != 1000) and (trade.abs().iloc[0] != 2000)):
    #         incorrect = True
    #         print("illegal trade in first insample DF. abs(trade) not ")
    #         break

    return trades

# Benchmark Strateggy
def Benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    Benchmark_trade = prices.copy()
    Benchmark_trade.ix[:, :] = 0
    Benchmark_trade.ix[0, :] = 1000
    #Benchmark_trade.ix[-1, :] = -1000
    return Benchmark_trade

def execute():
    start_val = 100000
    symbol = 'JPM'
    commission = 9.95
    impact = 0.005
    rfr = 0.0  # risk free rate
    sf = 252.0  # scaling factor

    # In-sample period
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    # Benchmark trade in-sample
    benchmark = Benchmark(symbol=symbol, sd=sd, ed=ed, sv=start_val)
    portval_b_in = mc.compute_portvals(benchmark, start_val=start_val, commission=commission, impact=impact)

    # performance for benchmark in-sample
    daily_returns = portval_b_in / portval_b_in.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_b_in.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_b_in.iloc[-1, 0] / portval_b_in.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance_b_in = [cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio]  # performance list
    # print(performance_b_in)

    # Manual strategy trade in-sample
    df_trades = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_val)
    portval_m_in = mc.compute_portvals(df_trades, start_val=start_val, commission=commission, impact=impact)

    # performance for manual strategy trade in-sample
    daily_returns = portval_m_in / portval_m_in.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_m_in.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_m_in.iloc[-1, 0] / portval_m_in.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance_m_in = [cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio]  # performance list
    # print(performance_m_in)

    # position hold
    position = df_trades.cumsum(axis=0)

    # Figure 1 Benchmark vs manual strategy in-sample
    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    for item in df_trades[(df_trades[symbol] > 0) & (position[symbol] == 1000)].index:
        plt.axvline(item, ymin=0, ymax=1, color='blue', linewidth=0.5)
    for item in df_trades[(df_trades[symbol] < 0) & (position[symbol] == -1000)].index:
        plt.axvline(item, ymin=0, ymax=1, color='black', linewidth=0.5)
    plt.plot(portval_b_in / start_val, color='green', label='Benchmark')
    plt.plot(portval_m_in / start_val, color='red', label='Manual Strategy')
    plt.axvline(df_trades[(df_trades[symbol] > 0) & (position[symbol] == 1000)].index[0], ymin=0, ymax=1, color='blue', linewidth=0.5, label='LONG entry points')
    plt.axvline(df_trades[(df_trades[symbol] < 0) & (position[symbol] == -1000)].index[0], ymin=0, ymax=1, color='black', linewidth=0.5, label='SHORT entry points')
    plt.xlabel('Date')
    plt.ylabel('Normalized portfolio value')
    plt.title('Manual Strategy vs Benchmark (in-sample)')
    plt.legend(loc=2)
    plt.xticks(rotation=12)
    plt.savefig('Fig1_ManualStrategy_in.png')
    # plt.show()
    plt.close()

    # Out-of-sample period
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    # Benchmark trade out-of-sample
    benchmark = Benchmark(symbol=symbol, sd=sd, ed=ed, sv=start_val)
    portval_b_out = mc.compute_portvals(benchmark, start_val=start_val, commission=commission, impact=impact)

    # performance for benchmark out-of-sample
    daily_returns = portval_b_out / portval_b_out.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_b_out.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_b_out.iloc[-1, 0] / portval_b_out.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance_b_out = [cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio]  # performance list
    # print(performance_b_out)

    # Manual strategy trade out-of-sample
    df_trades = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_val)
    portval_m_out = mc.compute_portvals(df_trades, start_val=start_val, commission=commission, impact=impact)

    # performance for manual strategy trade out-of-sample
    daily_returns = portval_m_out / portval_m_out.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_m_out.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_m_out.iloc[-1, 0] / portval_m_out.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance_m_out = [cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio]  # performance list
    # print(performance_m_out)

    # position hold
    position = df_trades.cumsum(axis=0)

    # Figure 2 Compare the performance of Manual Strategy versus the benchmark for the in-sample and out-of-sample time periods
    fig1, ax = plt.subplots(1, 2, constrained_layout=False, figsize=(8.5, 4))
    ax[0].plot(portval_b_in / start_val, color='green', label='Benchmark')
    ax[0].plot(portval_m_in / start_val, color='red', label='Manual Strategy')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Normalized portfolio value')
    ax[0].set_title('In-sample comparison')
    ax[0].legend(loc=3)
    plt.setp(ax[0].get_xticklabels(), rotation=20)
    ax[1].plot(portval_b_out / start_val, color='green', label='Benchmark')
    ax[1].plot(portval_m_out / start_val, color='red', label='Manual Strategy')
    ax[1].set_xlabel('Date')
    ax[1].set_title('Out-of-sample comparison')
    ax[1].legend(loc=3)
    plt.setp(ax[1].get_xticklabels(), rotation=20)
    plt.savefig('Fig2_in_vs_out.png')
    # plt.show()
    plt.close()


    # Figure 3 Benchmark vs manual strategy out-of-sample
    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    for item in df_trades[(df_trades[symbol] > 0) & (position[symbol] == 1000)].index:
        plt.axvline(item, ymin=0, ymax=1, color='blue', linewidth=0.5)
    for item in df_trades[(df_trades[symbol] < 0) & (position[symbol] == -1000)].index:
        plt.axvline(item, ymin=0, ymax=1, color='black', linewidth=0.5)
    plt.plot(portval_b_out / start_val, color='green', label='Benchmark')
    plt.plot(portval_m_out / start_val, color='red', label='Manual Strategy')
    plt.axvline(df_trades[(df_trades[symbol] > 0) & (position[symbol] == 1000)].index[0], ymin=0, ymax=1, color='blue', linewidth=0.5, label='LONG entry points')
    plt.axvline(df_trades[(df_trades[symbol] < 0) & (position[symbol] == -1000)].index[0], ymin=0, ymax=1, color='black', linewidth=0.5, label='SHORT entry points')
    plt.xlabel('Date')
    plt.ylabel('Normalized portfolio value')
    plt.title('Manual Strategy vs Benchmark (out-of-sample)')
    plt.legend(loc=3)
    plt.xticks(rotation=12)
    plt.savefig('Fig3_ManualStrategy_out.png')
    # plt.show()
    plt.close()

    # Table 1 Performance comparison benchmark vs manual strategy, in-sample vs out-of-sample
    performances = {'Statistics': ['Cumulative return', 'STDEV of daily returns', 'Mean of daily returns', 'Sharpe ratio'],
                    'Benchmark (in-sample)': performance_b_in,
                    'Manual Strategy (in-sample)': performance_m_in,
                    'Benchmark (out-of-sample)': performance_b_out,
                    'Manual Strategy (out-of-sample)': performance_m_out,
                    }
    performances = pd.DataFrame(performances, columns=['Statistics',
                                                       'Benchmark (in-sample)',
                                                       'Manual Strategy (in-sample)',
                                                       'Benchmark (out-of-sample)',
                                                       'Manual Strategy (out-of-sample)'
                                                       ])
    # print(performances)
    performances.to_csv(r'.\Table 1 Performance comparison.csv', index=False)

if __name__ == "__main__":
    execute()
