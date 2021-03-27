import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import StrategyLearner as sl
import marketsimcode as mc
import ManualStrategy as ms

def author():
    return "cpeng78"

def execute():
    np.random.seed(903646937)
    symbol = 'JPM'  # Trading symbol
    sd_in = dt.datetime(2008, 1, 1)  # In-sample start date
    ed_in = dt.datetime(2009, 12, 31)  # In-sample end date
    sd_out = dt.datetime(2010, 1, 1)  # Out-of-sample start date
    ed_out = dt.datetime(2011, 12, 31)  # Out-of-sample end date
    start_val = 100000  # Start value
    impact = 0.005  # 0.005
    commission = 9.95  # 9.95

    # Strategy Learner
    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
    learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=start_val)  # training phase
    learner_trades = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=start_val)  # strategy learner trade in-sample
    portval_sl = mc.compute_portvals(learner_trades, start_val=start_val, commission=commission, impact=impact)  # portfolio value

    # Manual Strategy
    manual_trades = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=start_val)  # manual trade in-sample
    portval_m = mc.compute_portvals(manual_trades, start_val=start_val, commission=commission, impact=impact)  # portfolio value

    # Benchmark trader
    benchmark = sl.Benchmark(symbol=symbol, sd=sd_in, ed=ed_in, sv=start_val)  # Benchmark trade
    portval_b = mc.compute_portvals(benchmark, start_val=start_val, commission=commission, impact=impact)  # portfolio value

    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('Strategy Learner vs Manual Strategy vs Benchmark', fontsize=12)
    plt.plot(portval_sl / start_val, color='blue', label='Strategy Learner')
    plt.plot(portval_m / start_val, color='red', label='Manual Strategy')
    plt.plot(portval_b / start_val, color='green', label='Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Normalized portfolio value')
    plt.legend(loc=2)
    plt.xticks(rotation=12)
    plt.savefig("Fig4_experiment1.png")
    plt.close()

    # performance for different strategy
    rfr = 0.0  # risk free rate
    sf = 252.0  # scaling factor
    performance = []
    daily_returns = portval_sl / portval_sl.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_sl.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_sl.iloc[-1, 0] / portval_sl.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance.append(['Strategy Learner', cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio])  # performance list

    daily_returns = portval_m / portval_m.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_m.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_m.iloc[-1, 0] / portval_m.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance.append(['Manual Strategy', cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio])  # performance list

    daily_returns = portval_b / portval_b.shift(1) - 1
    daily_returns.iloc[0, 0] = portval_b.iloc[0, 0] / start_val - 1  # First day return
    daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
    cum_ret = portval_b.iloc[-1, 0] / portval_b.iloc[0, 0] - 1  # cumulative return
    avg_daily_ret = daily_returns.mean()  # mean of daily returns
    std_daily_ret = daily_returns.std()  # STDEV of daily returns
    sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
    performance.append(['Benchmark', cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio])  # performance list

    performances = pd.DataFrame(performance, columns=['Strategy', 'cum_ret', 'std_daily_ret', 'avg_daily_ret', 'Sharpe_ratio'])
    performances.to_csv(r'.\Table 2 Strategy comparison.csv', index=False)

if __name__ == "__main__":
    execute()
