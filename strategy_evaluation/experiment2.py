import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import StrategyLearner as sl
import marketsimcode as mc
import ManualStrategy as manual

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
    impact_list = [0, 0.0003, 0.001, 0.005, 0.02]  # 0.005
    commission = 0  # 9.95

    portval_list = []
    for i, impact in enumerate(impact_list):
        np.random.seed(903646937)
        # Strategy Learner
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
        learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=start_val)  # training phase
        learner_trades = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=start_val)  # strategy learner trade in-sample
        portval_list.append(mc.compute_portvals(learner_trades, start_val=start_val, commission=commission, impact=impact))  # portfolio value

    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('Strategy Learner with different impact factors', fontsize=12)
    for i, portval in enumerate(portval_list):
        plt.plot(portval / start_val, label='impact = {}'.format(impact_list[i]))
    plt.xlabel('Date')
    plt.ylabel('Normalized portfolio value')
    plt.legend(loc=2)
    plt.xticks(rotation=12)
    plt.savefig("Fig5_experiment2.png")
    plt.close()

    # performance for different impact
    rfr = 0.0  # risk free rate
    sf = 252.0  # scaling factor
    performance = []
    for i, portval in enumerate(portval_list):
        daily_returns = portval / portval.shift(1) - 1
        daily_returns.iloc[0, 0] = portval.iloc[0, 0] / start_val - 1  # First day return
        daily_returns = daily_returns.iloc[:, 0]  # daily returns dataframe to series
        cum_ret = portval.iloc[-1, 0] / portval.iloc[0, 0] - 1  # cumulative return
        avg_daily_ret = daily_returns.mean()  # mean of daily returns
        std_daily_ret = daily_returns.std()  # STDEV of daily returns
        sharpe_ratio = np.sqrt(sf) * (avg_daily_ret - rfr) / std_daily_ret  # Sharpe ratio
        performance.append([impact_list[i], cum_ret, std_daily_ret, avg_daily_ret, sharpe_ratio])  # performance list
    performances = pd.DataFrame(performance, columns=['impact', 'cum_ret', 'std_daily_ret', 'avg_daily_ret', 'Sharpe_ratio'])
    performances.to_csv(r'.\Table 3 Impact comparison.csv', index=False)


if __name__ == "__main__":
    execute()