""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Chen Peng (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: cpeng78 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 903646937 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""

import datetime as dt
import random

import pandas as pd
import numpy as np
import util as ut
import indicators as ind
import BagLearner as bl
import RTLearner as rt
import matplotlib.pyplot as plt
import marketsimcode as mc


class StrategyLearner(object):
    """  		  	   		     		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    """
    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.5):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        leaf_size = 10
        self.N_days = 3
        bags = 20
        self.return_rate = 0.01
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size': leaf_size}, bags=bags, boost=False,
                            verbose=False)  # constructor

    def author(self):
        return "cpeng78"

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="AAPL",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,
    ):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		     		  		  		    	 		 		   		 		  

        :param symbol: The stock symbol to train on
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        """

        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(prices)

        # example use with new colname
        volume_all = ut.get_data(
            syms, dates, colname="Volume"
        )  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(volume)

        # Training X: indicators
        indicators = ind.indicators(symbol, sd, ed)
        boll = indicators.BOLL(N=20, K=2)
        bias = indicators.BIAS(N=12)
        momentum = indicators.MTM(N=12)
        dif = indicators.DIF(N=12, M=26)
        dea = indicators.DEA(N=12, M=26, span=9)

        # Training Y: N day return (N = 5)
        N_day_return = indicators.MTM(N=self.N_days).shift(-self.N_days)
        signal = N_day_return.rename(columns={'MTM': 'signal'})
        signal[signal['signal'] >= self.return_rate] = 1000
        signal[signal['signal'] <= -self.return_rate] = -1000
        signal[(signal['signal'] > -self.return_rate) & (signal['signal'] < self.return_rate)] = 0

        data = pd.concat([boll, bias, momentum, dif, dea, N_day_return, signal], axis=1)
        data = data.dropna(axis=0, how='any')  # drop all rows that have any NaN values
        #print(data)
        data = data.to_numpy()
        #print(data)

        # trades[symbol] = pred_returns
        # trades[trades[symbol] >= self.return_rate] = 1000
        # trades[trades[symbol] <= -self.return_rate] = -1000
        # trades[(trades[symbol] > -self.return_rate) & (trades[symbol] < self.return_rate)] = 0

        train_x = data[:, 0:-2]
        train_y = data[:, -1]

        self.learner.add_evidence(train_x, train_y)  # training step

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="AAPL",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000,
    ):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		     		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		     		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		     		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		     		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		     		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
        """

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol,]]  # only portfolio symbols
        prices.columns = ['prices']
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(prices)

        # example use with new colname
        volume_all = ut.get_data(
            [symbol], dates, colname="Volume"
        )  # automatically adds SPY
        volume = volume_all[[symbol]]  # only portfolio symbols
        volume.columns = ['volume']
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(volume)

        # Testing X: indicators
        indicators = ind.indicators(symbol, sd, ed)
        boll = indicators.BOLL(N=20, K=2)
        bias = indicators.BIAS(N=12)
        momentum = indicators.MTM(N=12)
        dif = indicators.DIF(N=12, M=26)
        dea = indicators.DEA(N=12, M=26, span=9)

        # Training Y: N day return (N = 5)
        N_day_return = indicators.MTM(N=self.N_days).shift(-self.N_days)
        N_day_return.columns = ['N_day_return']

        data = pd.concat([boll, bias, momentum, dif, dea, N_day_return], axis=1)
        data = data.dropna(axis=0, how='any')  # drop all rows that have any NaN values
        test_x = data.to_numpy()

        test_x = test_x[:, 0:-1]
        pred_y = self.learner.query(test_x)
        data['position'] = np.array(pred_y)

        trades = prices_all[[symbol, ]].copy()
        trades[symbol] = data.iloc[:, -1]
        trades.fillna(0, inplace=True)
        trades = trades.diff()
        trades.fillna(0, inplace=True)

        #if self.verbose:
        #    print(type(trades))  # it better be a DataFrame!
        #if self.verbose:
        #    print(trades)
        #if self.verbose:
        #    print(prices_all)
        return trades


def Benchmark(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    Benchmark_trade = prices.copy()
    Benchmark_trade.ix[:, :] = 0
    Benchmark_trade.ix[0, :] = 1000
    Benchmark_trade.ix[-1, :] = -1000
    return Benchmark_trade

def author():
    return "cpeng78"

def execute():
    print("One does not simply think up a strategy")
    np.random.seed(903646937)
    start_val = 100000
    learner = StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # training phase
    df_trades = learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                   sv=100000)  # testing phase

    portval = mc.compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)

    benchmark = Benchmark(symbol="JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval2 = mc.compute_portvals(benchmark, start_val=100000, commission=9.95, impact=0.005)

    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.title('MACD strategy vs Benchmark', fontsize=12)
    plt.plot(portval2 / start_val, label='Benchmark')
    plt.plot(portval / start_val, label='MA')
    plt.legend()
    # plt.show()
    plt.close()

if __name__ == "__main__":
    execute()
