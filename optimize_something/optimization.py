""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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

import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
import scipy.optimize as spo
import matplotlib.pyplot as plt  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  


# This is the function that will be tested by the autograder  		  	   		     		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		     		  		  		    	 		 		   		 		  
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		     		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		     		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		     		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		     		  		  		    	 		 		   		 		  
    statistics.  		  	   		     		  		  		    	 		 		   		 		  

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		     		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		     		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		     		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		     		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		     		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		     		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		     		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		     		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		     		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		     		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		     		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  

    # Read in adjusted closing prices for given symbols, date range  		  	   		     		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		     		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		     		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # find the allocations for the optimal portfolio  		  	   		     		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case

    normed_price = prices / prices.iloc[0, :]  # normalize price
    allocs = optimization(normed_price, compute_sharpe_ratio)  # find the optimized allocations

    # alloced_price = normed_price * allocs  # allocated price for each stock
    port_val = (normed_price * allocs).sum(1)  # compute daily portfolio values
    daily_returns = compute_daily_returns(port_val)  # optimized portfolio daily return
    avg_daily_returns = daily_returns.mean()  # mean of portfolio daily return
    std_daily_returns = daily_returns.std()  # std of portfolio daily return

    cr, adr, sddr, sr = [
        sum(normed_price.iloc[-1] * allocs),
        avg_daily_returns,
        std_daily_returns,
        -compute_sharpe_ratio(normed_price, allocs),
    ]  # add code here to compute stats  		  	   		     		  		  		    	 		 		   		 		  

    # Get daily portfolio value  		  	   		     		  		  		    	 		 		   		 		  
    # port_val = prices_SPY  # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		     		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		     		  		  		    	 		 		   		 		  
        # add code to plot here  		  	   		     		  		  		    	 		 		   		 		  
        df_temp = pd.concat(  		  	   		     		  		  		    	 		 		   		 		  
            [port_val, prices_SPY / prices_SPY[0]], keys=["Portfolio", "SPY"], axis=1
        )  		  	   		     		  		  		    	 		 		   		 		  
        df_temp.plot()
        plt.title('Daily Portfolio Value and SPY')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(linestyle=':')  # adds a grid
        plt.savefig('Fig1.png')

    return allocs, cr, adr, sddr, sr  		  	   		     		  		  		    	 		 		   		 		  


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


def optimization(normed_price, sharpe_ratio):
    # Generate initial portfolio allocations
    alloc_guess = tuple([1/normed_price.shape[1]]*normed_price.shape[1])
    bounds = spo.Bounds([0]*normed_price.shape[1], [1]*normed_price.shape[1])
    cons = ({'type': 'eq', 'fun': lambda inputs: 1 - np.sum(inputs)})

    result = spo.minimize(sharpe_ratio, alloc_guess, args=(normed_price,), method='SLSQP',
                          bounds=bounds, constraints=cons, options={'disp':True})

    return result.x


def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", 'X', "GLD", "JPM"]

    # Assess the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		     		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False
    )  		  	   		     		  		  		    	 		 		   		 		  

    # Print statistics  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		     		  		  		    	 		 		   		 		  


if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		     		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
