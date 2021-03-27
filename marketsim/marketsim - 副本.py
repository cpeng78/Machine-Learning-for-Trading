""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
import os  		  	   		     		  		  		    	 		 		   		 		  

import numpy as np

import pandas as pd
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  


def author():
        return 'cpeng78' # replace tb34 with your Georgia Tech username.


def compute_portvals(
    orders_file="./additional_orders/orders-short.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		     		  		  		    	 		 		   		 		  

    :param orders_file: Path of the order file or the file object  		  	   		     		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		     		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		     		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		     		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		     		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		     		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		     		  		  		    	 		 		   		 		  
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    # return rv

    # 1. Read in the price table
    record = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan']).sort_index()
    ### record = pd.read_csv(orders_file).sort_values('Date').reset_index(drop=True)
    dates = pd.date_range(record.index.min(), record.index.max())
    # Make the price table
    prices = get_data(set(record.Symbol), dates, addSPY=True).assign(CASH=1.)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    # print(prices)

    # 2. Read in trades history
    trades = pd.DataFrame(np.zeros(prices.shape), index=prices.index, columns=prices.columns)
    trades.loc[dates[0], ['CASH']] = start_val
    #print(record)
    for index, row in record.iterrows():
        if row['Order'] == 'BUY':
            trades.loc[index, row['Symbol']] = trades.loc[index, row['Symbol']] + row['Shares']
            trades.loc[index, 'CASH'] = trades.loc[index, 'CASH'] - row['Shares'] * prices.loc[index, row['Symbol']] * (1 + impact) - commission
        elif row['Order'] == 'SELL':
            trades.loc[index, row['Symbol']] = trades.loc[index, row['Symbol']] - row['Shares']
            trades.loc[index, 'CASH'] = trades.loc[index, 'CASH'] + row['Shares']*prices.loc[index, row['Symbol']] * (1 - impact) - commission
    # print(trades)

    # 3. Compute holdings
    holdings = trades.cumsum(axis='index')
    # print(holdings)

    # 4. Obtain value of each stock
    value = holdings.multiply(prices)
    # print(value)

    # 5. Get the portfolio value
    portvals = value.sum(axis='columns')
    # print(portvals.to_frame(name='portvals'))

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
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		     		  		  		    	 		 		   		 		  
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
