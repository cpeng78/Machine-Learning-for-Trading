# README

This submission includes 5 python code files:

1. testproject.py 
Include the main function which calls the indivator function in indicator.py and calls the Benchmark and testPolicy function in TheoreticallyOptimalStrategy.py to return a
dataframe of order book of the Benchmark strategy and Theoretically Optimal Strategy. Then draw a figure comparing the normalized portfolio returns of the two strategies.

2. indicator.py 
Include the indicator function which receive symbol, start_time, end_time, and start_value as input. The indicator function create 5 different stock indicators includes Bollinger
Bands value, SMA, momentum, volatility, MACD. Then draw figures to show the features of these indicators.

3. marketsimcode.py
Include compute_portvals function which receive a dataframe of order book, start_value, commission, and impact. Returns the portfolio value.

4. TheoreticallyOptimalStrategy.py
Includes Benchmark function that create a dataframe of order book based on Benchmark stategy.
Includes testPolicy function that create a dataframe of order book based on Theoretically Optimal stategy.

In order to run these code, call the main function in testproject.py. This will generate all figures for this project.