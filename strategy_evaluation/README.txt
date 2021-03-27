# Project 8 Strategy Evaluation

This project includes 9 Python files:
 ManualStrategy.py  # Manually set a trading strategy based on the stock indicators we have, then output an orderbook on the srategy.
 StrategyLearner.py  # Use BagLearner to ensemble a bunch of random tree classifier, learn the trading strategy and generate an order book.
 BagLearner.py  # This bag learner ensembles a bunch of RTLearners and return the mode of the classifier results.
 RTLearner.py  # This is a random tree classifier.
 indicators.py  # Take in the trading symbol and period, create a bunch of time series stock indicators in the period.
 marketsimcode.py  # Take in an one column dataframe orderbook, output the portfolio value for each day in the trading period.
 experiment1.py  # Call the ManualStrategy and the StrategyLearner, and create a chart and a table to compare them on in-sample trading behavior.
 experiment2.py  # Call StrategyLearner to show the value of impact affects the in-sample trading behavior.
 testproject.py  # Run all tasks and generate all charts and statistics for the report in a single Python call.


In order to run the porject, call:
 PYTHONPATH=../:. python testproject.py
with the correspoding data files.This will generate all charts and statistics for the project.
In testproject, we execute the ManualStrategy, experiment1, and experiment2.
ManualStrategy will generate 3 charts (Figure 1, Figure 2, and Figure 3) and table 1 to compare the performance of Manual Strategy of in-sample and out-of-sample period.
The experiment1 will call ManualStrategy and StrategyLearner to generate the orderbooks of Manual Rule-Based trade, Strategy Learner trade, and Benchmark trade. The experiment1 will then call the marketsimcode to evaluate the portfolio value and generate a chart (Figure 4) and table 2 to compare.
The experiment2 will call StrategyLearner many times with changing the impact factor and return the orderbook. Then call the marketsimcode and generate chart (Figure 5) and table 3 to compare the portfolio value.