# Machine Learning for Trading

This repository includes the projects of CS7646: Machine Learning for Trading course at GeorgiaTech.

<div align=center><img src="strategy_evaluation/Fig4_experiment1.png"/><div align=left>

The final project "strategy_evaluation" implemented a manual trading strategy and a machine learning based strategy learner. The manual strategy manually sets up trading rules based on 4 of the stock indicators implemented in the project "indicator_evaluation". The strategy learner automatically choose the trading actions by training a random forest learner and build the trading rules.

The trading strategies is backtested and compared with the market simulater implemented in project "marketsim". See [report](strategy_evaluation/report.pdf) for details.


## Projects: 
- **Project 1**: [Martingale](martingale). This project evaluates the actual betting strategy that Professor Balch uses at roulette when he goes to Las Vegas. See [report](martingale/report.pdf) for details.

- **Project 2**: [Optimize Something](optimize_something). This project uses optimizers to optimize a portfolio. You will find how much of a portfolio’s funds should be allocated to each stock to optimize it’s performance. We can optimize for many different metrics. In this version we will maximize Sharpe Ratio. See [report](optimize_something/report.pdf) for details.

- **Project 3**: [Assess Learners](assess_learners). This project implements and evaluates four learning algorithms as Python classes: a “classic” Decision Tree learner, a Random Tree learner, a Bootstrap Aggregating learner, and an Insane Learner. Note that a Linear Regression learner is provided for you in the assess learners zip file, name LinRegLearner. The new classes is named DTLearner, RTLearner,BagLearner and InsaneLearner respectively. Considering this as a regression problem. The goal for your learner is to return a continuous numerical result. In this project we are ignoring the time order aspect of the data and treating it as if it is static data and time does not matter. In a later project we will make the transition to considering time series data. See [report](assess_learners/report.pdf) for details.

- **Project 4**: [Defeat Learners](defeat_learners). This project generates data that works better for one learner than another. It's a test of the understanding of the strengths and weaknesses of various learners. The two learners of aim your datasets at are: a decision tree learner with leaf_size = 1 and the LinRegLearner provided as part of the repo.

- **Project 5**: [Marketsim](marketsim). This project creates a market simulator that accepts trading orders and keeps track of a portfolio’s value over time and then assesses the performance of that portfolio.

- **Project 6**: [Indicator Evaluation](indicator_evaluation). This project develops technical indicators and a Theoretically Optimal Strategy that will be the ground layer of the project 8. The technical indicators will be utilized in project 8 to devise an intuition-based trading strategy and a Machine Learning based trading strategy. Theoretically Optimal Strategy will give a baseline to gauge the later project’s performance against. See [report](indicator_evaluation/report.pdf) for details.

- **Project 7**: [Qlearning Robot](qlearning_robot). This project implements the Q-Learning and Dyna-Q solutions to the reinforcement learning problem. I apply them to a navigation problem in this project. In project 8 we can apply them to trading. The Q-Learning code doesn't care which problem it is solving. The difference is to wrap the learner in different code that frames the problem for the learner as necessary.

- **Project 8**: [Strategy Evaluation](strategy_evaluation). This project implements a Manual Strategy (manual rule based trader) by using the intuition and the indicators selected from project 6, and a strategy learner using the Random Forest learner. The project tests them against a stock using the market simulator. See [report](strategy_evaluation/report.pdf) for details.

## Dependencies for Running Locally:

The projects are in Python (version 3.6), and rely heavily on a few important libraries. These libraries are under active development, which unfortunately means there can be some compatibility issues between versions.

To create an environment for these projects:
```
conda env create --file environment.yml
```
Activate the new environment:
```
conda activate ml4t
```
The list of each library and its version number provided in the conda environment format:
```
name: ml4t
dependencies:
- python=3.6
- cycler=0.10.0
- kiwisolver=1.1.0
- matplotlib=3.0.3
- numpy=1.16.3
- pandas=0.24.2
- pyparsing=2.4.0
- python-dateutil=2.8.0
- pytz=2019.1
- scipy=1.2.1
- seaborn=0.9.0
- six=1.12.0
- joblib=0.13.2
- pytest=5.0
- future=0.17.1
- pip
- pip:
  - pprofile==2.0.2
  - jsons==0.8.8
```
To test the code, you’ll need to set up your PYTHONPATH to include the grading module and the utility module util.py, which are both one directory up from the project directories. Here’s an example of how to run the grading script for the optional (deprecated) assignment Assess Portfolio (note, grade_anlysis.py is included in the template zip file for Assess Portfolio):
```
PYTHONPATH=../:. python grade_analysis.py
```
which assumes you’re typing from the folder ML4T_2020Fall/assess_portfolio/. This will print out a lot of information, and will also produce two text files: points.txt and comments.txt. It will probably be helpful to scan through all of the output printed out in order to trace errors to your code, while comments.txt will contain a succinct summary of which test cases failed and the specific errors (without the backtrace).
