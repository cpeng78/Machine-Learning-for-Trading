# ML4T_2020Fall

This repository includes the projects of CS7646: Machine Learning for Trading course at GeorgiaTech.

<div align=center><img src="strategy_evaluation/Fig4_experiment1.png"/><div align=left>

The final project "strategy_evaluation" implemented a manual trading strategy and a machine learning based strategy learner. The manual strategy manually sets up trading rules based on 4 of the stock indicators implemented in the project "indicator_evaluation". The strategy learner automatically choose the trading actions by training a random forest learner and build the trading rules.

The trading strategies is backtested and compared with the market simulater implemented in project "marketsim". 


## Projects: 
- **Project 1**: [Martingale](martingale)This project evaluates the actual betting strategy that Professor Balch uses at roulette when he goes to Las Vegas. 

- **Project 2**: [Optimize Something](optimize_something)This project uses optimizers to optimize a portfolio. You will find how much of a portfolio’s funds should be allocated to each stock to optimize it’s performance. We can optimize for many different metrics. In this version we will maximize Sharpe Ratio.

- **Project 3**: [Assess Learners](assess_learners)This project implements and evaluates four learning algorithms as Python classes: a “classic” Decision Tree learner, a Random Tree learner, a Bootstrap Aggregating learner, and an Insane Learner. Note that a Linear Regression learner is provided for you in the assess learners zip file, name LinRegLearner. The new classes is named DTLearner, RTLearner,BagLearner and InsaneLearner respectively. Considering this as a regression problem. The goal for your learner is to return a continuous numerical result. In this project we are ignoring the time order aspect of the data and treating it as if it is static data and time does not matter. In a later project we will make the transition to considering time series data.

- **Project 4**: [Defeat Learners](defeat_learners)This project generates data that works better for one learner than another. It's a test of the understanding of the strengths and weaknesses of various learners. The two learners of aim your datasets at are: a decision tree learner with leaf_size = 1 and the LinRegLearner provided as part of the repo.

- **Project 5**: [Marketsim](marketsim)

- **Project 6**: [Indicator Evaluation](indicator_evaluation)

- **Project 7**: [Qlearning Robot](qlearning_robot)

- **Project 8**: [Strategy Evaluation](strategy_evaluation)

## Dependencies for Running Locally:
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
