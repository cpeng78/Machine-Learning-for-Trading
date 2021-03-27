# ML4T_2020Fall

This repository includes the projects of CS7646: Machine Learning for Trading course at GeorgiaTech.

<div align=center><img src="strategy_evaluation/Fig4_experiment1.png"/><div align=left>

The final project "strategy_evaluation" implemented a manual trading strategy and a machine learning based strategy learner. The manual strategy manually sets up trading rules based on 4 of the stock indicators implemented in the project "indicator_evaluation". The strategy learner automatically choose the trading actions by training a random forest learner and build the trading rules.

The trading strategies is backtested and compared with the market simulater implemented in project "marketsim". 


## Projects: 
- **Project 1**: [Martingale](martingale)

- **Project 2**: [Optimize Something](optimize_something)

- **Project 3**: [Assess Learners](assess_learners)

- **Project 4**: [Defeat Learners](defeat_learners)

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
