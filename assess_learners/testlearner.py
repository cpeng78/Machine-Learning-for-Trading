""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import math  		  	   		     		  		  		    	 		 		   		 		  
import sys  		  	   		     		  		  		    	 		 		   		 		  
import time

import numpy as np
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "cpeng78"  # replace tb34 with your Georgia Tech username

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    # inf = open('Data/Istanbul.csv')
    # print(inf.readlines()[1].split(','))
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )

    np.random.seed(903646937)
    np.random.shuffle(data)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		     		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  # train it  		  	   		     		  		  		    	 		 		   		 		  
    # print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions  		  	   		     		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)  		  	   		     		  		  		    	 		 		   		 		  
    # print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions  		  	   		     		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		     		  		  		    	 		 		   		 		  
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)  		  	   		     		  		  		    	 		 		   		 		  
    # print(f"corr: {c[0,1]}")

    # Experiment 1: Decision tree leaner overfitting effect
    leaf_sizes = []
    rmse_dt_in = []
    rmse_dt_out = []
    for leaf_size in range(1, train_rows):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        pred_y = learner.query(train_x)  # query
        rmse_dt_in.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))
        pred_y = learner.query(test_x)  # get the predictions
        rmse_dt_out.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))
        leaf_sizes.append(leaf_size)
    # print(rmse_dt_in)
    # print(rmse_dt_out)
    rmse_dt_in = np.array([leaf_sizes, rmse_dt_in]).T
    rmse_dt_out = np.array([leaf_sizes, rmse_dt_out]).T
    ax = plt.plot()
    plt.plot(rmse_dt_in[:200, 0], rmse_dt_in[:200, 1], label='RMSE in sample')
    plt.plot(rmse_dt_out[:200, 0], rmse_dt_out[:200, 1], label='RMSE out sample')
    plt.legend()
    plt.title('Overfitting Effect with Decision Tree Learner', fontsize=12)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.ylim(0, 0.01)
    plt.savefig('Fig1.png')
    # plt.show()
    plt.close()

    # Experiment 2: Bagger learner of decision tree overfitting effect
    leaf_sizes = []
    rmse_bag_in = []
    rmse_bag_out = []
    for leaf_size in range(1, train_rows):
        #learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=10, boost=False, verbose=False)
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size': leaf_size}, bags=20, boost=False, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        pred_y = learner.query(train_x)  # query
        rmse_bag_in.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))
        pred_y = learner.query(test_x)  # get the predictions
        rmse_bag_out.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))
        leaf_sizes.append(leaf_size)
        # print(leaf_size, rmse_in[-1], rmse_out[-1])
    # print(rmse_bag_in)
    # print(rmse_bag_out)
    rmse_bag_in = np.array([leaf_sizes, rmse_bag_in]).T
    rmse_bag_out = np.array([leaf_sizes, rmse_bag_out]).T
    ax = plt.plot()
    plt.plot(rmse_bag_in[:200, 0], rmse_bag_in[:200, 1], label='RMSE Bag Learner in sample')
    plt.plot(rmse_bag_out[:200, 0], rmse_bag_out[:200, 1], label='RMSE Bag Learner out sample')
    plt.plot(rmse_dt_in[:200, 0], rmse_dt_in[:200, 1], '--', label='RMSE DT Learner in sample', color='tab:blue')
    plt.plot(rmse_dt_out[:200, 0], rmse_dt_out[:200, 1], '--', label='RMSE DT Learner out sample', color='tab:orange')
    plt.legend()
    plt.title('Overfitting Effect with Bagging of Decision Tree Learner', fontsize=12)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.ylim(0, 0.01)
    plt.savefig('Fig2.png')
    # plt.show()
    plt.close()

    # Experiment 3: Classic decision tree learner vs random tree learner
    leaf_sizes = []
    train_time_dt = []
    train_time_rt = []
    query_time_dt = []
    query_time_rt = []
    mae_dt = []
    mae_rt = []
    mse_dt = []
    mse_rt = []
    rmse_dt = []
    rmse_rt = []
    r2_dt = []
    r2_rt = []
    for leaf_size in range(1, train_rows):
        leaf_sizes.append(leaf_size)
        start = time.clock()
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        end_train = time.clock()
        pred_y = learner.query(test_x)  # query
        end_query = time.clock()
        train_time_dt.append(end_train - start)
        query_time_dt.append(end_query - end_train)
        mae_dt.append(abs(test_y - pred_y).sum() / test_y.shape[0])
        rmse_dt.append(math.sqrt(((test_y - pred_y)**2).sum() / test_y.shape[0]))
        mse_dt.append(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        r2_dt.append(1 - mse_dt[-1] / np.var(test_y))

        start = time.clock()
        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # training step
        end_train = time.clock()
        pred_y = learner.query(test_x)  # query
        end_query = time.clock()
        train_time_rt.append(end_train - start)
        query_time_rt.append(end_query - end_train)
        mae_rt.append(abs(test_y - pred_y).sum() / test_y.shape[0])
        mse_rt.append(((test_y - pred_y)**2).sum() / test_y.shape[0])
        r2_rt.append(1 - mse_rt[-1] / np.var(test_y))
    # rmsle_rt.append(math.sqrt(((np.log(test_y + 1) - np.log(pred_y + 1)) ** 2).sum() / test_y.shape[0]))

    train_time_dt = np.array([leaf_sizes, train_time_dt]).T
    train_time_rt = np.array([leaf_sizes, train_time_rt]).T
    query_time_dt = np.array([leaf_sizes, query_time_dt]).T
    query_time_rt = np.array([leaf_sizes, query_time_rt]).T
    fig1, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8.5, 4))
    ax[0].plot(train_time_dt[:200, 0], train_time_dt[:200, 1], label='DT leaner')
    ax[0].plot(train_time_rt[:200, 0], train_time_rt[:200, 1], label='RT leaner')
    ax[0].legend()
    ax[0].set_title('Training time')
    ax[0].set_xlabel('Leaf Size')
    ax[0].set_ylabel('Time (s)')
    #plt.tight_layout()
    ax[1].plot(query_time_dt[:200, 0], query_time_dt[:200, 1], label='DT leaner')
    ax[1].plot(query_time_rt[:200, 0], query_time_rt[:200, 1], label='RT leaner')
    ax[1].legend()
    ax[1].set_title('Query time')
    ax[1].set_xlabel('Leaf Size')
    ax[1].set_ylabel('Time (s)')
    plt.savefig('Fig3.png')
    # plt.show()
    plt.close()

    mae_dt = np.array([leaf_sizes, mae_dt]).T
    mae_rt = np.array([leaf_sizes, mae_rt]).T
    mse_dt = np.array([leaf_sizes, mse_dt]).T
    mse_rt = np.array([leaf_sizes, mse_rt]).T
    r2_dt = np.array([leaf_sizes, r2_dt]).T
    r2_rt = np.array([leaf_sizes, r2_rt]).T
    fig2, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8.5, 4))
    ax[0].plot(mae_dt[:200, 0], mae_dt[:200, 1], label='MAE of DT')
    ax[0].plot(mae_rt[:200, 0], mae_rt[:200, 1], label='MAE of RT')
    ax[0].legend()
    ax[0].set_title('MAE of DT Learner and RT Learner')
    ax[0].set_xlabel('Leaf Size')
    ax[0].set_ylabel('MAE')
    ax[1].plot(r2_dt[:200, 0], r2_dt[:200, 1], label='$R^2$ of DT')
    ax[1].plot(r2_rt[:200, 0], r2_rt[:200, 1], label='$R^2$ of RT')
    ax[1].legend()
    ax[1].set_title('R Square of DT Learner and RT Learner')
    ax[1].set_xlabel('Leaf Size')
    ax[1].set_ylabel('R square')
    plt.savefig('Fig4.png')
    # plt.show()
    plt.close()