import pandas as pd
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

if __name__ == '__main__':
    import os
    #os.chdir("../data/decision_tree_data")
    data = pd.read_csv('Data/Istanbul.csv', index_col='date')
    data = data.rename(columns={x:y for x,y in zip(data.columns, range(len(data.columns)))})
    data = np.array(data)
    print(data)
    print(data.shape)

    x_train = data[:300, :-1]
    y_train = data[:300, -1]
    x_test = data[300:500, :-1]
    y_test = data[300:500, -1]

    # Linear regression learner
    learner = lrl.LinRegLearner()
    learner.add_evidence(x_train, y_train)
    y_query = learner.query(x_test)
    print(learner)
    print(((y_test - y_query) ** 2).sum())

    # Decision tree learner
    learner = dt.DTLearner(leaf_size=1, verbose=False)  # .add_evidence(x_train, y_train)
    learner.add_evidence(x_train, y_train)
    y_query = learner.query(x_test)
    print(learner)
    print(((y_test - y_query)**2).sum())

    # Random tree learner
    learner = rt.RTLearner(leaf_size=1, verbose=False)
    learner.add_evidence(x_train, y_train)
    y_query = learner.query(x_test)
    print(learner)
    print(((y_test - y_query) ** 2).sum())

    # Insane learner
    learner = it.InsaneLearner()  # constructor
    learner.add_evidence(x_train, y_train)  # training step
    print(learner)
    print(learner.learner_list)
    y_query = learner.query(x_test)  # query
    print(((y_test - y_query) ** 2).sum())


    # Bag learner for linear regression
    learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
    learner.add_evidence(x_train, y_train)
    y_query = learner.query(x_test)
    print(learner)
    print(learner.learners)
    print(((y_test - y_query) ** 2).sum())

    # Bag learner for decision tree
    learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=20, boost=False, verbose=False)
    learner.add_evidence(x_train, y_train)
    y_query = learner.query(x_test)
    print(learner)
    print(learner.learners)
    print(((y_test - y_query) ** 2).sum())

    # Bag learner for random tree
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size":1}, bags=10, boost=False, verbose=False)
    learner.add_evidence(x_train, y_train)
    y_query = learner.query(x_test)
    print(learner)
    print(learner.learners)
    print(((y_test - y_query) ** 2).sum())



