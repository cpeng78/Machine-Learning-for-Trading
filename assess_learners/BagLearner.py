""""""
"""  		  	   		     		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  

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

import numpy as np


class BagLearner(object):
    """
    This is a Linear Regression Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs={}, bags=10, boost=False, verbose=False):
        """
        Constructor method
        """
        # pass  # move along, these aren't the drones you're looking for
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "cpeng78"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        for learner in self.learners:
            x_train, y_train = self.data_prep(data_x, data_y)
            learner.add_evidence(x_train, y_train)

    def data_prep(self, data_x, data_y):
        row_num = data_x.shape[0]
        randrow = np.random.randint(row_num)
        x_train = data_x[randrow]
        y_train = np.array([data_y[randrow]])
        for i in range(data_x.shape[0]-1):
            randrow = np.random.randint(row_num)
            x_train = np.vstack((x_train, data_x[randrow]))
            y_train = np.hstack((y_train, data_y[randrow]))
        return x_train, y_train


    def query(self, data_x):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        data_y = []
        for learner in self.learners:
            data_y.append(learner.query(data_x))
        return np.mean(data_y, axis=0)

        #data_y = []
        #for x in data_x:
        #    data_y.append(self.query_x(x.reshape(1, x.shape[0])))
        #return np.array(data_y)

    #def query_x(self, data_x):
    #    y_query = []
    #    for learner in self.learners:
    #        y_query.append(learner.query(data_x))
    #    return np.mean(y_query)


if __name__ == "__main__":
    print("Bag learner")
