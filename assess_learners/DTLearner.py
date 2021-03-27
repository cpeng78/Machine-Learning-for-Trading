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


class DTLearner(object):
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		     		  		  		    	 		 		   		 		  

    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """
    def __init__(self, leaf_size=1, verbose=False):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """
        #pass  # move along, these aren't the drones you're looking for
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

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
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if data_y.shape[0] <= self.leaf_size or len(set(data_y)) == 1:
            return np.array([[-1, data_y.mean(), -1, -1]])
        else:
            i = np.nanargmax(abs(np.corrcoef(data_x, data_y, rowvar=False)[-1, :-1]))  # determine best feature i to split on: Highest absolute value of correlation.
            SplitVal = np.median(data_x[:, i])
            if data_x.shape[0] == data_x[data_x[:, i] <= SplitVal].shape[0]:
                SplitVal = data_x[:, i].mean()

            lefttree = self.build_tree(data_x[data_x[:, i] <= SplitVal], data_y[data_x[:, i] <= SplitVal])
            righttree = self.build_tree(data_x[data_x[:, i] > SplitVal], data_y[data_x[:, i] > SplitVal])
            root = [i, SplitVal, 1, lefttree.shape[0] + 1]
            return np.concatenate((np.array([root]), lefttree, righttree), axis=0)

    def query(self, data_x):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		     		  		  		    	 		 		   		 		  

        :param points: A numpy array with each row corresponding to a specific query.  		  	   		     		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """

        data_y = []
        if data_x.ndim == 1:
            data_y.append(self.query_tree(data_x))
        else:
            for x in data_x:
                data_y.append(self.query_tree(x))
        return np.array(data_y)

    def query_tree(self, data_x):
        node = 0
        while self.tree[node, 0] != -1:
            if data_x[int(self.tree[node, 0])] <= self.tree[node, 1]:
                node = int(node + self.tree[node, 2])
            else:
                node = int(node + self.tree[node, 3])
        return self.tree[node, 1]


if __name__ == "__main__":
    print("Decision tree learner")
