""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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

import random as rand

import numpy as np

def author():
    return 'cpeng78'  # replace tb34 with your Georgia Tech username.

class QLearner(object):
    """
    This is a Q learner object.  		  	   		     		  		  		    	 		 		   		 		  

    :param num_states: The number of states to consider.
    :type num_states: int  		  	   		     		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		     		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		     		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		     		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		     		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		     		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		     		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		     		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """
    def author(self):
        return 'cpeng78'

    def __init__(
        self,  		  	   		     		  		  		    	 		 		   		 		  
        num_states=100,  # the number of states to consider
        num_actions=4,  # the number of actions available
        alpha=0.2,  # learning rate
        gamma=0.9,  # the discount rate
        rar=0.5,  # random action rate
        radr=0.99,  # random action decay rated
        dyna=0,  # number of dyna updates for each regular update
        verbose=False,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.Q = np.zeros((num_states, num_actions), dtype=float)  # Q table
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0  # last state
        self.a = 0  # last action

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.r = np.zeros((num_states, num_actions), dtype=float)  # reward table for dyna
        self.t_c = np.full((num_states, num_actions, num_states), 0.00001, dtype=float)  # trasition count table


    def querysetstate(self, s):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		     		  		  		    	 		 		   		 		  

        :param s: The new state  		  	   		     		  		  		    	 		 		   		 		  
        :type s: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.s = s
        action = np.argmax(self.Q[s, :])
        self.a = action
        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(f"s = {s}, a = {action}")  		  	   		     		  		  		    	 		 		   		 		  
        return action  		  	   		     		  		  		    	 		 		   		 		  

    def query(self, s_prime, r):
        """  		  	   		     		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		     		  		  		    	 		 		   		 		  

        :param s_prime: The new state  		  	   		     		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		     		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		     		  		  		    	 		 		   		 		  
        :type r: float  		  	   		     		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
        """

        # Update Q table
        self.Q[self.s, self.a] = \
            (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, :].max())

        # Decide new action
        if np.random.rand() < self.rar:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q[s_prime, :])

        # Dyna Q
        # 1.Learn model T and R
        self.t_c[self.s, self.a, s_prime] += 1
        self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + self.alpha * r
        #print(self.r)
        for i in range(self.dyna):
            # 2.Halucinate experience
            s = np.random.randint(self.num_states)
            a = np.random.randint(self.num_actions)
            # s_prob = self.t_c[s, a, :].cumsum() / self.t_c[s, a, :].sum()
            # s_new = np.where(s_prob > np.random.rand())[0][0]
            # s_prob = self.t_c[s, a, :] / self.t_c[s, a, :].sum()
            # s_new = np.random.choice(self.num_states, p=s_prob.ravel())
            s_new = self.t_c[s, a, :].argmax()

            r_dyna = self.r[s, a]
            # 3.Update Q
            self.Q[s, a] = \
                (1 - self.alpha) * self.Q[s, a] + self.alpha * (r_dyna + self.gamma * self.Q[s_new, :].max())
            #self.rar *= self.radr


        # Update random action rate, new state, and action
        self.rar *= self.radr
        self.s = s_prime
        self.a = action

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")  		  	   		     		  		  		    	 		 		   		 		  
        return action  		  	   		     		  		  		    	 		 		   		 		  

    def get_Q(self):
        return self.Q
    def print_Q(self):
        print(self.Q)

if __name__ == "__main__":
    learner = QLearner()