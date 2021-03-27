""""""
"""Assess a betting strategy.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		     		  		  		    	 		 		   		 		  
    """
    return "cpeng78"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
    """
    return 903646937  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		     		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		     		  		  		    	 		 		   		 		  

    :param win_prob: The probability of winning  		  	   		     		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		     		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		     		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def episode_result(win_prob=18./38.):
    winnings = [0, ]
    episode_winnings = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            if won:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
            winnings.append(episode_winnings)
    return winnings


def sim_results(win_prob, sim_num):
    df = pd.DataFrame(columns=np.arange(1000))
    for i in range(sim_num):
        df_temp = pd.DataFrame([episode_result(win_prob)])
        df = df.append(df_temp)
    return df


def real_episode(win_prob=18./38.):
    winnings = [0, ]
    episode_winnings = 0
    while episode_winnings < 80:
        won = False
        bet_amount = 1
        while not won:
            won = get_spin_result(win_prob)
            if won:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
                if bet_amount > episode_winnings + 256:
                    bet_amount = episode_winnings + 256
            winnings.append(episode_winnings)
            if bet_amount == 0:
                break
        if bet_amount == 0:
            break
    return winnings


def real_results(win_prob, sim_num):
    df = pd.DataFrame(columns=np.arange(1000))
    for i in range(sim_num):
        df_temp = pd.DataFrame([real_episode(win_prob)])
        df = df.append(df_temp)
    return df


def get_band(values, v_std):
    upper_band = values + v_std
    lower_band = values - v_std
    return upper_band, lower_band


def test_code():
    """  		  	   		     		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		     		  		  		    	 		 		   		 		  
    """
    win_prob = 18./38.  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    # 10 times simulation
    df1 = sim_results(win_prob, 10)
    df1.fillna(method='ffill', axis=1, inplace=True)

    # Figure 1
    for i in range(10):
        ax = df1.iloc[i].plot()
    plt.legend(['Sim 1', 'Sim 2', 'Sim 3', 'Sim 4', 'Sim 5', 'Sim 6', 'Sim 7', 'Sim 8', 'Sim 9', 'Sim 10'], loc=5)
    ax.set_title('Fig 1. 10 times simulation results', fontsize=15)
    ax.set_xlabel('bet times', fontsize=12)
    ax.set_ylabel('winnings', fontsize=12)
    plt.axis([0, 300, -256, 100])
    plt.savefig('fig1.png')

    # 1000 times simulation
    df2 = sim_results(win_prob, 1000)
    df2.fillna(method='ffill', axis=1, inplace=True)

    # Statistical results (mean, median, std)
    df_mean = df2.mean()
    df_median = df2.median()
    df_std = df2.std()
    prob = df2[999].value_counts(normalize=True)
    print('The winning probability of 1000 simulation is', prob[80] * 100, '% in Experiment 1.')
    print('The estimated expected value of our winnings after 1000 sequential bets is $', df2[999].sum()/1000, 'in Experiment 1.')

    # Figure 2
    upper_band_mean, lower_band_mean = get_band(df_mean, df_std)
    plt.close()
    plt.figure()
    df_mean.plot()
    upper_band_mean.plot()
    ax2 = lower_band_mean.plot()
    plt.legend(['mean', 'mean + std', 'mean - std'], loc=4)
    ax2.set_title('Fig 2. 1000 times simulation winning mean', fontsize=15)
    ax2.set_xlabel('bet times', fontsize=12)
    ax2.set_ylabel('winnings', fontsize=12)
    plt.axis([0, 300, -256, 100])
    plt.savefig('fig2.png')

    # Figure 3
    upper_band_mean, lower_band_mean = get_band(df_median, df_std)
    plt.close()
    plt.figure()
    df_median.plot()
    upper_band_mean.plot()
    ax2 = lower_band_mean.plot()
    plt.legend(['median', 'median + std', 'median - std'], loc=4)
    ax2.set_title('Fig 3. 1000 times simulation winning median', fontsize=15)
    ax2.set_xlabel('bet times', fontsize=12)
    ax2.set_ylabel('winnings', fontsize=12)
    plt.axis([0, 300, -256, 100])
    plt.savefig('fig3.png')

    # 1000 times realistic gambling simulation
    df3 = real_results(win_prob, 1000)
    df3.fillna(method='ffill', axis=1, inplace=True)

    # Statistical results (mean, median, std)
    real_mean = df3.mean()
    real_median = df3.median()
    real_std = df3.std()
    # num = 0
    prob = df3[999].value_counts(normalize=True)
    print('The winning probability of 1000 simulation is', prob[80]*100, '% in Experiment 2.')
    print('The estimated expected value of our winnings after 1000 sequential bets is $', sum(df3[999])/1000, 'in Experiment 2.')

    # Figure 4
    upper_band_mean, lower_band_mean = get_band(real_mean, real_std)
    plt.close()
    plt.figure()
    real_mean.plot()
    upper_band_mean.plot()
    ax2 = lower_band_mean.plot()
    plt.legend(['mean', 'mean + std', 'mean - std'], loc=5)
    ax2.set_title('Fig 4. 1000 times realistic gambling winning mean', fontsize=15)
    ax2.set_xlabel('bet times', fontsize=12)
    ax2.set_ylabel('winnings', fontsize=12)
    plt.axis([0, 300, -256, 100])
    plt.savefig('fig4.png')

    # Figure 5
    upper_band_mean, lower_band_mean = get_band(real_median, real_std)
    plt.close()
    plt.figure()
    real_median.plot()
    upper_band_mean.plot()
    ax2 = lower_band_mean.plot()
    plt.legend(['median', 'median + std', 'median - std'], loc=4)
    ax2.set_title('Fig 5. 1000 times realistic gambling winning median', fontsize=15)
    ax2.set_xlabel('bet times', fontsize=12)
    ax2.set_ylabel('winnings', fontsize=12)
    plt.axis([0, 300, -256, 100])
    plt.savefig('fig5.png')


if __name__ == "__main__":
    test_code()
