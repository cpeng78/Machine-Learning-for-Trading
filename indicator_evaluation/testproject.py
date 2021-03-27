"""
Student Name: Chen Peng
GT User ID: cpeng78
GT ID: 903646937
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TheoreticallyOptimalStrategy as tos
import marketsimcode as ms
import indicators as ind


def author():
    return 'cpeng78'

if __name__ == "__main__":
    ind.indicator(symbol = "JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)

    start_val = 100000
    trade1 = tos.Benchmark(symbol = "JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval1 = ms.compute_portvals(trade1, start_val=100000, commission=0.0, impact=0.0)
    print('trade\n', trade1)
    print('portval\n', portval1)

    trade2 = tos.testPolicy(symbol = "JPM", sd='2008-01-01', ed='2009-12-31', sv=100000)
    portval2 = ms.compute_portvals(trade2, start_val=100000, commission=0.0, impact=0.0)
    print('trade\n', trade2)
    print('portval\n', portval2)

    ax = plt.plot()
    pd.plotting.register_matplotlib_converters()
    plt.plot(portval1 / start_val, label='Benchmark', color='tab:green')
    plt.plot(portval2 / start_val, label='Theoretically optimal portfolio', color='tab:red')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.title('Theoretical Optimal Strategy vs Benchmark', fontsize=12)
    plt.savefig('Fig6.png')
    # plt.show()
    plt.close()