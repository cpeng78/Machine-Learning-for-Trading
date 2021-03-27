"""
Student Name: Chen Peng
GT User ID: cpeng78
GT ID: 903646937
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import marketsimcode as ms
import util as ut

def author():
    return "cpeng78"

class indicators(object):
    def __init__(self, symbol="JPM", sd='2008-01-01', ed='2009-12-31'):
        self.dates = pd.date_range(sd, ed)
        self.syms = [symbol]
        self.prices_all = ut.get_data(self.syms, self.dates)  # automatically adds SPY
        self.prices = self.prices_all[self.syms]  # only portfolio symbols

    ## Bollinger Bands value BOLL(20, 2)
    def BOLL(self, N=20, K=2):
        SMA = self.prices.rolling(N).mean()
        stdev = self.prices.rolling(N).std()
        # bb_upper = SMA + 2 * stdev
        # bb_lower = SMA - 2 * stdev
        bb_value = (self.prices - SMA) / (K * stdev)
        bb_value.columns = ['BOLL']
        return bb_value

    ## price/SMA BIAS(12)
    def BIAS(self, N=12):
        SMA = self.prices.rolling(N).mean()
        BIAS = (self.prices - SMA) / SMA * 100
        BIAS.columns = ['BIAS']
        return BIAS

    ## momentum index = (price(t)/price(t-N))-1  MTM(12)
    def MTM(self, N=12):
        MTM = self.prices / self.prices.shift(N) - 1
        MTM.columns = ['MTM']
        return MTM

    ## momentum moving average = (price(t)/price(t-N))-1  MTM(12)
    def MTMMA(self, N=12, M=6):
        MTM = self.prices / self.prices.shift(N) - 1
        MTMMA = MTM.rolling(M).mean()
        MTMMA.columns = ['MTMMA']
        return MTMMA

    ## Volatility
    def volatility(self, N=10):
        volatility = (self.prices / self.prices.shift(1) - 1).rolling(N).std()
        volatility.columns = ['volatility']
        return volatility

    ## Exponential Moving Average EMA(N)
    def EMA(self, N):
        EMA = self.prices.ewm(span=N, adjust=False).mean()
        EMA.columns = ['EMA']
        return EMA

    ## DIF(N, M)
    def DIF(self, N=12, M=26):
        EMA12 = self.EMA(N=N)
        EMA26 = self.EMA(N=M)
        DIF = EMA12 - EMA26
        DIF.columns = ['DIF']
        return DIF

    ## DEA(N, M, span)
    def DEA(self, N=12, M=26, span=9):
        DEA = self.DIF(N=N, M=M).ewm(span=span, adjust=False).mean()
        DEA.columns = ['DEA']
        return DEA

    ## Moving Average Convergence and Divergence
    def MACD(self, N=12, M=26, span=9):
        MACD = 2 * (self.DIF(N=N, M=M) - self.DEA(N=N, M=M, span=span))
        MACD.columns = ['MACD']
        return MACD

    def author(self):
        return "cpeng78"

if __name__ == '__main__':
    print('indicators')