#!/usr/bin/env python
# coding: utf-8

# # classes for VaR example
class VarPositionStock:
    def __init__(self, ticker, shares):
        self.ticker = ticker
        self.shares = shares
    def value(self, price):
        return self.shares * price
import scipy.stats as si
import numpy as np
class VarPositionEuropeanOption:
    daysInYear=252
    def __init__(self, ticker, share, contracts, callPutType, strike, expiryDays):
        self.ticker = ticker
        self.share = share
        self.contracts = contracts
        self.callPutType = callPutType # 'call' or 'put'
        self.strike = strike
        self.expiryDays = expiryDays # in practice could be a date type
    def value(self,S,r,sigma,t):  
        K=self.strike
        T=t/VarPositionEuropeanOption.daysInYear
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if self.callPutType=='call': 
            call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) 
            return (call*self.contracts*self.share)
        elif self.callPutType=='put':
            put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
            return (put*self.contracts*self.share)
        
