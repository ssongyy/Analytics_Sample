#!/usr/bin/env python
# coding: utf-8

# Position classes for VaR example
import scipy.stats as si # for normal cdf (options calculation)
import numpy as np

# position in a stock of a given ticker and number of shares
class VarPositionStock_:
    # constructor for the static aspects of the position
    def __init__(self, ticker, shares):
        self.ticker = ticker
        self.shares = shares
    # value the stock position
    def value(self, price):
        return self.shares * price

# position in a European option
class VarPositionEuropeanOption_:
    daysInYear=252 # assumes 252 business days in a year; can change
    
    # constructor for the static aspects of the option
    def __init__(self, ticker, share, contracts, callPutType, strike, expiryDays):
        self.ticker = ticker
        self.share = share
        self.contracts = contracts
        self.callPutType = callPutType # 'call' or 'put'
        self.strike = strike
        self.expiryDays = expiryDays # in practice could be a date type

    # Black Scholes calculation for valuing a call option (internal function)
    @staticmethod
    def value_call(d1, d2, K, S, r, sigma, t):
        T=t/VarPositionEuropeanOption_.daysInYear
        value = (S * si.norm.cdf(d1, 0.0, 1.0) -
                K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) 
        return value
    
    # Black Scholes calculation for valuing a put option (internal function)
    @staticmethod
    def value_put(d1, d2, K, S, r, sigma, t):
        T=t/VarPositionEuropeanOption_.daysInYear
        value = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) -
                 S * si.norm.cdf(-d1, 0.0, 1.0))
        return value
    
    # value the option position
    def value(self,S,r,sigma,t):  
        K=self.strike
        T=t/VarPositionEuropeanOption_.daysInYear
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if self.callPutType=='call': 
            unitValue = VarPositionEuropeanOption_.value_call(d1, d2, K, S, r, sigma, t)
            return (unitValue*self.contracts*self.share)
        elif self.callPutType=='put':
            unitValue = VarPositionEuropeanOption_.value_put(d1, d2, K, S, r, sigma, t)
            return (unitValue*self.contracts*self.share)
        else: # option type not known
            raise ValueError("Unknown option type: " + self.callPutType)
'''
    # Black Scholes calculation for valuing a call option (internal function)
    @staticmethod
    def value_call(d1, d2, K, S, r, sigma, t):
        value = (S * si.norm.cdf(d1, 0.0, 1.0) -
                K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) 
        return value

    # Black Scholes calculation for valuing a put option (internal function)
    @staticmethod
    def value_put(d1, d2, K, S, r, sigma, t):
        value = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) -
                 S * si.norm.cdf(-d1, 0.0, 1.0))
        return value
'''
