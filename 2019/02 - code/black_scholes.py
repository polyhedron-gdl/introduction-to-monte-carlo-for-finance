'''
Created on 14 mag 2017

@author: User
'''

import numpy as np
import scipy.stats as ss
import time 

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/float(K)) + (r + sigma**2 / 2.0) * T)/ float(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
    return (np.log(S0/float(K)) + (r - sigma**2 / 2.0) * T) / float(sigma * np.sqrt(T))
 
def BlackScholes(payoff, S0, K, r, sigma, T):
    if payoff == 1:
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))
    

#_______________________________________________________________________________________________________________________________
#
if __name__ == "__main__":
    S0     = 10
    K      = 10
    r      = 0.05
    T      = 0.25
    sigma  = 0.20
    
    print BlackScholes(1,S0,K,r,sigma, T)