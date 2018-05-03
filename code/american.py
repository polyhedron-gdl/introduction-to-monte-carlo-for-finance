from pylab import *
from matplotlib import pyplot as pl

import numpy as np
import scipy.stats as ss

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

np.random.seed(150000)
# Model Parameters
S0 = 36.  # initial stock level
K = 40.  # strike price
T = 1.0  # time-to-maturity
r = 0.06  # short rate
sigma = 0.2  # volatility

# Simulation Parameters
I = 10000   # number of paths
M = 50      # number of points for each path
dt = T / M
df = math.exp(-r * dt)

# Stock Price Paths
S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
    + sigma * math.sqrt(dt) * np.random.standard_normal((M + 1, I)), axis=0))
S[0] = S0

# plotting some path
t = np.linspace(0, T, M+1)
paths = S[:,1:100]

# plotting expiry price distribution
expiry = S[-1,:]
hist = np.histogram(expiry, 100)
index = np.arange(100)

pl.figure(figsize=(15,5))
pl.subplot(121)
pl.plot(t, paths)

pl.subplot(122)
pl.bar(index, hist[0])

pl.show()

# the value n = 25 is simply to emulate the example with only two exercise times
n = 25
maturity  = S[50,:]
reference = S[50-n,:]
payoff = np.maximum(K-maturity,0)*math.exp(-r *n*dt)
pl.plot(reference, payoff,'.')

pl.show()

C = BlackScholes('P',reference, K, r, sigma, 0.5*T)

pl.plot(reference, payoff,'.')
pl.plot(reference, C, '.', color='r')

pl.show()

npol = 7
rg   = np.polyfit(reference, payoff, npol)
#y = [sum(rg[k]*x**(npol-k) for k in range(npol+1)) for x in reference]

y = np.polyval(rg, reference)
    
xx=np.array(reference)

pl.plot(reference, payoff,'.')
pl.plot(xx,y,'.',color='g')    

pl.show()

pass