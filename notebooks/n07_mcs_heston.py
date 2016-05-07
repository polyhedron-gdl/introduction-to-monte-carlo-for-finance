
# coding: utf-8

# In[58]:

get_ipython().magic(u'matplotlib inline')

from pylab import *
from matplotlib import pyplot as pl
from BCC_option_valuation import H93_call_value

import numpy as np


# ## The Heston Model and Its Simulation
# 
# ##### from Yves Hilpisch "Derivatives Analytics with Python" Wiley Finance (2015) 
# (in my personal opinion, one of the most useful books in quantitative finance of recent years)
# You cannot miss to visit his [web site](http://pythonquants.com/)!!! 
# 
# <img src="http://hilpisch.com/images/derivatives_analytics_front.jpg" alt="Derivatives Analytics with Python" width="10%" align="left" border="0">
# 
# 
# <img src="img/tpq_logo.png" alt="" width="10%" align="left" border="0">

# ### General Description
# 
# In this section we are going to consider the stochastic volatility model MH93 with constant short rate. This section values European call and put options in this model by MCS. As for the ZCB values, we also have available a semi-analytical pricing formula which generates natural bench- mark values against which to compare the MCS estimates.
# For $0 \le t \le T$ , the risk-neutral dynamics of the index in the H93 stochastic volatility model are given by
# 
# $$
# dS_t = rS_t dt + \sqrt{v_t} S_t dZ_t^1
# $$
# 
# with the variance following the square-root diffusion
# 
# $$
# dv_t = \kappa_v (\theta_v - v_t) dt + \sigma_v \sqrt{v_t} dZ_t^2
# $$
# 
# The two Brownian motions are instantaneously correlated with $dZ_t^1 dZ_t^2 = \rho$ This correlation introduces a new problem dimension into the discretization for simulation purposes. To avoid problems arising from correlating normally distributed increments (of $S$) with chi-squared distributed increments (of $v$), we will in the following only consider Euler schemes for both the $S$ and $v$ process. This has the advantage that the increments of $v$ become normally distributed as well and can therefore be easily correlated with the increments of $S$.
# 
# ...
# 

# In[59]:

def random_number_generator(M, I):
    ''' Function to generate pseudo-random numbers.

    Parameters
    ==========
    M: int
        time steps
    I: int
        number of simulation paths

    Returns
    =======
    rand: NumPy array
        random number array
    '''
    if antipath:
        rand = np.random.standard_normal((2, M + 1, I / 2))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((2, M + 1, I))
    if momatch:
        rand = rand / np.std(rand)
        rand = rand - np.mean(rand)
    return rand


# ### Square-Root Diffusion Simulation Function

# In[60]:

def SRD_generate_paths(x_disc, x0, kappa, theta, sigma,
                       T, M, I, rand, row, cho_matrix):
    ''' Function to simulate Square-Root Diffussion (SRD/CIR) process.

    Parameters
    ==========
    x0: float
        initial value
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor
    T: float
        final date/time horizon
    M: int
        number of time steps
    I: int
        number of paths
    row: int
        row number for random numbers
    cho_matrix: NumPy array
        cholesky matrix

    Returns
    =======
    x: NumPy array
        simulated variance paths
    '''
    dt = T / M
    x = np.zeros((M + 1, I), dtype=np.float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0
    sdt = math.sqrt(dt)
    for t in xrange(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        if x_disc == 'Full Truncation':
            xh[t] = (xh[t - 1] + kappa * (theta -
                 np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        elif x_disc == 'Partial Truncation':
            xh[t] = (xh[t - 1] + kappa * (theta - xh[t - 1]) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        elif x_disc == 'Truncation':
            x[t] = np.maximum(0, x[t - 1]
                 + kappa * (theta - x[t - 1]) * dt +
                 np.sqrt(x[t - 1]) * sigma * ran[row] * sdt)
        elif x_disc == 'Reflection':
            xh[t] = (xh[t - 1]
                 + kappa * (theta - abs(xh[t - 1])) * dt +
                 np.sqrt(abs(xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = abs(xh[t])
        elif x_disc == 'Higham-Mao':
            xh[t] = (xh[t - 1] + kappa * (theta - xh[t - 1]) * dt +
                 np.sqrt(abs(xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = abs(xh[t])
        elif x_disc == 'Simple Reflection':
            x[t] = abs(x[t - 1] + kappa * (theta - x[t - 1]) * dt +
                 np.sqrt(x[t - 1]) * sigma * ran[row] * sdt)
        elif x_disc == 'Absorption':
            xh[t] = (np.maximum(0, xh[t - 1])
                 + kappa * (theta - np.maximum(0, xh[t - 1])) * dt +
                 np.sqrt(np.maximum(0, xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        else:
            print x_disc
            print "No valid Euler scheme."
            sys.exit(0)
    return x


# ### Function for Heston Asset Process

# In[61]:

def H93_generate_paths(S0, r, v, row, cho_matrix):
    ''' Simulation of Heston (1993) index process.
    
    Parameters
    ==========
    S0: float
        initial value
    r: float
        constant short rate
    v: NumPy array
        simulated variance paths
    row: int
        row/matrix of random number array to use
    cho_matrix: NumPy array
        Cholesky matrix

    Returns
    =======
    S: NumPy array
        simulated index level paths
    '''
    S = np.zeros((M + 1, I), dtype=np.float)
    S[0] = S0
    bias = 0.0
    sdt = math.sqrt(dt)
    for t in xrange(1, M + 1, 1):
        ran = np.dot(cho_matrix, rand[:, t])
        if momatch:
            bias = np.mean(np.sqrt(v[t]) * ran[row] * sdt)
        if s_disc == 'Log':
            S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
                    np.sqrt(v[t]) * ran[row] * sdt - bias)
        elif s_disc == 'Naive':
            S[t] = S[t - 1] * (math.exp(r * dt) +
                    np.sqrt(v[t]) * ran[row] * sdt - bias)
        else:
            print "No valid Euler scheme."
            exit(0)
    return S


# In[76]:

r       = 0.05   # Fixed Short Rate
theta_v = 0.02   # long-term variance level
S0      = 100.0  # initial index level
K       = 90

v0      = 0.01
kappa_v = 1.5
sigma_v = 0.15
rho     = 0.1

covariance_matrix    = np.zeros((2, 2), dtype=np.float)
covariance_matrix[0] = [1.0, rho]
covariance_matrix[1] = [rho, 1.0]
cho_matrix           = np.linalg.cholesky(covariance_matrix)

# time step (per year)
M0      = 50       
# expiry (y)
T       = 1.0/12.0     
# number of paths per valuation
I       = 100000     

antipath = False # antithetic paths for variance reduction
momatch  = True # random number correction (std + mean + drift)
x_disc   = 'Full Truncation'
s_disc   = 'Log' 

# memory clean-up
v, S, rand, h = 0.0, 0.0, 0.0, 0.0
M = int(M0 * T)  # number of total time steps 
dt = T / M  # time interval in years
# random numbers
rand = random_number_generator(M, I)
# volatility process paths
v = SRD_generate_paths(x_disc, v0, kappa_v, theta_v,sigma_v, T, M, I, rand, 1, cho_matrix)
# index level process paths
S = H93_generate_paths(S0, r, v, 0, cho_matrix)
# discount factor
B0T = math.exp(-r * T)      

# European call option value (semi-analytical)
C0  = H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0)

# inner value matrix call
h = np.maximum(S - K, 0)

pv     = B0T * h[-1]                # present value vector
V0_MCS = np.sum(pv) / I             # MCS estimator
SE     = np.std(pv) / math.sqrt(I)  # standard error

rel_error = (V0_MCS - C0) / C0
                        


# In[77]:

print V0_MCS, C0, SE, 100*rel_error

