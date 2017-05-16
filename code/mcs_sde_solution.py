'''
Created on 23 apr 2016

@author: Giovanni Della Lunga
@summary: Study of Eulero-Milshstein Approximation of a Geometric Brownian Motion, for educational purpose only
'''

import matplotlib

from pylab import *
from math import exp, sqrt
from matplotlib import pyplot as plt

import numpy

def euler_approx(nstep, M, S0, vol, rand, dt):
    '''
    @summary: computation of Euler approximation
    '''
    S            = zeros_like(rand)
    S[0, :]      = S0  # initial values
    S_prev       = S0
    jump         = M / nstep

    for k in range(1, nstep + 1):
        dwti = 0
        delta_t = (times[k*jump] - times[(k-1)*jump])*dt
        
        for t in times[(k-1)*jump+1: k*jump+1]:
            dwti = dwti + rand[t, :] * sqrt(dt)
        
        S[k*jump,0] = S_prev * (1 +  r * delta_t +  vol * dwti) 
        
        for l in range(1, jump):
            S[(k-1)*jump + l] = S_prev + l * (S[k*jump,0] - S_prev) / jump
        
        S_prev = S[t,0]    
    return S    

def milshstein_approx(nstep, M, S0, vol, rand, dt):
    '''
    @summary: computation of Milshstein approximation
    '''
    S = zeros_like(rand)
    S[0, :] = S0  # initial values
    S_prev      = S0
    jump         = M / nstep
    
    for k in range(1, nstep+1):
        dwti = 0
        delta_t = (times[k*jump] - times[(k-1)*jump])*dt
        
        for t in times[(k-1)*jump+1: k*jump+1]:
            dwti = dwti + rand[t, :] * sqrt(dt)
        
        S[k*jump,0] = S_prev * (1 +  r * delta_t +  vol * dwti)  + 0.5* S_prev * vol * vol * (dwti*dwti-delta_t)
        
        for l in range(1, jump):
            S[(k-1)*jump + l] = S_prev + l * (S[k*jump,0] - S_prev) / jump
        
        S_prev = S[t,0]    
    return S    

# Simulate a number of years of daily stock quotes
# Stock Parameters
S0  = 100.0 # initial index level
T   = 10.0  # time horizon (years)
r   = 0.00  # risk -less short rate
vol = 0.6   # instantaneous volatility

# Simulation Parameters
#seed(1000)
M  = 1024         # time steps
I  = 1            # index level paths
dt = T / M        # time interval
df = exp(-r * dt) # discount factor
nsample = 5

for n in range(1,nsample+1):
    # Stock Price Paths
    rand = standard_normal ((M + 1, I)) # random numbers
    S = zeros_like(rand)                # stock matrix
    S[0, :] = S0                        # initial values
    times = range(1, M + 1, 1)
    for t in times:        # stock price paths
        S[t, :] = S[t - 1, :] * exp((r - vol ** 2 / 2)
            * dt + vol * rand[t, :] * sqrt(dt))
    
    times = [0] + times
    time_labels = [float(t) * dt for t in times]
    
    #---------------------------------------------------------------------------------------------------------------------
    
    nstep = 2
    
    x  = time_labels
    y1 = S[:,0]
    y2 = euler_approx(nstep, M, S0, vol, rand, dt)
    y3 = milshstein_approx(nstep, M, S0, vol, rand, dt)
    
    plt.ion()
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    
    line1, = ax.plot(x, y1, 'r-', color='black') # Returns a tuple of line objects, thus the comma
    line2, = ax.plot(x, y2, 'r-', color='red') # Returns a tuple of line objects, thus the comma
    line3, = ax.plot(x, y3, 'r-', color='green') # Returns a tuple of line objects, thus the comma
    
    nsteps = [2**k for k in range(2,10)]
    for nstep in nsteps:
        plt.title('example: %s/%s - npoints = %s'%(str(n),str(nsample),str(nstep)))
        y2 = euler_approx(nstep, M, S0, vol, rand, dt)
        y3 = milshstein_approx(nstep, M, S0, vol, rand, dt)
        line2.set_ydata(y2)
        line3.set_ydata(y3)
        fig.canvas.draw()
        plt.pause(1)

while True:
    plt.pause(0.05)
    
    