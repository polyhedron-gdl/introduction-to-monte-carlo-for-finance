'''
Created on 23 apr 2016

@author: Giovanni Della Lunga
@summary: Brownian Bridge Construction, for educational purpose only
'''

import numpy as np
from math  import sqrt
from pylab import seed, standard_normal
from matplotlib import pyplot as plt

#seed(1000)

# number of step of refinement
M = 12
# number of points
h = 2**M

T_0 = 0
T_N = 1

z     = standard_normal(h+1)
wt    = np.zeros(h+1)

t     = np.linspace(T_0, T_N, h+1)
wt[h] = sqrt(T_N) * z[h]

plt.ion()
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
line1, = ax.plot(t, wt, 'r-', color='black') # Returns a tuple of line objects, thus the comma
plt.ylim((-3,3))

j_max = 1
for k in range(1, M + 1):
    i_min = h / 2
    i = i_min
    l = 0
    r = h
    for j in range(1, j_max+1):
        i_old = i
        a = ((t[r] - t[i]) * wt[l] + (t[i] - t[l]) * wt[r]) / (t[r] - t[l])
        b = sqrt((t[i] - t[l]) * (t[r] - t[i]) / (t[r] - t[l]))
        wt[i] = a + b * z[i]
        
        for i2 in range(l + 1, i_old):
            wt[i2] = wt[l] + (float(i2 - l) / float(i - l)) * (wt[i] - wt[l])

        for i2 in range(i_old + 1, r):
            wt[i2] = wt[i] + (float(i2 - i) / float(r - i)) * (wt[r] - wt[i])
        
        i = i + h
        l = l + h
        r = r + h
            
    j_max = 2 * j_max
    h = i_min
    
    line1.set_ydata(wt)
    d = abs(np.max(wt)-np.min(wt))
    plt.ylim((np.min(wt)-.1*d, np.max(wt)+.1*d))
    plt.title('step = %s'%str(k))
    fig.canvas.draw()
    plt.pause(.25)

while True:
    plt.pause(0.01)
