
# coding: utf-8

# ### Introduction to Monte Carlo Simulation in Finance
# # Multiasset Simulation
# 
# ## Cholesky Decomposition
# 
# The **Choleski Decomposition** makes an appearance in Monte Carlo Methods where it is used to simulating systems with correlated variables.  Cholesky decomposition is applied to the correlation matrix, providing a lower triangular matrix $A$, which when applied to a vector of uncorrelated samples, $u$, produces the covariance vector of the system. Thus it is highly relevant for quantitative trading.
# 
# The standard procedure for generating a set of correlated normal random variables is through a linear combination of uncorrelated normal random variables;
# Assume we have a set of $n$ independent standard normal random variables $Z$ and we want to build a set of $n$ correlated standard normals $Z^\prime$ with correlation matrix $\Sigma$
# $$
# Z^\prime = AZ, \quad \quad AA^t = \Sigma
# $$
# 
# We can find a solution for $A$ in the form of a triangular matrix
# $$
# \begin{pmatrix} 
# A_{11} & 0 & \dots & 0  \\ 
# A_{21} & A_{22} & \dots & 0  \\ 
# \vdots & \vdots & \ddots & \dots  \\ 
# A_{n1} & A_{n2} & \dots & A_{nn}   
# \end{pmatrix}
# $$
# 
# **diagonal elements**
# $$
# a_{ii} = \sqrt{\Sigma_{ii} - \sum\limits_{k=1}^{i-1} a_{ik}^2}
# $$
# 
# **off-diagonal elements**
# $$
# a_{ij} = \frac{1}{a_{ii}} \left( \Sigma_{ij} - \sum\limits_{k=1}^{i-1} a_{ik} a_{jk} \right)
# $$
# 
# Using Python, the most efficient method in both development and execution time is to make use of the NumPy/SciPy linear algebra (linalg) library, which has a built in method cholesky to decompose a matrix. The optional lower parameter allows us to determine whether a lower or upper triangular matrix is produced: 

# In[1]:

import pprint
import scipy
import scipy.linalg   # SciPy Linear Algebra Library

A = scipy.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])
L = scipy.linalg.cholesky(A, lower=True)
U = scipy.linalg.cholesky(A, lower=False)

print "A:"
pprint.pprint(A)

print "L:"
pprint.pprint(L)

print "U:"
pprint.pprint(U)


# For example, for a two-dimension random vector we have simply
# $$
# A=
# \begin{pmatrix} 
# \sigma_1        & 0   \\ 
# \sigma_2 \rho & \sigma_2 \sqrt{1-\rho^2}   
# \end{pmatrix}
# $$
# 
# Say one needs to generate two correlated normal variables $x_1$ and $x_2$. All one needs to do is to generate two uncorrelated Gaussian random variables $z_1$ and$ z_2$ and set
# $$
# x_1 = z_1 
# $$
# 
# $$
# x_2 =  \rho z_1 + \sqrt{1-\rho^2} z_2
# $$
# 
# In Python everything you need is available in the *numpy* library, as we can see in the next example.

# In[2]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import scipy as sc

from math        import sqrt
from scipy.stats import norm as scnorm
from pylab       import *
from matplotlib  import pyplot as pl

xx = np.array([-0.51, 51.2])
yy = np.array([0.33, 51.6])
means = [xx.mean(), yy.mean()]  
stds = [xx.std() / 3, yy.std() / 3]
corr = 0.8         # correlation
covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
        [stds[0]*stds[1]*corr,           stds[1]**2]] 

m = np.random.multivariate_normal(means, covs, 1000).T
scatter(m[0], m[1])


# ## Brownian simulation of correlated assets
# 
# When using Monte Carlo methods to price options dependent on a basket of underlying assets (multidimensional stochastic simulations), the correlations between assets should be considered. Here I will show an example of how this can be simulated using pandas. 
# 
# Download and prepare the data
# 
# First we download some data from Yahoo:

# In[3]:

from pandas.io.data import DataReader
from pandas import Panel, DataFrame

symbols = ['AAPL',    # Apple Inc.
           'GLD',     # SPDR Gold Trust ETF
           'SNP',     # S&P 500 Index
           'MCD']     # McDonald's Corporation
data = dict((symbol, DataReader(symbol, "yahoo", pause=1)) for symbol in symbols)
panel = Panel(data).swapaxes('items', 'minor')
closing = panel['Close'].dropna()
closing.head()


# Now we can calculate the log returns:

# In[4]:

rets = log(closing / closing.shift(1)).dropna()
rets.head()


# The correlation matrix has information about the historical correlations between stocks in the group. We work under the assumption that this quantity is conserved, so the generated stocks will need to satisfy this condition:

# In[5]:

corr_matrix = rets.corr()
corr_matrix


# So the most correlated assets are MCD (McDonald's Corporation) and the SPX (S&P 500 Index). Pandas has a nice utility to plot the correlations:

# In[6]:

from pandas.tools.plotting import scatter_matrix

scatter_matrix(rets, figsize=(8,8));


# ### Simulation
# 
# The simulation procedure for generating random variables will go like this:
# 
# 1. Calculate the Cholesky Decomposition matrix, this step will return an upper triangular matrix  $L^T$.
# 2. Generate random vector  $X \sim N(0,1)$.
# 3. Obtain a correlated random vector  $Z=XL^T$.
# 
# As we have previously seen the Cholesky decomposition of the correlation matrix corr_matrix is impemented in scipy:

# In[7]:

from scipy.linalg import cholesky

upper_cholesky = cholesky(corr_matrix, lower=False)
upper_cholesky


# We set up the parameters for the simulation:

# In[8]:

import numpy as np 
from pandas import bdate_range   # business days

n_days = 21
dates = bdate_range(start=closing.ix[-1].name, periods=n_days)
n_assets = len(symbols)
n_sims = 1000
dt = 1./252
mu = rets.mean().values
sigma = rets.std().values*sqrt(252)
np.random.seed(1234)            # init random number generator for reproducibility


# Now we generate the correlated random values $X$:

# In[63]:

rand_values = np.random.standard_normal(size = (n_days * n_sims, n_assets)) #
corr_values = rand_values.dot(upper_cholesky)*sigma
corr_values


# With everything set up we can start iterating through the time interval. The results for each specific time are saved along the third axis of a pandas Panel.

# In[64]:

prices = Panel(items=range(n_sims), minor_axis=symbols, major_axis=dates)
prices.ix[:, 0, :] = closing.ix[-1].values.repeat(1000).reshape(4,1000).T # set initial values

for i in range(1,n_days):
    prices.ix[:, i, :] = prices.ix[:, i-1,:] * (exp((mu-0.5*sigma**2)*dt +  sqrt(dt)*corr_values[i::n_days])).T    

prices.ix[123, :, :].head()   # show random path


# And thats all! Now it is time to check our results. First a plot of all random paths for AAPL (Apple Inc.).

# In[65]:

prices.ix[::10, :, 'AAPL'].plot(title='AAPL', legend=False);


# We can take a look at the statistics for the last day:

# In[66]:

prices.ix[:, -1, :].T.describe()


# ## Simulating correlated random walks with Copulas

# In[9]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from copulalib.copulalib import Copula
plt.style.use('ggplot')


def generateData():
    global x,y
    x = np.random.normal(size=250)
    y = 2.5*x + np.random.normal(size=250)

# Data and histograms
def plotData():
    global x,y
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.hist(x,bins=20,color='green',alpha=0.8,align='mid')
    plt.title('X variable distribution')
    fig.add_subplot(2,2,3)
    plt.scatter(x,y,marker="o",alpha=0.8)
    fig.add_subplot(2,2,4)
    plt.title('Joint X,Y')
    plt.hist(y,bins=20,orientation='horizontal',color='red',alpha=0.8,align='mid')
    plt.title('Y variable distribution')    
    plt.show()

def generateCopulas():
    global x,y
    fig = plt.figure()

    frank = Copula(x,y,family='frank')
    uf,vf = frank.generate_uv(1000)
    fig.add_subplot(2,2,1)
    plt.scatter(uf,vf,marker='.',color='blue')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.title('Frank copula')

    clayton = Copula(x,y,family='clayton')
    uc,vc = clayton.generate_uv(1000)
    fig.add_subplot(2,2,2)
    plt.scatter(uc,vc,marker='.',color='red')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.title('Clayton copula')

    gumbel = Copula(x,y,family='gumbel')
    ug,vg = gumbel.generate_uv(1000)
    fig.add_subplot(2,2,3)
    plt.scatter(ug,vg,marker='.',color='green')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.title('Gumbel copula')

    plt.show()

#-------------------------------------------------------------------------------
# Run the functions

generateData()
plotData()
generateCopulas()


# In[ ]:



