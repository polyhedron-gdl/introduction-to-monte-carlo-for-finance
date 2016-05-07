
# coding: utf-8

# # Introduction to Monte Carlo Simulation in Finance
# 
# ## Derivatives CVA calculation example Monte-Carlo with python

# Here we'll show an example of code for CVA calculation (Credit Valuation Adjustment) using python with simple Monte-Carlo method with portfolio consisting just of a single interest rate swap.It's easy to generalize code to include more financial instruments.
# 
# #### CVA calculation algorithm:
# 
# 1) Simulate yield curve at future dates
# 
# 2) Calculate your derivatives portfolio NPV (net present value) at each time point for each scenario
# 
# 3) Calculate CVA as sum of Expected Exposure multiplied by probability of default at this interval
# 
# $$ CVA=(1−R) \int DF(t)EE(t)dQ_t $$
# 
# where $R$ is the Recovery Rate (normally set to 40%) $EE(t)$ is the expected exposure at time $t$ and $dQ_t$ the survival probability density, $DF(t)$ is the discount factor at time $t$.
# 
# #### Outline
# 
# 1. In this simple example we will use Hull White model to generate future yield curves. In practice many banks use some yield curve evolution models based on this model. As you can see in the slides, in Hull White model the short rate $r_t$ is distributed normally with known mean and variance.
# 
# 2. For each point of time we will generate whole yield curve based on short rate. Then we will price our interest rate swap on each of these curves;
# 
# 3. to approximate CVA we will use BASEL III formula for regulatory capital charge approximating default probability [or survival probability ] as $exp(-S_T/(1-R))$ so we get
# 
# $$
# CVA=(1−R) \sum\limits_i \frac{EE(T_i)^\star + EE(T_{i−1}^\star}{2}
# \left( e^{−S(T_{i−1})/(1−R)}−e^{−S(T_i)/(1−R)} \right)
# $$
# where $EE^\star$ is the discounted Expected Exposure of portfolio.
# 
# #### Details
# 
# For this first example we'll take 2% flat forward yield curve. 

# In[ ]:



