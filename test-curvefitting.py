## Curve fitting - Test 

# this exercise was used to practise and test a basis for the curve-fitting algorithm eventually to be used in the main program
# the code was perceived with help from the following resource: 
# scipy-cookbook (from GitHub) https://github.com/scipy/scipy-cookbook/blob/master/ipython/FittingData.ipynb

import numpy as np 
import matplotlib.pyplot as plt
from pandas import DataFrame 
from scipy import optimize

def comparefit(pfitted,a,bmean):  # creates & exports a table comparing parameters 
    pcompare = np.array([[pfitted[0], a],[pfitted[1],bmean]])
    pcompare = DataFrame(pcompare, index=['a','b'], columns=['fitted','original'])
    return print(pcompare)

t = np.linspace(0,100,num=50)   # create a x-domain
a = np.random.randn()           # generate a random gradient 
b = np.random.exponential(3,len(t)) # generate a random vector for y-intercept
bmean = np.mean(b)
yt = a*t + b    # evaluate the y-values of the function from t

fitfunc = lambda p, t: p[0]*t + p[1]    # Target function
errfunc = lambda p, t, y: fitfunc(p, t) - y # Distance to the target function
p0 = [1, 1]     # Initial guess for the parameters
pfitted, success = optimize.leastsq(errfunc, p0[:], args=(t,yt))    # fit 
comparefit(pfitted,a,bmean) # generates a table comparing the fitted values 


# Plot the (1) Fit & (2) Residual
plt.subplot(2,1,1)
plt.plot(t, yt, "o", t, fitfunc(pfitted,t),"-")
plt.subplot(2,1,2)
plt.plot(t,errfunc(pfitted,t,yt),'bo')
