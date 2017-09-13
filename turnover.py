import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd 
from pandas import Series,DataFrame 

time = np.arange(0,100,0.5)
#ntime = len(time)
#fluo = 10*time - np.random.exponential(5,ntime)
#plt.plot(fluo)

## Import & process fluorescence data 


## Parameters 
kd = 10
ki = 5
mtot = 9.6e-6
Atot = 1e-1


## Solution for M(t) when Acov steady state 
#def Mss_sol(t,M):
#    return 


## Optimise fluorescence data to the steady state solution 


## ODE Numerical evaluator 
def dx_dt(x,t):
    return [-kd*x[0] + ki*(mtot-x[1])*(Atot-x[0]), kd*x[0]]

IVs = [0,0]
X = odeint(dx_dt, IVs, time)
Acov = X[:,0]
M = X[:,1]


## Plot all data i.e. experimental, optimised and ODE evaluation 
plt.plot(time,M)
plt.xlabel("Time")
plt.ylabel("Aggregate conc.")
plt.title("Turn Over model");
