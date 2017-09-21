# Purpose of this exercise was to solve an ODE across multiple concentrations 

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
import pandas as pd
from pandas import DataFrame

kd = 6e-6 # initial guess for the optimisation
ki = 3  # initial guess for the optimisation
kj = 3e-9 # initial guess for the optimisation
max_mtot = 9.6e-6   # total concentration of monomers 
m0 = np.array([max_mtot,max_mtot/2, max_mtot/4])
Atot = 1e-1 # total area of available interface
Acov_initial = 0 
M_initial = 0
IVs = [Acov_initial, M_initial]
time = np.arange(0,60)
M = np.zeros((len(time),len(m0)))


def dx_dt(x,t,kd,ki,kj,mTot,ATot):
    return [-(kd+kj)*x[0] + ki*(mTot-x[1])*(ATot-x[0]), kd*x[0]]


for i in range(0,len(m0)):
    sol = odeint(dx_dt, IVs, time, args=(kd, ki, kj, m0[i], Atot))
    M[:,i] = sol[:,1]

col_label = ['80','40','20']
M_frame = DataFrame(M,columns=col_label, index=time)

for i in range(0,len(m0)):
    plt.plot(M_frame[col_label[i]],label=col_label[i])
plt.legend()