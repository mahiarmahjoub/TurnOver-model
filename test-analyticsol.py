# This program was created to check that the steady-state solution and 
# ODE of the system are correct and agree. 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import lambertw

## Parameters 
kd = 10
ki = 5
kj = 10
mtot = 9.6e-6   # total concentration of monomers 
Atot = 1e-1 # total area of available interface
Acov_initial = 0 
M_initial = 0
p = np.array([kd, ki, kj, mtot, Atot])
time = np.linspace(0,50,num=100)

## Solution for M(t) when Acov steady state 
def Mss_sol(p,t,Mini):
    out1 = (p[0] + p[2])/p[1]
    out2 = (p[1]*(p[3]-Mini))/(p[0] + p[2])
    out3 = (p[1]*p[4]*p[0]*t)/(p[0] + p[2])
    out4 = np.exp(out2 - out3)
    out5 = np.real(lambertw(out2*out4))
    
    out = p[3] - out5*out1
    return out

## ODE Numerical evaluator 
def dx_dt(x,t,kd,ki,kj,mtot,Atot):
    return [-(kd+kj)*x[0] + ki*(mtot-x[1])*(Atot-x[0]), kd*x[0]]

IVs = [Acov_initial, M_initial]
X = odeint(dx_dt, IVs, time, args=(kd, ki, kj, mtot, Atot))
Acov_num = X[:,0]
Mnum = X[:,1]
Mnum_rel = Mnum/mtot


## Plot all data i.e. experimental, optimised and ODE evaluation 
plt.plot(time,Mnum_rel, 'r', label='Numerical')
plt.plot(time, Mss_sol(p,time,M_initial)/mtot, 'b', label='Steady-state')
plt.xlabel("Time")
plt.ylabel("Aggregate conc.")
plt.title("Turn Over model")
plt.legend()

# the steady-state solution and numerical solution agree well. They are now 
# ready to be incorporated into the main turnover.py program. 