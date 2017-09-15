import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd 
from scipy import optimize
from scipy.special import lambertw


## Import & process fluorescence data 
filename = '160816_M24_pH8_80ugmL.csv'
fluo_data = pd.read_csv(filename)   # import 
[xname,yname] = fluo_data.columns   # extract column names 
expfluo = fluo_data[yname][:]       # create vector containing fluo values
maxindex = np.argmax(expfluo)       # index of max fluorescence value 
expfluo_norm = expfluo[:maxindex] - expfluo[0]  # normalise
expfluo_normrel = expfluo_norm/np.amax(expfluo_norm)    # relative data
time = fluo_data[xname][:maxindex]  # extract time vector 


## Parameters 
kd = 10
ki = 5
kj = 10
mtot = 9.6e-6   # total concentration of monomers 
Atot = 1e-1 # total area of available interface
#p = np.array([kd, kj, ki, mtot, Atot])


## Solution for M(t) when Acov steady state 
def Mss_sol(p, t):
    a = abs(p[0])
    b = abs(p[1])
    c = abs(p[2])
    y11 = c*mtot/(a+b)
    y12 = c*(mtot-(Atot*a*t))/(a+b)
    y1 = y11*np.exp(y12)
    out = mtot - ((a+b)/c)*np.real(lambertw(y1))
    return out
    
#Mss_sol = lambda p, t: mtot - ((p[0]+p[1])/p[2])*lambertw(p[2]*mtot/(p[0]+p[1])*np.exp(p[2]*(mtot-(Atot*p[0]*t))/(p[0]+p[1])))
## Optimise fluorescence data to the steady state solution 
errfunc = lambda p, t, y: Mss_sol(p, t) - y # Distance to the target function
p0 = [kd,ki,kj]     # Initial guess for the parameters
pfitted,success = optimize.leastsq(errfunc, p0[:], args=(time,expfluo_normrel))  # fit 

[kd, ki, kj] = abs(pfitted)


## ODE Numerical evaluator 
def dx_dt(x,t,kd,ki,kj):
    return [-kd*x[0] + ki*(mtot-x[1])*(Atot-x[0]), kd*x[0]]

IVs = [0,0]
X = odeint(dx_dt, IVs, time, args=(kd,ki,kj))
Acov_num = X[:,0]
Mnum = X[:,1]
Mnum_rel = Mnum/mtot


## Plot all data i.e. experimental, optimised and ODE evaluation 
plt.plot(time,expfluo_normrel, label='Experimental')
plt.plot(time,Mnum_rel,'r', label='Numerical')
plt.plot(time, Mss_sol(pfitted,time)/mtot,'b', label='Steady-state')
plt.xlabel("Time")
plt.ylabel("Aggregate conc.")
plt.title("Turn Over model")
plt.legend()