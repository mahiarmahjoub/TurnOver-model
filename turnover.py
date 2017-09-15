import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd 
from scipy import optimize
from scipy.special import lambertw


## user parameters; currently hardcoded, some might be replaced by argparse later
filename = '160816_M24_pH8_80ugmL.csv'
baseline_N_points_to_average = 2
plateau_N_points_to_average = 2


## Import & process fluorescence data 

fluo_data = pd.read_csv(filename)   # import 
[xname,yname] = fluo_data.columns   # extract column names 
expfluo = fluo_data[yname][:]       # create vector containing fluo values
#maxindex = np.argmax(expfluo)       # index of max fluorescence value 
baseline_average = np.mean(expfluo[0:baseline_N_points_to_average]) 
plateau_average = np.mean(expfluo[plateau_N_points_to_average:-1]) 
expfluo_norm = expfluo - baseline_average  # normalise
expfluo_normrel = expfluo_norm/(plateau_average - baseline_average)  # relative data
time = fluo_data[xname]  # extract time vector 

## Parameters 
#kd = 10
#ki = 5
#kj = 10
mtot = 9.6e-6   # total concentration of monomers 
Atot = 1e-1 # total area of available interface
#p = np.array([kd, kj, ki, mtot, Atot])


## Solution for M(t) when Acov steady state 
#def Mss_sol(p, t):
#    y11 = p[2]*mtot/(p[0]+p[1])
#    y12 = p[2]*(mtot-(Atot*p[0]*t))/(p[0]+p[1])
#    y1 = y11*np.exp(y12)
#    return mtot - ((p[0]+p[1])/p[2])*lambertw(y1)
    
Mss_sol = lambda p, t: mtot - ((p[0]+p[1])/p[2])*lambertw(p[2]*mtot/(p[0]+p[1])*np.exp(p[2]*(mtot-(Atot*p[0]*t))/(p[0]+p[1])))
## Optimise fluorescence data to the steady state solution 
errfunc = lambda p, t, y: Mss_sol(p, t) - y # Distance to the target function
p0 = [1, 1, 1]     # Initial guess for the parameters
pfitted = optimize.leastsq(errfunc, p0[:], args=(time,expfluo_normrel))    # fit 

[kd, kj, ki] = pfitted


## ODE Numerical evaluator 
def dx_dt(x,t):
    return [-kd*x[0] + ki*(mtot-x[1])*(Atot-x[0]), kd*x[0]]

IVs = [0,0]
X = odeint(dx_dt, IVs, time)
Acov_num = X[:,0]
Mnum = X[:,1]
Mnum_rel = Mnum/mtot


## Plot all data i.e. experimental, optimised and ODE evaluation 
plt.plot(time,expfluo_normrel, label='Experimental')
plt.plot(time,Mnum_rel,'r', label='Numerical')
plt.plot(time, Mss_sol(time,param)/mtot,'b', label='Steady-state')
plt.xlabel("Time")
plt.ylabel("Aggregate conc.")
plt.title("Turn Over model")
plt.legend()
