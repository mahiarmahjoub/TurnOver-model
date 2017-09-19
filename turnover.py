## Main program

# function of this program:
    # import and process i.e. normalise etc. fluorescence data
    # fit steady state solution to the experimental data 
    # use fitted parameters for the ODE numerical evaluation and cross-checking 


## Required packages 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd 
from scipy import optimize
from scipy.special import lambertw
import argparse

## User parameters
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="insert the name of file")
parser.add_argument("baseline", type=int,
                    help="# points to calculate baseline fluorescence")
parser.add_argument("endpoint", 
                    help="# points at the end to derive relative fluorescence",
                    type=int)
args = parser.parse_args()

filename = args.filename
baseline_N_points_to_average = args.baseline
plateau_N_points_to_average = args.endpoint

#filename = 'M2480averagepH8.csv'
#baseline_N_points_to_average = 2
#plateau_N_points_to_average = 5


## Import & process fluorescence data 

fluo_data = pd.read_csv(filename)    # import 
[xname,yname] = fluo_data.columns    # extract column names 
expfluo = fluo_data[yname]           # create vector containing fluo values
baseline_average = np.mean(expfluo[0:baseline_N_points_to_average]) 
plateau_average = np.mean(expfluo[-plateau_N_points_to_average:-1]) 
expfluo_norm = expfluo - baseline_average  # normalise
expfluo_normrel = expfluo_norm/(plateau_average - baseline_average)  # relative data
time = fluo_data[xname]  # extract time vector 

## Parameters 
kd_guess = 10 # initial guess for the optimisation
ki_guess = 1  # initial guess for the optimisation
kj_guess = 0.1 # initial guess for the optimisation
mtot = 9.6e-6   # total concentration of monomers 
Atot = 1e-1 # total area of available interface
Acov_initial = 0 
M_initial = 0


## Solution for M(t) when Acov steady state 
def Mss_sol(p,t,mTot,ATot,Mini): 
    a = abs(p[0])
    b = abs(p[1])
    c = abs(p[2])
    out1 = (a + c)/b
    out2 = (b*(mTot-Mini))/(a + c)
    out3 = (b*ATot*a*t)/(a + c)
    out4 = np.exp(out2 - out3)
    out5 = np.real(lambertw(out2*out4))
    
    out = mTot - out5*out1
    return out
    
## Optimise fluorescence data to the steady state solution 
errfunc = lambda p,t,y,mTot,ATot,Mini: Mss_sol(p,t,mTot,ATot,Mini) - y # Distance to the target function


p0 = [kd_guess,ki_guess,kj_guess]     # Initial guess for the parameters
fit_result = optimize.leastsq(errfunc,
                        p0[:], 
                         args=(time,expfluo_normrel,mtot,Atot,M_initial),
                         full_output = True,
                         ftol = 1e-12, maxfev = 1000) # fit 

pfitted = fit_result[0]
[kd, ki, kj] = abs(pfitted)

## ODE Numerical evaluator 
def dx_dt(x,t,kd,ki,kj,mTot,ATot):
    return [-(kd+kj)*x[0] + ki*(mTot-x[1])*(ATot-x[0]), kd*x[0]]

IVs = [Acov_initial, M_initial]
X = odeint(dx_dt, IVs, time, args=(kd, ki, kj, mtot, Atot))
Acov_num = X[:,0]
Mnum = X[:,1]
Mnum_rel = Mnum/mtot


## Plot all data i.e. experimental, optimised and ODE evaluation 
plt.plot(time,expfluo_normrel, label='Experimental')
plt.plot(time,Mnum_rel,'r', label='Numerical')
plt.plot(time, Mss_sol(abs(pfitted),time,mtot,Atot,M_initial)/mtot, 'b', label='Steady-state')
#plt.plot(time, Mss_sol(p0,time,mtot,Atot,M_initial)/mtot, 'b', label='Steady-state')
plt.xlabel("Time")
plt.ylabel("Aggregate conc.")
plt.title("Turn Over model")
plt.legend()