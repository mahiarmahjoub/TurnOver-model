# Purpose of this exercise was to solve an ODE across multiple concentrations 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd 
from scipy import optimize
from scipy.special import lambertw
import argparse
from pandas import DataFrame


filename = 'M24averagepH8.csv'
plateau_N_points_to_average = 5
baseline_N_points_to_average = 0

expfluo = pd.read_csv(filename,index_col=0)    # import 
time = np.array(expfluo.index)  # extract time vector 
maxt = max(time)
baseline_average = np.mean(expfluo[0:baseline_N_points_to_average]) 
plateau_average = np.mean(expfluo[maxt-plateau_N_points_to_average:maxt]) 
expfluo_norm = expfluo - baseline_average  # normalise
expfluo_normrel = expfluo_norm/(plateau_average - baseline_average)  # relative data
expfluo_normrel = pd.DataFrame.as_matrix(expfluo_normrel)   # change from pandas to numpy array 


kd_guess = 6e-6 # initial guess for the optimisation
ki_guess = 3  # initial guess for the optimisation
kj_guess = 3e-9 # initial guess for the optimisation
max_mtot = 9.6e-6   # total concentration of monomers 
m0 = np.array([max_mtot,max_mtot/2, max_mtot/4])
Atot = 1e-1 # total area of available interface
Acov_initial = 0 
M_initial = 0
IVs = [Acov_initial, M_initial]
M = np.zeros((len(time),len(m0)))



def Mss_sol(p,t,mTot,ATot,Mini): 
    a = abs(p[0])
    b = abs(p[1])
    c = abs(p[2])
    out1 = (a + c)/b
    out2 = (b*(mTot-Mini))/(a + c)
    out3 = (b*ATot*a*t)/(a + c)
    out4 = np.exp(out2 - out3)
    out5 = np.real(lambertw(out2*out4))
    out = 1 - out5*out1/mTot
    return out
    
## Optimise fluorescence data to the steady state solution 
def errfunc(p,t,y,mTot,ATot,Mini): 
    out_sum = 0
    for i in range(0,len(mTot)):
        out = ((Mss_sol(p,t,mTot[i],ATot,Mini) - y[:,i])**2)
        out_sum = out_sum + np.sum(out)
        out_ave = out_sum/len(mTot)
    return out_ave


p0 = [kd_guess,ki_guess,kj_guess]     # Initial guess for the parameters
fit_result = optimize.basinhopping(errfunc,
                        p0,disp = True,niter = 100,
                        minimizer_kwargs = {'method':'Nelder-Mead', 'args':(time,expfluo_normrel,m0,Atot,M_initial)}) # fit 

pfitted = fit_result['x']
[kd, ki, kj] = abs(pfitted)


col_label = expfluo.columns
#M_frame = DataFrame(M,columns=col_label, index=time)
def dx_dt(x,t,kd,ki,kj,mTot,ATot):
    return [-(kd+kj)*x[0] + ki*(mTot-x[1])*(ATot-x[0]), kd*x[0]]


for i in range(0,len(m0)):
    sol = odeint(dx_dt, IVs, time, args=(kd, ki, kj, m0[i], Atot))
    M[:,i] = sol[:,1]

for i in range(0,len(m0)):
    plt.subplot(1,2,1)
    plt.plot(time,Mss_sol(pfitted,time,m0[i],Atot,M_initial),label=col_label[i])
    plt.plot(time,expfluo_normrel[:,i],'o')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(time,M[:,i],label=col_label[i])
    plt.legend()