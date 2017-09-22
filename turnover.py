## Main program

# function of this program:
    # import and process i.e. normalise etc. fluorescence data
    # globally fit steady state solution to the experimental data 
    # use fitted parameters for the ODE numerical evaluation and cross-checking 
    # export fitting plot and parameter values 


## -- Required packages ------------------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd 
from scipy import optimize
from scipy.special import lambertw
import argparse
from pandas import DataFrame

## -- Prompt parameters ------------------------------------------------------
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

## -- Import & process fluorescence data -------------------------------------
expfluo = pd.read_csv(filename,index_col=0)    # import 
time = np.array(expfluo.index)  # extract time vector 
col_label = expfluo.columns  # extract the column names 
nconc = len(col_label)  # number of sets of different concentrations 
maxt = max(time)
baseline_average = np.mean(expfluo[0:baseline_N_points_to_average]) 
plateau_average = np.mean(expfluo[maxt-plateau_N_points_to_average:maxt]) 
expfluo_norm = expfluo - baseline_average  # normalise
expfluo_normrel = expfluo_norm/(plateau_average - baseline_average)  # relative data
expfluo_normrel = pd.DataFrame.as_matrix(expfluo_normrel)   # change from pandas to numpy array 

## -- Set parameters, initial values and vectors -----------------------------
kd_guess = 6e-6 # initial guess for the optimisation
ki_guess = 3  # initial guess for the optimisation
kj_guess = 3e-9 # initial guess for the optimisation
max_mtot = 9.6e-6   # total concentration of monomers 
colour = ['b','g','r','c','m']   # select colours for the plot 
exp_details = 'EASd15 pH 8.0'  # insert protein name + other conditions 
m0 = np.zeros(nconc)  
for i in range(0,nconc):   # generate an array containing all monomer conc
    if i == 0:
        m0[0] = max_mtot 
    else:
        m0[i] = m0[i-1]/2
Atot = 1e-7   # total area of available interface
Acov_initial = 0 
M_initial = 0
IVs = [Acov_initial, M_initial]
M = np.zeros((len(time),nconc))
Mrel = M

## -- Solution for M(t) when Acov steady state -------------------------------
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
    
## -- Define the least square error function for fitting globally ------------
def errfunc(p,t,y,mTot,ATot,Mini): 
    out_sum = 0
    nconc = len(mTot)
    for i in range(0,nconc):
        out = ((Mss_sol(p,t,mTot[i],ATot,Mini) - y[:,i])**2)
        out_sum = out_sum + np.sum(out)
    return out_sum

## -- Fitting ----------------------------------------------------------------
p0 = [kd_guess,ki_guess,kj_guess]     # Initial guess for the parameters
fit_result = optimize.basinhopping(errfunc,
                        p0,disp = True,niter = 10,
                        minimizer_kwargs = {'method':'Nelder-Mead', 
                                            'args':(time,expfluo_normrel,
                                                    m0,Atot,M_initial)}) # fit 
pfitted = fit_result['x']
abs_pfitted = abs(pfitted)
[kd, ki, kj] = abs_pfitted  # assign fitted parameters 
pfitted_frame = DataFrame([pfitted,abs_pfitted],index=['Fit','abs(Fit)'],
                          columns=['kd','ki','kj'])
pfitted_frame.to_csv(filename.replace('.csv','_param.txt'), header=True,
                     index=True)  # export paramters to csv 

## -- Define ODE functions ---------------------------------------------------
def dx_dt(x,t,kd,ki,kj,mTot,ATot):
    return [-(kd+kj)*x[0] + ki*(mTot-x[1])*(ATot-x[0]), kd*x[0]]

## -- Solve the ODEs using the fitted parameters -----------------------------
for i in range(0,nconc):
    sol = odeint(dx_dt, IVs, time, args=(kd, ki, kj, m0[i], Atot))
    M[:,i] = sol[:,1]
    Mrel[:,i] = sol[:,1]/max(sol[:,1])

## -- Plot the results & export an image of the fitted solution --------------
plt.figure(0)
for i in range(0,nconc):
    plt.plot(time,Mss_sol(pfitted,time,m0[i],Atot,M_initial),
             label=col_label[i], color=colour[i], linewidth=3)
    plt.plot(time,expfluo_normrel[:,i],'o',color=colour[i],alpha=0.3)
    plt.title('Fitted TurnOver model')
    plt.xlabel('Time (min)')
    plt.ylabel('Relative aggregate conc')
    plt.legend(loc='center right')
    plt.table(cellText=np.round(np.transpose([pfitted]),1),
              colWidths = [0.2]*2,
              rowLabels=['kd','ki','kj'],
              colLabels=['Fit','abs(Fit)'],
              loc='lower right')
    plt.text(31,0.25,exp_details)
    plt.savefig(filename.replace('.csv','_fit.png'),dpi=300)
plt.figure(1)
for i in range(0,len(m0)):
    plt.plot(time,Mrel[:,i],label=col_label[i], color=colour[i], linewidth=3)
    plt.plot(time,Mss_sol(pfitted,time,m0[i],Atot,M_initial),
             'o', color=colour[i], alpha=0.3)
    plt.title('Check steady-state solution vs numerical')
    plt.xlabel('Time (min)')
    plt.legend(loc='lower right')

