import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

time = np.arange(0,40) 
ki = 3118206  # initial guess for the optimisation
kj = 0.1 # initial guess for the optimisation
kd = 10
max_mtot = 9.6e-6   # total concentration of monomers 
Atot = 1e-7     # total area of available interface
Acov_initial = 0 
M_initial = 0
IVs = [Acov_initial, M_initial]

def Acov(x,t,kd,ki,kj,mTot,ATot):
    return [-(kd+kj)*x[0] + ki*(mTot-x[1])*(ATot-x[0]), kd*x[0]]

sol = odeint(Acov,  IVs, time, args=(kd, ki, kj, max_mtot, Atot))


def forward_flux(x,t,kd,ki,kj,mTot,ATot):
    xm = x[:,1]
    xacov = x[:,0]
    return ki*(mTot-xm)*(ATot-xacov)

def reverse_flux(x,t,kd,ki,kj,mTot,ATot):
    xacov = x[:,0]
    return (kd)*xacov


plt.figure(0)
plt.plot(time, 1000000*forward_flux(sol,time,kd,ki,kj,max_mtot,Atot), label='surface coverage')
plt.plot(time, 1000000*reverse_flux(sol,time,kd,ki,kj,max_mtot,Atot), label='fibril formation')
plt.title('Flux controlling Acov(t)')
plt.xlabel('Time (min)')
plt.ylabel('Rate (uM/s)')
plt.legend()
plt.savefig('turnover-Acov-flux-analysis.png', dpi=150)


