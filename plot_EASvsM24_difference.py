import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas import DataFrame

filename = ['160622_M24_pH8.csv','160706_EAS_pH8.csv']
baseline_N_points_to_average = 0
plateau_N_points_to_average = 3

expfluoM24 = pd.read_csv(filename[0],index_col=0)    # import 
expfluoEAS = pd.read_csv(filename[1],index_col=0)    # import 

col_label_M24 = expfluoM24.columns  # extract the column names 
col_label_EAS = expfluoEAS.columns  # extract the column names 

nconc = min([len(col_label_M24), len(col_label_EAS)])  # number of sets of different concentrations 
maxt_EAS = max(expfluoEAS.index)
maxt_M24 = max(expfluoM24.index)
timeM24 = np.array(expfluoM24.index)
timeEAS = np.array(expfluoEAS.index)
colour = ['b','g','r','c','m']   # select colours for the plot 

baseline_average_M24 = np.mean(expfluoM24[0:baseline_N_points_to_average]) 
baseline_average_EAS = np.mean(expfluoEAS[0:baseline_N_points_to_average]) 
plateau_average_M24 = np.mean(expfluoM24[maxt_M24-plateau_N_points_to_average:maxt]) 
plateau_average_EAS = np.mean(expfluoEAS[maxt_EAS-plateau_N_points_to_average:maxt]) 

expfluo_norm_M24 = expfluoM24 - baseline_average_M24  # normalise
expfluo_norm_EAS = expfluoEAS - baseline_average_EAS  

expfluo_normrel_M24 = pd.DataFrame.as_matrix(expfluo_norm_M24/(plateau_average_M24 - baseline_average_M24)) # relative data
expfluo_normrel_EAS = pd.DataFrame.as_matrix(expfluo_norm_EAS/(plateau_average_EAS - baseline_average_EAS)) # relative data

plt.figure(0)
for i in range(0,nconc):
    plt.subplot(nconc,1,i+1)
    plt.plot(timeEAS, expfluo_normrel_EAS[:,i], label=col_label_EAS[i], color=colour[i], 
             linewidth=3)
    plt.plot(timeM24, expfluo_normrel_M24[:,i],'o', label=col_label_M24[i], 
             color=colour[i], alpha=0.3)
    if i == 0:
        plt.title('Fitted TurnOver model')
    elif i == nconc:
        plt.xlabel('Time (min)')
    plt.ylabel('Relative aggregate conc')
    plt.legend(loc='center right')
    #plt.text(maxt-10,0.3,filename[0].replace('.csv',' '))
    #plt.savefig(filename.replace('.csv','_fit.png'),dpi=300)