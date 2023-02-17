#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# January 2023; Last revision: 
# 
# Reads the results from the different DRTO implementations _ CSTR
#=======================================================

# reading data from MATLAB
import scipy.io

# for plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

# for loading data
import pickle

# library that helps you with matrices "numpy":
import numpy as np

# importing statistics to handle statistical operations
import statistics

################
# LOADING DATA #
################
a = scipy.io.loadmat('deltaMeanSqrData.mat')

# Do you want to save the plots?
savePlot = True

#%% BarPLots
fig1 = plt.figure()
 
# creating the bar plot
name = ['$\Delta H$',r'$\rho C_p$','$UA$','$\Psi$','$\Omega$']

plt.axhline(y=0.1, color='k', linestyle=':', linewidth=2)
plt.bar(name,a['dmsqr'][:,0], fill=False, linewidth = 2, width = 0.8)
 
plt.xlabel("Parameters")
plt.ylabel("$\delta^{msqr}$")
plt.title("Delta Mean-square Measure - $C_A$")
plt.show()



if savePlot is True:
    fig1.savefig("deltaMeanSquare_CA.pdf", format="pdf", bbox_inches="tight")
    fig1.savefig('deltaMeanSquare_CA.tif', format='tif', bbox_inches="tight", dpi=600)    

fig2 = plt.figure()

plt.axhline(y=0.1, color='k', linestyle=':', linewidth=2)
plt.bar(name,a['dmsqr'][:,1], fill=False, linewidth = 2, width = 0.8)
 
plt.xlabel("Parameters")
plt.ylabel("$\delta^{msqr}$")
plt.title("Delta Mean-square Measure - $T$")
plt.show()



if savePlot is True:
    fig2.savefig("deltaMeanSquare_T.pdf", format="pdf", bbox_inches="tight")
    fig2.savefig('deltaMeanSquare_T.tif', format='tif', bbox_inches="tight", dpi=600)    
