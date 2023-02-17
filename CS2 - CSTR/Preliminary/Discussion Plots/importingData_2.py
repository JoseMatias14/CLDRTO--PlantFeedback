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
a = scipy.io.loadmat('CSTRsensitivities.mat')
b = scipy.io.loadmat('MC_parameters_CLv2_med.mat')


# Do you want to save the plots?
savePlot = True

#%% Sensitivities
majorR = 10
minorR = 2

fig1, (ax1, ax2) = plt.subplots(2, sharex=True)
fig1.suptitle('Sensitivities CSTR: Concentration of A')

ax1.plot(a['tsimgrid'].T, a['SX_Array'][0,0][0,:],'b', linewidth=4)
ax1.set(ylabel='d $C_A$/d $\Delta U$')
ax1.minorticks_on()
#ax1.set_ylim([0.0025, 0.03])
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(a['tsimgrid'].T, a['SX_Array'][0,0][2,:].T,'b', linewidth=4)
ax2.set(ylabel=r'd $C_A$/d $UA$', xlabel='time [h]')
ax2.minorticks_on()
#ax2.set_ylim([0.0025, 0.03])
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

fig1.savefig("sensCA.pdf", format="pdf", bbox_inches="tight")
fig1.savefig('sensCA.tif', format='tif', bbox_inches="tight", dpi=600)

fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
fig2.suptitle('Sensitivities CSTR: Temperature')

ax1.plot(a['tsimgrid'].T, a['SX_Array'][0,1][0,:].T,'b', linewidth=4)
ax1.set(ylabel=r'd $T$/d $\Delta H$')
ax1.minorticks_on()
#ax1.set_ylim([-0.01, -0.65])
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(a['tsimgrid'].T, a['SX_Array'][0,1][2,:].T,'b', linewidth=4)
ax2.set(ylabel=r'd $T$/d $UA$', xlabel='time [h]')
ax2.minorticks_on()
#ax2.set_ylim([-0.01, 0.01])
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

fig2.savefig("sensT.pdf", format="pdf", bbox_inches="tight")
fig2.savefig('sensT.tif', format='tif', bbox_inches="tight", dpi=600)
    
#%% INPUTS
majorR = 20
minorR = 10

# # simulation time (from disturbance file)
tgrid = [2*k for k in range(np.int8(b['nEnd'].item()) + 1)]

uLab = ['$C_{A,f}$','$T_{f}$','$T^{SP}$']
thetaLab = ['$\Delta H$','$UA$']

fig3, axs = plt.subplots(3, sharex=True)
fig3.suptitle('Inputs - Effect of $\Delta H$') 

for ii in range(3): 
    for kk in range(3 + np.int8(b['nSim'].item())):
        axs[ii].step(tgrid,b['UPlot'][0,kk][ii,:],'g',alpha=0.1,linewidth=2)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, b['tEnd']]) 
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=uLab[ii])
     
axs[2].set(xlabel='time [h]') 

if savePlot is True:
    fig3.savefig("CSTRinputsMC_DU.pdf", format="pdf", bbox_inches="tight")
    fig3.savefig('CSTRinputsMC_DU.tif', format='tif', bbox_inches="tight", dpi=600)    
    
fig4, axs = plt.subplots(3, sharex=True)
fig4.suptitle('Inputs - Effect of $UA$') 

for ii in range(3): 
    for kk in range(3 + np.int8(b['nSim'].item())):
        axs[ii].step(tgrid,b['UPlot'][1,kk][ii,:],'g',alpha=0.1,linewidth=2)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, b['tEnd']]) 
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=uLab[ii])
     
axs[2].set(xlabel='time [h]') 

if savePlot is True:
    fig4.savefig("CSTRinputsMC_UA.pdf", format="pdf", bbox_inches="tight")
    fig4.savefig('CSTRinputsMC_UA.tif', format='tif', bbox_inches="tight", dpi=600)        

#%% OBJECTIVE FUNCTION
fig5, ax = plt.subplots()

for kk in range(3 + np.int8(b['nSim'].item())):
    ax.plot(tgrid,b['OFPlot'][0,kk].T,'g',alpha=0.1,linewidth=2)
        
ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)


fig5.suptitle('Objective Function - Effect of $\Delta H$')
ax.set(xlabel='time [h]', ylabel='$\phi \ [-]$')  

if savePlot is True:
    fig5.savefig("CSTR_OF_MC_DU.pdf", format="pdf", bbox_inches="tight")
    fig5.savefig('CSTR_OF_MC_DU.tif', format='tif', bbox_inches="tight", dpi=600)    
    
fig6, ax = plt.subplots()

for kk in range(3 + np.int8(b['nSim'].item())):
    ax.plot(tgrid,b['OFPlot'][1,kk].T,'g',alpha=0.1,linewidth=2)
        
ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)


fig6.suptitle('Objective Function - Effect of $UA$')
ax.set(xlabel='time [h]', ylabel='$\phi \ [-]$')  

if savePlot is True:
    fig6.savefig("CSTR_OF_MC_UA.pdf", format="pdf", bbox_inches="tight")
    fig6.savefig('CSTR_OF_MC_UA.tif', format='tif', bbox_inches="tight", dpi=600)    

#%% PARAMETER DISTRIBUTION
theta_nom = [-5.960,1.6]
theta_sig = [0.01,0.01]

#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

theta_tgrid_0 = np.linspace(theta_nom[0] - 3*theta_sig[0],theta_nom[0] + 3*theta_sig[0],20)
theta_tgrid_1 = np.linspace(theta_nom[1] - 3*theta_sig[1],theta_nom[1] + 3*theta_sig[1],20)

#Apply function to the data.
pdf_0 = normal_dist(theta_tgrid_0,theta_nom[0],theta_sig[0])
pdf_1 = normal_dist(theta_tgrid_1,theta_nom[1],theta_sig[1])

fig7, (ax1, ax2) = plt.subplots(1,2, sharey=True)
fig7.suptitle('Parameter Distribution') 

ax1.axvline(x=theta_nom[0], color='k', linestyle=':', linewidth=2)
ax1.plot(theta_tgrid_0,pdf_0,linewidth=4)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
ax1.set(xlabel='$\Delta H$ [1e3 kcal/kmol]', ylabel= 'pdf')


ax2.axvline(x=theta_nom[1], color='k', linestyle=':', linewidth=2)
ax2.plot(theta_tgrid_1,pdf_1,linewidth=4)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
ax2.set(xlabel='$UA$ [1e3 kcal/(K \ h)]')

if savePlot is True:
    fig7.savefig("CSTR_parDist.pdf", format="pdf", bbox_inches="tight")
    fig7.savefig('CSTR_parDist.tif', format='tif', bbox_inches="tight", dpi=600)        



