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
a = scipy.io.loadmat('test3_BIAS_2p.mat')
b = scipy.io.loadmat('test3_MHE_2p.mat')
nData = 2

# # simulation time (from disturbance file)
tgrid = [np.int8(a['dt_sys'].item())*k for k in range(np.int8(a['nEnd'].item()) + 1)]
# deltaPlot = 5*12 # plot one point every 5 minutes

# # for plottimg
majorR = 20
minorR = 10

# Do you want to save the plots?
savePlot = True

# ORDER: 
# 0: biasModelArray
# 1: dt_sys
# 2: execTimeArray
# 3: Ne
# 4: nEnd
# 5: Np
# 6: OFPlantArray
# 7: SolDRTOFlag
# 8: SolMHEFlag
# 9: tEnd
# 10: thetaHatArray
# 11: thetaHatTrajectory
# 12: thetaPlantArray
# 13: uDRTOArray
# 14: uPlantArray
# 15: XHatArray
# 16: XHatTrajectory
# 17: XMeasArray
# 18: XModelArray
# 19: XPlantArray


#%% Labels
xLab = ['$C_A$ [kmol]','$T$ [K]']
uLab = ['$C_{A,f}$ [kmol]','$T_{f}$ [K]','$T^{SP}$ [K]']
thetaLab = ['$\Delta H$ [1e3 kcal/kmol]','$\Psi$ [-]']
methodLabel = ["bias", "st + par"]

colorMethod = ['b', 'r']
markerMethod = ['bo', 'rv']

    
#%% STATES 
fig1, axs = plt.subplots(3, sharex=True)
fig1.suptitle('Plant States') 

for ii in range(2): 
    #axs[ii].axhline(y=xStarSS[ii], color='k', linestyle=':', linewidth=2)
    axs[ii].plot(tgrid,b['XPlantArray'][ii,:],colorMethod[1], label =methodLabel[1],linewidth=4)
    axs[ii].plot(tgrid,a['XPlantArray'][ii,:],colorMethod[0], label =methodLabel[0],linewidth=4)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, a['tEnd']]) 
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=xLab[ii])
    
axs[2].plot(tgrid,b['XPlantArray'][3,:],colorMethod[1], label =methodLabel[1],linewidth=4)
axs[2].plot(tgrid,a['XPlantArray'][3,:],colorMethod[0], label =methodLabel[0],linewidth=4)
 
axs[2].minorticks_on()
axs[2].xaxis.set_major_locator(MultipleLocator(majorR))
axs[2].xaxis.set_minor_locator(MultipleLocator(minorR))
axs[2].yaxis.set_tick_params(which='minor', bottom=False)
axs[2].set_xlim([0, a['tEnd']]) 
axs[2].grid(b=True, which='major', linestyle='-')
axs[2].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
 
axs[2].set(ylabel='$T_c$ [K]')
  
axs[2].set(xlabel='time [h]') 
axs[2].legend(loc='best')    

if savePlot is True:
    fig1.savefig("CSTRstates.pdf", format="pdf", bbox_inches="tight")
    fig1.savefig('CSTRstates.tif', format='tif', bbox_inches="tight", dpi=600)    


#%% INPUTS
fig2, axs = plt.subplots(3, sharex=True)
fig2.suptitle('Inputs') 

for ii in range(3): 
    #axs[ii].axhline(y=uStarSS[ii], color='k', linestyle=':', label ='SS Opt.' , linewidth=2)
    axs[ii].step(tgrid,b['UPlantArray'][ii,:],colorMethod[1], label =methodLabel[1],linewidth=4)
    axs[ii].step(tgrid,a['UPlantArray'][ii,:],colorMethod[0], label =methodLabel[0],linewidth=4)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, a['tEnd']]) 
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=uLab[ii])
     
axs[2].set(xlabel='time [h]') 
axs[2].legend(loc='best')


if savePlot is True:
    fig2.savefig("CSTRinputs.pdf", format="pdf", bbox_inches="tight")
    fig2.savefig('CSTRinputs.tif', format='tif', bbox_inches="tight", dpi=600)    

#%% OBJECTIVE FUNCTION
fig3, ax = plt.subplots()

#ax.axhline(y=OF_SS, color='k', linestyle=':', linewidth=2)
#ax.plot(tgrid, np.array(a['OFPlantArray'].T/a['dt_sys'],ndmin=0),colorMethod[0], label =methodLabel[0], linewidth=4)
#ax.plot(tgrid, np.array(b['OFPlantArray'].T/b['dt_sys'],ndmin=0),colorMethod[1], label =methodLabel[1], linewidth=4)
ax.plot(tgrid, (1 - b['XPlantArray'][0,:]/b['UPlantArray'][0,:]),colorMethod[1], label =methodLabel[1], linewidth=4)
ax.plot(tgrid, (1 - a['XPlantArray'][0,:]/a['UPlantArray'][0,:]),colorMethod[0], label =methodLabel[0], linewidth=4)


ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax.legend(loc='best')

fig3.suptitle('Objective Function')
ax.set(xlabel='time [h]', ylabel='$\phi \ [-]$')  

if savePlot is True:
    fig3.savefig("CSTR_OF.pdf", format="pdf", bbox_inches="tight")
    fig3.savefig('CSTR_OF.tif', format='tif', bbox_inches="tight", dpi=600)    

#%% BIAS 
fig4, axs = plt.subplots(2, sharex=True)
fig4.suptitle('Computed Bias') 

for ii in range(2): 
    axs[ii].plot(tgrid,a['biasModelArray'][ii,:],colorMethod[0], label =methodLabel[0],linewidth=4)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, a['tEnd']])
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=xLab[ii])
     
axs[nData-1].set(xlabel='time [h]') 
axs[nData-1].legend(loc='best')

axs[0].set_ylim([-1,1])
axs[1].set_ylim([-10,10])

if savePlot is True:
    fig4.savefig("CSTRCompBias.pdf", format="pdf", bbox_inches="tight")
    fig4.savefig('CSTRCompBias.tif', format='tif', bbox_inches="tight", dpi=600)    

#%% STATES AND PARAMETERS 
fig5, axs = plt.subplots(2, sharex=True)
fig5.suptitle('Parameters: Estimated vs. True') 

for ii in range(2): 
    axs[ii].plot(tgrid[16:],b['thetaPlantArray'][ii,16:],color='k', linestyle=':', label = 'True', linewidth=2)
    axs[ii].plot(tgrid[16:],b['thetaHatArray'][ii,16:],markerMethod[1], label = 'Est', markersize=4)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, a['tEnd']]) 
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=thetaLab[ii])
     
axs[nData-1].set(xlabel='time [h]') 
axs[nData-1].legend(loc='best') 


if savePlot is True:
    fig5.savefig("CSTREstPar.pdf", format="pdf", bbox_inches="tight")
    fig5.savefig('CSTREstPar.tif', format='tif', bbox_inches="tight", dpi=600)    

fig6, axs = plt.subplots(2, sharex=True)
fig6.suptitle('States: Estimated vs. True') 

for ii in range(2): 
    axs[ii].plot(tgrid[16:],b['XPlantArray'][ii,16:],color='k', linestyle=':', label = 'True', linewidth=2)
    axs[ii].plot(tgrid[16:],b['XHatArray'][ii,16:],markerMethod[1], label = 'Est', markersize=4)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_xlim([0, a['tEnd']]) 
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel=xLab[ii])
     
axs[nData-1].set(xlabel='time [h]') 
axs[nData-1].legend(loc='best') 

if savePlot is True:
    fig6.savefig("CSTREstStates.pdf", format="pdf", bbox_inches="tight")
    fig6.savefig('CSTREstStates.tif', format='tif', bbox_inches="tight", dpi=600)    
         