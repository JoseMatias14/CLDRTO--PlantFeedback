#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# April 2022; Last revision: 15-06-2022
# 
# Generating data for sensitivity analysis
#=======================================================

#%% IMPORTING PACKAGES AND MODULES + LOADING FILES
# importing  all the
# functions defined in casadi.py
from casadi import *

# library that helps you with matrices "numpy":
import numpy as np

# for plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# import column model 
from ColumnModelRigorous import *

# saving data in a csv file
import csv


#%% BUILDING FUNCTIONS 
# building open-loop dynamic column model 
# used as plant, for estimation [EKF + MHE], and for OL-DRTO
Integ, Fx, Fp, xdot, xvar, pvar = CreateColumnModelClosedLoop()

# Initial condition
dx0, y0, u0, p0 = InitialConditionCL()

# Model Parameters
par = SystemParameters()

# Control parameters
ctrlPar = ControlTuning()

# Controllers
# initializing bias (error term)
ubarB = ctrlPar['Bs']
ubarD = ctrlPar['Ds']
ubarL = ctrlPar['Ls']
ubiask = np.vstack((ubarB,ubarD,ubarL))

# simulation time (from disturbance file)
simTime = 540*12  # minutes for reflux 100*12 | minutes for levels 20*12

# Preparing time array
timeArray = np.linspace(0,simTime*par['T'],simTime)

#%% CREATING VARIABLES FOR SAVING SIMULATION INFO                 
statesSensArray = []
# sensitivities
statesSensArray.append(vertcat(0,0,0,0))


#%% SIMULATION (loop in sampling intervals)
for k in range(1,simTime): 
    print('Simulation time >> ',"{:.3f}".format(k*par['T']), '[min]')
     
    # Simulating 
    # updating inputs using true values of zf and alpha
    pk = vertcat(u0,p0,ubiask,k*par['T'])
    
    # Sensitivities 
    Fk = Fp(x0=dx0, p=pk)['jac_xf_p'].full()
   
    ############################  
    # Saving  Information #
    ############################
    # sensitivities
    sx_p = vertcat(Fk[par['NT'] - 1,5],Fk[par['NT'] - 1,7],Fk[0,5],Fk[0,7])
    statesSensArray.append(sx_p)
    

#%%   
######################    
# Preparing for plot #
######################
# System
statesSensArray = np.hstack(statesSensArray)

#%% 
########
# Plot #
########
majorR = 60
minorR = 30

#%% 1. Reboiler level
fig1, (ax1, ax2) = plt.subplots(2, sharex=True)
fig1.suptitle('Sensitivities Top Fraction')

ax1.plot(timeArray, statesSensArray.T[:,0],'b', linewidth=4)
ax1.set(ylabel='d $x_D$/d $z_F$')
ax1.minorticks_on()
#ax1.set_ylim([0.0025, 0.03])
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray, statesSensArray.T[:,1],'b', linewidth=4)
ax2.set(ylabel=r'd $x_D$/d $\alpha$', xlabel='time [min]')
ax2.minorticks_on()
#ax2.set_ylim([0.0025, 0.03])
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

fig1.savefig("sensXD.pdf", format="pdf", bbox_inches="tight")
fig1.savefig('SensXD.tif', format='tif', bbox_inches="tight", dpi=600)

#%% 2. Condenser level
fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
fig2.suptitle('Sensitivities Bottom Fraction')

ax1.plot(timeArray, statesSensArray.T[:,2],'b', linewidth=4)
ax1.set(ylabel=r'd $x_B$/d $z_F$')
ax1.minorticks_on()
#ax1.set_ylim([-0.01, -0.65])
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray, statesSensArray.T[:,3],'b', linewidth=4)
ax2.set(ylabel=r'd $x_B$/d $\alpha$', xlabel='time [min]')
ax2.minorticks_on()
#ax2.set_ylim([-0.01, 0.01])
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)


fig2.savefig("sensXB.pdf", format="pdf", bbox_inches="tight")
fig2.savefig('SensXB.tif', format='tif', bbox_inches="tight", dpi=600)