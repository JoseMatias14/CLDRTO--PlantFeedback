#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# April 2022; Last revision: 15-06-2022
# 
# Generating data for control tuning
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
Integ, Fx, Fp, xdot, xvar, pvar = CreateColumnModel()

# Initial condition
dx0, y0, u0, p0 = InitialCondition()

# Model Parameters
par = SystemParameters()

# Control parameters
ctrlPar = ControlTuning()

# revenue (price for top and bottom streams + purity)
alphaR = np.array([1.5,0.2]) # 1.5 | 1.2 | 0.8 | 0.6
 
# costs associated with 
# boil up vaporization (steam to reboiler)
# top vapor condensation (cooling water to condenser)
# feed costs (no purity associated)
alphaC = np.array([0.03,0.02,0.05])
alphV = np.concatenate((alphaR,-alphaC))
alph = np.expand_dims(alphV, axis = 0)

#%% STARTING THE SIMULATION 
# simulation time (from disturbance file)
simTime = 150*12  # minutes for reflux 100*12 | minutes for levels 20*12

# Preparing time array
timeArray = np.linspace(0,simTime*par['T'],simTime)

# Manipulated variable array
MVArray = u0*np.ones([u0.size, simTime])

#%% DISTURBANCE 2 - pressure (rel. volality) disturbance
# 1% decrease in nominal value (N.B. alpha = 1.5 is the nominal value)
#Period 1: high
period1 = 1.0*np.ones((300*12,)) # *12 because plant sampling time is 5 sec and variables are in minutes

#Period 3: low
period2 = 0.99*np.ones((240*12,))
       
# converting to numpy array
dist2 = p0*np.concatenate((period1,period2))

#%% CREATING VARIABLES FOR SAVING SIMULATION INFO                 
#########
# Plant #
#########
statesArray = []
ofArray = []

#%% INITIALIZING SIMULATION
#########
# Plant #
#########
# states
xk = dx0
statesArray.append(xk)
temp = np.array(
    [xk[par['NT'] - 1]*MVArray[2,0],
     (1 - xk[0])*MVArray[3,0],
    [MVArray[1,0]],
     [MVArray[0,0] + MVArray[2,0]],
     [MVArray[4,0]]
     ])

ofArray.append(np.dot(alph,temp))

# PI-controller
# initializing bias (error term)
ubarB = ctrlPar['Bs']
ubarD = ctrlPar['Ds']
ubarL = ctrlPar['Ls']

#%% SIMULATION (loop in sampling intervals)
for k in range(1,simTime): 
    print('Simulation time >> ',"{:.3f}".format(k*par['T']), '[min]')
      
    ####################
    # Simulating Plant #
    ####################
    # updating inputs using true values of zf and alpha
    pk = vertcat(MVArray[:,k-1],dist2[k],par['T'])
    
    # Evolving plant in time
    Ik = Integ(x0=xk,p=pk)
    xk = Ik['xf']

    ############################  
    # Saving Plant Information #
    ############################
    # true states
    statesArray.append(np.array(xk))
    
    temp1 = np.array(xk)
    temp = np.array(
        [temp1[par['NT'] - 1]*MVArray[2,k],
         (1 - temp1[0])*MVArray[3,k],
        [MVArray[1,k]],
         [MVArray[0,k] + MVArray[2,k]],
         [MVArray[4,k]]
         ])

    ofArray.append(np.dot(alph,temp))

    #########################################################
    # PI-Controllers - reboiler, condenser levels and comp. #
    #########################################################
    ## REBOILER ##
    # Actual reboiler holdup
    MB = xk[par['NT']].full()
    # computing error
    eB = MB - ctrlPar['MBs']
    # adjusting bias (accounting for integral action)
    ubarB = ubarB + ctrlPar['KcB']/ctrlPar['tauB']*eB*par['T']
    # Bottoms flor
    MVArray[3,k] = ubarB + ctrlPar['KcB']*eB
     
    ## CONDENSER ##
    # Actual condenser holdup
    MD = xk[2*par['NT'] - 1].full() 
    # computing error
    eD = MD - ctrlPar['MDs']
    # adjusting bias (accounting for integral action)
    ubarD = ubarD + ctrlPar['KcD']/ctrlPar['tauD']*eD*par['T']
    # Distillate flow
    MVArray[2,k] = ubarD + ctrlPar['KcD']*eD           
     
    ## TOP COMP. ##
    # Actual top composition
    xD = xk[2*par['NT']+1].full() # par['NT'] - 1 | 2*par['NT']+1
    # computing error
    eL = xD - ctrlPar['xDs']
    # adjusting bias (accounting for integral action)
    ubarL = ubarL + ctrlPar['KcL']/ctrlPar['tauL']*eL*par['T']
    # Distillate flow
    MVArray[0,k] = ubarL + ctrlPar['KcL']*eL    
    
    
#%%   
######################    
# Preparing for plot #
######################
# System
statesArray = np.hstack(statesArray)
ofArray = np.hstack(ofArray)

# #%%
# ########################
# # For saving variables #
# ########################
# np.savetxt('data_L_xD.csv', [p for p in zip(MVArray.T[:,0], statesArray.T[:,par['NT'] - 1], statesArray.T[:,2*par['NT']], statesArray.T[:,2*par['NT'] + 1])], delimiter=',')
# # np.savetxt('data_D_MD.csv', [p for p in zip(MVArray.T[:,2],statesArray.T[:,2*par['NT'] - 1])], delimiter=',')
# # np.savetxt('data_B_MB.csv', [p for p in zip(MVArray.T[:,3],statesArray.T[:,par['NT']])], delimiter=',')

# # open the file in the write mode
# f = open('C:/Users/MACC-Jose/Documents/GitHub/DRTO Model Update/Paper/Time Delay/ControlData.csv', 'w')

# # create the csv writer
# writer = csv.writer(f)

# # write a row to the csv file
# writer.writerow(statesArray)

# # close the file
# f.close()

#%% 
########
# Plot #
########
majorR = 50
minorR = 10


#%% 3. Top concentration
fig3, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig3.suptitle('Top/bottom purities')

ax1.plot(timeArray, statesArray.T[:,par['NT'] - 1],'b', linewidth=2, label ='true')
#ax1.plot(timeArray, statesArray.T[:,2*par['NT']+1],'b', linewidth=4, label ='delayed')
#  ax1.set_ylim([0.988, 1.0])
ax1.set(ylabel='$x_D$ [kmol]')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
#ax1.legend(loc='best')

ax2.plot(timeArray, statesArray.T[:,0],'b', linewidth=2)
#ax2.set_ylim([2.5, 2.8])
ax2.set(ylabel='$x_B$ [kmol]')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(visible=True, which='major', linestyle='-')
ax2.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax3.plot(timeArray, ofArray.T,'r', linewidth=2)
#ax2.set_ylim([2.5, 2.8])
ax3.set(ylabel='$\phi$ [\$]',xlabel='t [min]')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(majorR))
ax3.xaxis.set_minor_locator(MultipleLocator(minorR))
ax3.yaxis.set_tick_params(which='minor', bottom=False)
ax3.grid(visible=True, which='major', linestyle='-')
ax3.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
