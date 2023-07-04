#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# February 2022; Last revision: 15-06-2022 
# 
# Creating disturbance for the column case study
# CASE 2: Ramp disturbance
#=======================================================

###################
# IMPORTING STUFF #
###################
# library that helps you with matrices "numpy":
import numpy as np

# for plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# for saving variables 
import pickle

# import column model 
from InitialConditionAndParameters import *

# Import math Library
import math

############################
# SIMULATION CONFIGURATION #
############################
# Initial condition
dx0, y0, u0, p0 = InitialCondition()

# Model Parameters
par = SystemParameters()

# Saving figures?
savePlot = True

################################################
# Creating disturbance - deterministic profile #
################################################  
#%% DISTURBANCE 1 - feed disturbance
# 20% step decrease in nominal value (N.B. zf = 0.5 is the nominal value)
#Period 1: high
period1 = 1.0*np.ones((120*12,)) # *12 because plant sampling time is 5 sec and variables are in minutes

#Period 2: ramp from 120 and 240 the ramp
period2 = []
for ii in range(60*12 + 1): period2.append(1.0 - ii*(1.0 - 0.8)/(60*12))
period2 = np.hstack(period2)

#Period 3: low
period3 = 0.8*np.ones(((360)*12,))

dist1 = 0.5*np.concatenate((period1,period2,period3))

# simulation time 
simTime = dist1.shape[0] # minutes 

# value that is going to be used by the methods that do not update the parameters is the mean
dist1Approx = 0.9*0.5*np.ones((len(dist1),))

#%% DISTURBANCE 2 - pressure (rel. volality) disturbance
# 1% decrease in nominal value (N.B. alpha = 1.5 is the nominal value)
#Period 1: high
period1 = 1.0*np.ones((300*12,)) # *12 because plant sampling time is 5 sec and variables are in minutes

#Period 2: ramp from 120 and 240 the ramp
period2 = []
for ii in range(120*12 + 1): period2.append(1.0 - ii*(1.0 - 0.99)/(120*12))
period2 = np.hstack(period2)

#Period 3: low
period3 = 0.99*np.ones((120*12,))
       
# converting to numpy array
dist2 = p0*np.concatenate((period1,period2,period3))

# value that is going to be used by the methods that do not update the parameters is the nominal
dist2Approx = p0*np.ones((len(dist1),))

#%% NOISE
#####################################
# Pre-computing random noise vector #
#####################################
# NoNoise scale = 0.000 
# LowNoise scale = 0.001 
# NoNoise scale = 0.005 

# fraction states - sensor noise

# x1, x4, x6, x11, x16, x21, x26, x31, x36, x39, x84 (x41 + FO 'delay')
noise1 = np.random.normal(loc=0.0, scale=0.001, size=(11,dist1.shape[0]))

# holdup states - sensor noise
# M1, M11, M21, M31, M41 
noise2 = np.random.normal(loc=0.0, scale=0.001, size=(5,dist1.shape[0]))

# concatenating both
noise = np.vstack((noise1,noise2))
 
#%% Creating time array
# When I created the disturbance, the Dt = 1s, now I adapt the time to the simulation sampling time (i.e. Dt = 5s)
timeArray = np.linspace(0,simTime*par['T'],simTime)
# converting list in numpy array
timeArray = np.array(timeArray)

#%% Plot disturbances
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
majorR = 60
minorR = 30

# fig2 = plt.figure(2)
ax1.plot(timeArray,dist1Approx,'c:', linewidth=4)
ax1.plot(timeArray,dist1,'k', linewidth=4)

ax1.set_ylabel('$z_F \ [-]$', fontsize=12)
ax1.set_title('Column feed (light component fraction) disturbance', fontsize=12)
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray,dist2Approx,'c:', linewidth=4)
ax2.plot(timeArray,dist2,'k', linewidth=4)

ax2.set_ylabel(r'$\alpha \ [-]$', fontsize=12)
ax2.set_xlabel('time [min]', fontsize=12)
ax2.set_title('Rel. Volatility disturbance', fontsize=12)
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)


# saving figure 
if savePlot is True:
    fig.savefig("rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig.savefig('rampDisturbances.tif', format='tif', dpi=600)

    
#%% Plot noise
fig2, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(timeArray,y0[1] + noise1[1,:], markersize=2)
ax1.set(ylabel='x [-]')  
ax1.set_title('Measured fraction states')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray,y0[5 + 0] + noise2[0,:], markersize=2)
ax2.set(xlabel='t [min]',ylabel='M [-]')     
ax2.set_title('Measured holdups')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

#%% Open a file and use dump()
with open('disturbance_ramp.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(timeArray, file)
    pickle.dump(dist1, file)
    pickle.dump(dist1Approx,file)
    pickle.dump(dist2,file)
    pickle.dump(dist2Approx,file)
    pickle.dump(noise,file)