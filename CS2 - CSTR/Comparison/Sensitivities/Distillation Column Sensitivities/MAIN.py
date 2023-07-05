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
from matplotlib import cm 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap 
from matplotlib.collections import PolyCollection

# import column model 
from ColumnModelRigorous import *

# saving data in a csv file
import csv


#%% BUILDING FUNCTIONS 
# building open-loop dynamic column model 
# used as plant, for estimation [EKF + MHE], and for OL-DRTO
Integ, Fx_d, Fp_d, xdot_d, xvar_d, pvar_d = CreateColumnModel()
Integ_d, Fx, Fp, xdot, xvar, pvar = CreateColumnModelClosedLoop()

# Initial condition
dx0, y0, u0, p0 = InitialCondition()

# Initial condition
dx0_cl, y0_cl, u0_cl, p0_cl = InitialConditionCL()

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
simTime = 150*12  # minutes for reflux 100*12 | minutes for levels 20*12

# Preparing time array
timeArray = np.linspace(0,simTime*par['T'],simTime)

# #%% CREATING VARIABLES FOR SAVING SIMULATION INFO                 
# statesSensArray = []
# # sensitivities
# statesSensArray.append(vertcat(0,0,0,0))


# #%% SIMULATION (loop in sampling intervals)
# for k in range(1,simTime): 
#     print('Simulation time >> ',"{:.3f}".format(k*par['T']), '[min]')
     
#     # Simulating 
#     # updating inputs using true values of zf and alpha
#     pk_cl = vertcat(u0_cl,p0_cl,ubiask,k*par['T'])
    
#     # Sensitivities 
#     Fk = Fp(x0=dx0_cl, p=pk_cl)['jac_xf_p'].full()
   
#     ############################  
#     # Saving  Information #
#     ############################
#     # sensitivities
#     sx_p = vertcat(Fk[par['NT'] - 1,5],Fk[par['NT'] - 1,7],Fk[0,5],Fk[0,7])
#     statesSensArray.append(sx_p)
    

# #%%   
# ######################    
# # Preparing for plot #
# ######################
# # System
# statesSensArray = np.hstack(statesSensArray)

# #%% 
# ########
# # Plot #
# ########
# majorR = 60
# minorR = 30

# #%% 1. Reboiler level
# fig1, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig1.suptitle('Sensitivities Top Fraction')

# ax1.plot(timeArray, statesSensArray.T[:,0],'b', linewidth=4)
# ax1.set(ylabel='d $x_D$/d $z_F$')
# ax1.minorticks_on()
# #ax1.set_ylim([0.0025, 0.03])
# ax1.xaxis.set_major_locator(MultipleLocator(majorR))
# ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
# ax1.yaxis.set_tick_params(which='minor', bottom=False)
# ax1.grid(b=True, which='major', linestyle='-')
# ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

# ax2.plot(timeArray, statesSensArray.T[:,1],'b', linewidth=4)
# ax2.set(ylabel=r'd $x_D$/d $\alpha$', xlabel='time [min]')
# ax2.minorticks_on()
# #ax2.set_ylim([0.0025, 0.03])
# ax2.xaxis.set_major_locator(MultipleLocator(majorR))
# ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
# ax2.yaxis.set_tick_params(which='minor', bottom=False)
# ax2.grid(b=True, which='major', linestyle='-')
# ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

# fig1.savefig("sensXD.pdf", format="pdf", bbox_inches="tight")
# fig1.savefig('SensXD.tif', format='tif', bbox_inches="tight", dpi=600)

# #%% 2. Condenser level
# fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
# fig2.suptitle('Sensitivities Bottom Fraction')

# ax1.plot(timeArray, statesSensArray.T[:,2],'b', linewidth=4)
# ax1.set(ylabel=r'd $x_B$/d $z_F$')
# ax1.minorticks_on()
# #ax1.set_ylim([-0.01, -0.65])
# ax1.xaxis.set_major_locator(MultipleLocator(majorR))
# ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
# ax1.yaxis.set_tick_params(which='minor', bottom=False)
# ax1.grid(b=True, which='major', linestyle='-')
# ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

# ax2.plot(timeArray, statesSensArray.T[:,3],'b', linewidth=4)
# ax2.set(ylabel=r'd $x_B$/d $\alpha$', xlabel='time [min]')
# ax2.minorticks_on()
# #ax2.set_ylim([-0.01, 0.01])
# ax2.xaxis.set_major_locator(MultipleLocator(majorR))
# ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
# ax2.yaxis.set_tick_params(which='minor', bottom=False)
# ax2.grid(b=True, which='major', linestyle='-')
# ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)


# fig2.savefig("sensXB.pdf", format="pdf", bbox_inches="tight")
# fig2.savefig('SensXB.tif', format='tif', bbox_inches="tight", dpi=600)

#%% Checking model linearity

# number of discretization points in the expected parameter range variation
multLin = 10 # 10

# obtaining parameter grid - z / alpha
thetaLin = [np.linspace(0.40,0.50,multLin),np.linspace(1.485,1.50,multLin)]

# preparing for saving data. One variable per state
XLin_xB = np.zeros((2, multLin,simTime))
XLin_xD = np.zeros((2, multLin,simTime))


# loop to calculate the state profiles for each of the possible parameter
# values. First for zf, then for alpha

for pp in range(2): #parameters
    for ll in range(multLin): # parameter range
        
        # initialize vectors
        uk = u0
        xk = dx0
        ppk = p0
            
        # replace entry with value being evaluated
        if pp == 0:
            uk[5] = thetaLin[pp][ll]
        
        if pp == 1:
            ppk = thetaLin[pp][ll]
        
        # PI-controller
        # initializing bias (error term)
        ubarB = ctrlPar['Bs']
        ubarD = ctrlPar['Ds']
        ubarL = ctrlPar['Ls']

        # simulating in time
        for kk in range(1,simTime): 
            print('Simulation (lin) time >> ',"{:.3f}".format(kk*par['T']), '[min]')
     
            ##############
            # Simulating #
            ##############
            # updating inputs using true values of zf and alpha
            pk = vertcat(uk,ppk,par['T'])
                
            # Evolving in time
            Ik = Integ(x0=xk,p=pk)
            xk = Ik['xf']

            ######################  
            # Saving Information #
            ######################
            XLin_xD[pp,ll,kk] = xk[par['NT'] - 1].full()
            XLin_xB[pp,ll,kk] = xk[0].full()
                
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
            uk[3] = ubarB + ctrlPar['KcB']*eB
                 
            ## CONDENSER ##
            # Actual condenser holdup
            MD = xk[2*par['NT'] - 1].full() 
            # computing error
            eD = MD - ctrlPar['MDs']
            # adjusting bias (accounting for integral action)
            ubarD = ubarD + ctrlPar['KcD']/ctrlPar['tauD']*eD*par['T']
            # Distillate flow
            uk[2] = ubarD + ctrlPar['KcD']*eD           
                 
            ## TOP COMP. ##
            # Actual top composition
            xD = xk[2*par['NT']].full() # par['NT'] - 1 | 2*par['NT']+1
            # computing error
            eL = xD - ctrlPar['xDs']
            # adjusting bias (accounting for integral action)
            ubarL = ubarL + ctrlPar['KcL']/ctrlPar['tauL']*eL*par['T']
            # Distillate flow
            uk[0] = ubarL + ctrlPar['KcL']*eL    


#%% plotting data

# Slices
ppLabel = ['$z_f$ [-]',r'$\alpha$ [-]']

# color
plasma=cm.get_cmap('plasma', 5*(simTime+1))

# for Bottom Fraction
fig3, axs = plt.subplots(1,2, sharey=True)

for pp in range(2): # parameters
    axs[pp].plot(thetaLin[pp],XLin_xD[pp,:,10*12],color=plasma.colors[5*(10*12)],linewidth=3,label='t = 10min')
    axs[pp].plot(thetaLin[pp],XLin_xD[pp,:,75*12],color=plasma.colors[5*(75*12)],linewidth=3,label='t = 75min') 
    axs[pp].plot(thetaLin[pp],XLin_xD[pp,:,150*12 - 1],color=plasma.colors[5*(150*12)],linewidth=3,label='t = 150min') 
    axs[pp].grid()
    axs[pp].set(xlabel=ppLabel[pp])
    

axs[0].set(ylabel='$x_D$ [-]')
axs[1].legend(loc='best')

fig3.savefig("sens2_XD.pdf", format="pdf", bbox_inches="tight")
fig3.savefig('sens2_XD.tif', format='tif', bbox_inches="tight", dpi=600)

# for Bottom Fraction
fig4, axs = plt.subplots(1,2, sharey=True)

for pp in range(2): # parameters
    axs[pp].plot(thetaLin[pp],XLin_xB[pp,:,10*12],color=plasma.colors[5*(10*12)],linewidth=3,label='t = 10min')
    axs[pp].plot(thetaLin[pp],XLin_xB[pp,:,75*12],color=plasma.colors[5*(75*12)],linewidth=3,label='t = 50min') 
    axs[pp].plot(thetaLin[pp],XLin_xB[pp,:,150*12 - 1],color=plasma.colors[5*(150*12)],linewidth=3,label='t = 150min') 
    axs[pp].grid()
    axs[pp].set(xlabel=ppLabel[pp])
    

axs[0].set(ylabel='$x_B$ [-]')
axs[1].legend(loc='best')

fig4.savefig("sens2_XB.pdf", format="pdf", bbox_inches="tight")
fig4.savefig('sens2_XB.tif', format='tif', bbox_inches="tight", dpi=600)

# 3D
#%%
ax = plt.figure().add_subplot(projection='3d')

##plots
for kk in range(1,simTime,10*12):
    times = kk*np.ones(multLin)
    ax.plot3D(times, thetaLin[0], XLin_xD[0,:,kk],linewidth=3,color=plasma.colors[5*kk])
 
ax.set(xlabel='tim [min]',ylabel='$z_f$',zlabel='$x_D$')

mystring = []

for digit in range(0,160,25): 
    mystring.append(str(digit))

plt.xticks(np.arange(0.0, 150.0*12.0 + 1, step=25.0*12.0),labels=mystring)

plt.savefig("sens3_zf_XD.pdf", format="pdf", bbox_inches="tight")
plt.savefig('sens3_zf_XD.tif', format='tif', bbox_inches="tight", dpi=600)

#%%
ax = plt.figure().add_subplot(projection='3d')


##plots
for kk in range(1,simTime,10*12):
    times = kk*np.ones(multLin)
    ax.plot3D(times, thetaLin[1], XLin_xD[1,:,kk],linewidth=3,color=plasma.colors[5*kk])
 
ax.set(xlabel='tim [min]',ylabel=r'$\alpha$',zlabel='$x_D$')

mystring = []

for digit in range(0,180,30): 
    mystring.append(str(digit))

plt.xticks(np.arange(0.0, 150.0*12.0 + 1, step=30.0*12.0),labels=mystring)

plt.savefig("sens3_alpha_XD.pdf", format="pdf", bbox_inches="tight")
plt.savefig('sens3_alpha_XD.tif', format='tif', bbox_inches="tight", dpi=600)