#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# March 2022; Last revision: 
# 
# Reads the results from the different DRTO implementations
#=======================================================

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

# import column model 
from ColumnModelRigorous import *

# Model Parameters
par = SystemParameters()

################
# LOADING DATA #
################
#%% loading previously built disturbances array
# ...\Disturbance Sequence\MAIN.py
# ORDER:
# timeArray, dist1, dist1Approx, dist2, dist2Approx
distArray = []
with (open('./disturbance_ramp.pkl', 'rb')) as openfile:
    while True:
        try:
            distArray.append(pickle.load(openfile))
        except EOFError:
            break

# simulation time (from disturbance file)
simTime = distArray[0].shape[0] # minutes
deltaPlot = 5*12 # plot one point every 5 minutes

# for plottimg
majorR = 60
minorR = 30

# Do you want to save the plots?
savePlot = True

# ORDER: 
# 0: MVArray
# 1: PArray
# 2: forecastArray
# 3: setpointArray
# 4: statesArray
# 5: measArray
# 6: inputArray
# 7: ubiasArray
# 8: ecoOFArray
# 9: timeArray
# 10: timeEst
# 11: timeOpt
# 12: statesHatArray
# 13: biasHatArray
# 14: ofEstArray
# 15: wHatArray
# 16: thetaHatArray
# 17: arrivalCostArray
# 18: PkSTD
# 19: PkFrob
# 20: PkCond
# 21: estTimeArray
# 22: estSolArray
# 23: pmHatTrajArray
# 24: xHatTrajArray  
# 25: statesOptkArray
# 26: uOptkArray
# 27: ofOptkArray
# 28: optTimeArray
# 29: optSolArray  
# 30: timeTrajArray
# 31: mvTrajArray
# 32: xTrajArray        
    
#%% loading Perfect DRTO data
data_1_DRTO = []
with (open('./Results_CLDRTO_bias03_ramp.pkl', 'rb')) as openfile:
    while True:
        try:
            data_1_DRTO.append(pickle.load(openfile))
        except EOFError:
            break

timeArray = data_1_DRTO[9]
timeDRTOArray = data_1_DRTO[11]
timeEstArray = data_1_DRTO[10]


#%% loading bias update
data_2_DRTO = []
with (open('./Results_CLDRTO_states_ramp.pkl', 'rb')) as openfile:
    while True:
        try:
            data_2_DRTO.append(pickle.load(openfile))
        except EOFError:
            break       

#%% loading state estimation
data_3_DRTO = []
with (open('./Results_CLDRTO_staPar_ramp.pkl', 'rb')) as openfile:
    while True:
        try:
            data_3_DRTO.append(pickle.load(openfile))
        except EOFError:
            break       
     
        
#%%
dataBundle = []
dataBundle.append(data_1_DRTO)
dataBundle.append(data_2_DRTO)
dataBundle.append(data_3_DRTO)

nData = 3

methodLabel = ["bias", "st", "st + par"]
colorMethod = ['b', 'g', 'r']
markerMethod = ['bo', 'gd', 'rv']

#%% MVS
fig1, ax = plt.subplots()

for ii in range(nData):
    ax.plot(timeArray[0:simTime:deltaPlot], dataBundle[ii][0][0,0:simTime:deltaPlot],colorMethod[ii], label =methodLabel[ii], linewidth=2)

ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax.legend(loc='best')

fig1.suptitle('Reflux Rate')
ax.set(xlabel='time [min]', ylabel='L [kmol/min]')  

##############
fig2, ax = plt.subplots()

for ii in range(nData):
    ax.plot(timeArray[0:simTime:deltaPlot], 
            dataBundle[ii][0][1,0:simTime:deltaPlot],colorMethod[ii],drawstyle='steps', label = methodLabel[ii], linewidth=2)

ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax.legend(loc='best')

fig2.suptitle('Boilup Rate')
ax.set(xlabel='time [min]', ylabel='V [kmol/min]')  

#ax.set_xlim([180, 420])
ax.set_ylim([2.6, 3.8]) 

# saving figure 
if savePlot is True:
    fig2.savefig("V_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig2.savefig('V_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

##############
fig3, ax = plt.subplots()

for ii in range(nData):
    ax.plot(timeArray[0:simTime:deltaPlot], dataBundle[ii][0][2,0:simTime:deltaPlot],colorMethod[ii], label =methodLabel[ii], linewidth=2)

ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax.legend(loc='best')

fig3.suptitle('Distillate Rate')
ax.set(xlabel='time [min]', ylabel='D [kmol/min]')  
 
##############
fig4, ax = plt.subplots()

for ii in range(nData):
    ax.plot(timeArray[0:simTime:deltaPlot], dataBundle[ii][0][3,0:simTime:deltaPlot],colorMethod[ii], label =methodLabel[ii], linewidth=2)
    
ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax.set_ylim([0.0, 1.0]) 

ax.legend(loc='best')

fig4.suptitle('Bottoms Rate')
ax.set(xlabel='time [min]', ylabel='B [kmol/min]')  

# saving figure 
if savePlot is True:
    fig4.savefig("B_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig4.savefig('B_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

# #%% Objective function
# # instantaneous
# fig5, axs = plt.subplots(nData, sharex=True)
# fig5.suptitle('Instantaneous Profit [$/min]')

# for ii in range(nData):
    
#     axs[ii].plot(timeArray[0:simTime:deltaPlot],dataBundle[ii][8][0:simTime:deltaPlot],colorMethod[ii],linewidth=2)
    
#     axs[ii].minorticks_on()
#     axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
#     axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
#     axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
#     axs[ii].set_ylim([0.0, 1.0]) 
#     axs[ii].grid(visible=True, which='major', linestyle='-')
#     axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
#     axs[ii].set(ylabel=methodLabel[ii])
     
# axs[nData-1].set(xlabel='t [min]') 

# # saving figure 
# if savePlot is True:
#     fig5.savefig("OFInst_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
#     fig5.savefig('OFInst_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

# # cummulative
# fig6, ax = plt.subplots()
# #fig6, axs = plt.subplots(3, sharex=True)
# fig6.suptitle('Cummulative Profit Difference [$]')

# for ii in range(1,nData):
#         ax.plot(timeArray[0:simTime:deltaPlot], np.cumsum(dataBundle[ii][8][0:simTime:deltaPlot] - dataBundle[0][8][0:simTime:deltaPlot], dtype=float)/12,colorMethod[ii], label = methodLabel[ii], linewidth=2)

# ax.xaxis.set_major_locator(MultipleLocator(majorR))
# ax.xaxis.set_minor_locator(MultipleLocator(minorR))
# ax.yaxis.set_tick_params(which='minor', bottom=False)
# ax.set_ylim([-0.1, 2.0])
# ax.grid(visible=True, which='major', linestyle='-')
# ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
# ax.set(xlabel='time [min]', ylabel='sum(Profit dif) [$]')  
# ax.legend(loc='best')

# # saving figure 
# if savePlot is True:
#     fig6.savefig("OF_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
#     fig6.savefig('OF_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

#%% Purity Constraint
fig7, axs = plt.subplots(3, sharex=True)
fig7.suptitle('Top Constraint')

for ii in range(nData): 
    axs[ii].plot(timeArray[0:simTime:deltaPlot],dataBundle[ii][4][par['NT'] - 1,0:simTime:deltaPlot],colorMethod[ii],linewidth=2)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_ylim([0.98, 1.00]) 
    axs[ii].grid(visible=True, which='major', linestyle='-')
    axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    #axs[ii].set(ylabel=methodLabel[ii])
    axs[ii].set(ylabel='$x_D$ [-]')
     
axs[nData-1].set(xlabel='time [min]') 

# saving figure 
if savePlot is True:
    fig7.savefig("xD_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig7.savefig('xD_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

fig8, axs = plt.subplots(3, sharex=True)
fig8.suptitle('Bottom product purity (unconstrained)')

for ii in range(nData): 
    axs[ii].plot(timeArray[0:simTime:deltaPlot],dataBundle[ii][4][0,0:simTime:deltaPlot],colorMethod[ii],linewidth=2)
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_ylim([0.005, 0.075]) 
    axs[ii].grid(visible=True, which='major', linestyle='-')
    axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    #axs[ii].set(ylabel=methodLabel[ii])
    axs[ii].set(ylabel='$x_B$ [-]')
    
axs[nData-1].set(xlabel='time [min]') 
 
if savePlot is True:
    fig8.savefig("xB_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig8.savefig('xB_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)


#%% DRTO Solution time
fig9, axs = plt.subplots(2, sharex=True)
fig9.suptitle('DRTO Solution time')

for ii in range(nData):
    axs[0].plot(timeDRTOArray, dataBundle[ii][28],markerMethod[ii], label =methodLabel[ii], markersize=4)

axs[0].set(ylabel='sol time [ms]') # [kmol/min]  

for ii in range(nData):
    axs[1].plot(timeDRTOArray, dataBundle[ii][29],markerMethod[ii], label =methodLabel[ii], markersize=4)

axs[1].set_yticks([0,1])
axs[1].set_yticklabels(['No','Yes'])
axs[1].set(ylabel='sol?') # [kmol/min] 

for ii in range(2):
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(visible=True, which='major', linestyle='-')
    axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
#%% Estimation Solution time
fig10, ax = plt.subplots()
fig10.suptitle('Estimation Solution time')

dataBundle[0][21] = np.hstack(dataBundle[0][21])
ax.plot(timeEstArray[1:],dataBundle[0][21][1,:],markerMethod[0], label =methodLabel[0], markersize=4)

for ii in range(1,nData):
    ax.plot(timeEstArray[1:],dataBundle[ii][21][0,:] + dataBundle[ii][21][1,:],markerMethod[ii], label =methodLabel[ii], markersize=4)

ax.set(xlabel='time [min]', ylabel='sol time [s]') # [kmol/min]  
ax.minorticks_on()
ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)   

ax.legend(loc='best')

# saving figure 
if savePlot is True:
    fig10.savefig("MHEtime_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig10.savefig('MHEtime_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)    


#np.mean(dataBundle[0][21][0,:] + dataBundle[0][21][1,:])
#np.mean(dataBundle[1][21][0,:] + dataBundle[1][21][1,:])
#np.mean(dataBundle[2][21][0,:] + dataBundle[2][21][1,:])

# # computing the model update layer execution time
# aveExecTime = []
# for ii in range(1,nData):
#       # computing the mean estimation time
#     aveExecTime.append(statistics.mean(dataBundle[ii][21][0,:])/statistics.mean(dataBundle[ii][28]))

# for ii in range(1,nData):
#     axs[1].plot(timeEstArray[1:], dataBundle[ii][22], markerMethod[ii], label =methodLabel[ii], markersize=4)

# axs[1].set_yticks([0,1])
# axs[1].set_yticklabels(['No','Yes'])
# axs[1].set(ylabel='sol?') # [kmol/min] 

# for ii in range(2):
 
    
#%% L/V ratio
fig11, ax = plt.subplots()
fig11.suptitle('L/V ratio')

for ii in range(nData): 
    ax.plot(timeArray[0:simTime:deltaPlot],dataBundle[ii][0][0,0:simTime:deltaPlot]/dataBundle[ii][0][1,0:simTime:deltaPlot],colorMethod[ii],label =methodLabel[ii],linewidth=2)

ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax.legend(loc='best')

ax.set(xlabel='time [min]', ylabel='L/V [-]')  

#%% Fixing OF
alphaR = np.array([1.5,0.2])
alphaC = np.array([0.03,0.02,0.05])
alph = np.concatenate((alphaR,-alphaC))

fig12, axs = plt.subplots(3, sharex=True)
fig12.suptitle('Instantaneous Profit')


for ii in range(nData): 
    tempOF = []
    for kk in range(len(timeArray)):
        data_p1 = alph[0]*dataBundle[ii][4][par['NT'] - 1,kk]*dataBundle[ii][0][2,kk]
        data_p2 = alph[1]*(1 - dataBundle[ii][4][0,kk])*dataBundle[ii][0][3,kk]
        data_c1 = -alph[2]*dataBundle[ii][0][1,kk]
        data_c2 = -alph[3]*(dataBundle[ii][0][0,kk] + dataBundle[ii][0][2,kk])
    
        if dataBundle[ii][4][par['NT'] - 1,kk] < 0.99-0.0013:  # 0.00065 | 0.0013:
            tempOF.append(data_p2 + data_c2 + data_c2)
        else:
            tempOF.append(data_p1 + data_p2 + data_c2 + data_c2)
        
    tempOF = np.hstack(tempOF)
    
    if ii == 0: tempOF_bias = tempOF
    if ii == 1: tempOF_sta = tempOF
    if ii == 2: tempOF_staPar = tempOF
    
    axs[ii].plot(timeArray[0:simTime:deltaPlot], tempOF[0:simTime:deltaPlot], colorMethod[ii], linewidth=2)    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].set_ylim([0.0, 1.5]) 
    axs[ii].grid(visible=True, which='major', linestyle='-')
    axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(ylabel='$\phi$ [\$/min]')
     
axs[nData-1].set(xlabel='time [min]') 

# saving figure 
if savePlot is True:
    fig12.savefig("OFInst_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig12.savefig('OFInst_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)
    
# cummulative
fig13, ax = plt.subplots()
#fig6, axs = plt.subplots(3, sharex=True)
fig13.suptitle('Cummulative Profit Difference [$]')

ax.plot(timeArray[0:simTime:deltaPlot], np.cumsum(tempOF_sta[0:simTime:deltaPlot] - tempOF_bias[0:simTime:deltaPlot], dtype=float)/12,colorMethod[1], label = methodLabel[1], linewidth=2)
ax.plot(timeArray[0:simTime:deltaPlot], np.cumsum(tempOF_staPar[0:simTime:deltaPlot] - tempOF_bias[0:simTime:deltaPlot], dtype=float)/12,colorMethod[2], label = methodLabel[2], linewidth=2)

ax.xaxis.set_major_locator(MultipleLocator(majorR))
ax.xaxis.set_minor_locator(MultipleLocator(minorR))
ax.yaxis.set_tick_params(which='minor', bottom=False)
ax.set_ylim([-0.1, 2.0])
ax.grid(visible=True, which='major', linestyle='-')
ax.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
ax.set(xlabel='time [min]', ylabel='$\sum \Phi_{diff} [\$]$')  
ax.legend(loc='best')

# saving figure 
if savePlot is True:
    fig13.savefig("OF_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig13.savefig('OF_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)    

#%% Bias Estimation (bias)
fig12, (ax1, ax2) = plt.subplots(2, sharex=True)
fig12.suptitle('Computed Bias')

ax1.plot(timeEstArray, dataBundle[0][13][10,:],'bx',markersize=3)
ax1.set(ylabel='$x_D$')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeEstArray, dataBundle[0][13][1,:],'bx',markersize=3)
ax2.set(xlabel='time [min]', ylabel='$x_B$')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(visible=True, which='major', linestyle='-')
ax2.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

# saving figure 
if savePlot is True:
    fig12.savefig("xBias_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig12.savefig('xBias_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

#%% State Estimation (states only)
fig13, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig13.suptitle('States: Estimated vs. true')

ax1.plot(timeArray[0:simTime:deltaPlot], dataBundle[1][4][34,0:simTime:deltaPlot],'k:', label ='True', linewidth=2)
ax1.plot(timeEstArray, dataBundle[1][12][34,:],'gx',markersize=3, label ='Est.')
ax1.set(ylabel='$x_{35}$')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray[0:simTime:deltaPlot], dataBundle[1][4][2,0:simTime:deltaPlot],'k:', label ='True', linewidth=2)
ax2.plot(timeEstArray, dataBundle[1][12][2,:],'gx',markersize=3,label ='Est.')
ax2.set(ylabel='$x_3$')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(visible=True, which='major', linestyle='-')
ax2.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax3.plot(timeArray[0:simTime:deltaPlot], dataBundle[1][4][77,0:simTime:deltaPlot],'k:', label ='True', linewidth=2)
ax3.plot(timeEstArray, dataBundle[1][12][77,:],'gx',markersize=3, label ='Est.')
ax3.set(xlabel='time [min]',ylabel='$M_{37}$')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(majorR))
ax3.xaxis.set_minor_locator(MultipleLocator(minorR))
ax3.yaxis.set_tick_params(which='minor', bottom=False)
ax3.grid(visible=True, which='major', linestyle='-')
ax3.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
ax3.legend(loc='best')

# saving figure 
if savePlot is True:
    fig13.savefig("xSta_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig13.savefig('xSta_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

#%% States and Parameter estimates
fig14, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(timeEstArray,dataBundle[2][16][0,:],'rx', markersize =2,label='Est.')
ax1.plot(timeArray[0:simTime:deltaPlot],distArray[1][0:simTime:deltaPlot],'k:', markersize=2,label='True', linewidth=2)

ax1.set(ylabel='$z_F$')  
ax1.set_title('Parameters: Estimated vs. true')
#ax1.legend(loc='best')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeEstArray,dataBundle[2][16][1,:],'rx', markersize=2,label='Est.')
ax2.plot(timeArray[0:simTime:deltaPlot],distArray[3][0:simTime:deltaPlot],'k:', markersize=2,label='True', linewidth=2)

ax2.set(ylabel=r'$\alpha$', xlabel='time [min]')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(visible=True, which='major', linestyle='-')
ax2.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

# saving figure 
if savePlot is True:
    fig14.savefig("thetaStaPar_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig14.savefig('thetaStaPar_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

fig15, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig15.suptitle('States: Estimated vs. true')

ax1.plot(timeEstArray, dataBundle[2][12][34,:],'rx',markersize=3, label ='Est.')
ax1.plot(timeArray[0:simTime:deltaPlot], dataBundle[2][4][34,0:simTime:deltaPlot],'k:', label ='True', linewidth=2)

ax1.set(ylabel='$x_{35}$')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(visible=True, which='major', linestyle='-')
ax1.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeEstArray, dataBundle[2][12][2,:],'rx',markersize=3,label ='Est.')
ax2.plot(timeArray[0:simTime:deltaPlot], dataBundle[2][4][2,0:simTime:deltaPlot],'k:', label ='True', linewidth=2)

ax2.set(ylabel='$x_3$')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(visible=True, which='major', linestyle='-')
ax2.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax3.plot(timeArray[0:simTime:deltaPlot], dataBundle[2][4][77,0:simTime:deltaPlot],'k:', label ='True', linewidth=2)
ax3.plot(timeEstArray, dataBundle[2][12][77,:],'rx',markersize=3, label ='Est.')
ax3.set(xlabel='time [min]',ylabel='$M_{37}$')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(majorR))
ax3.xaxis.set_minor_locator(MultipleLocator(minorR))
ax3.yaxis.set_tick_params(which='minor', bottom=False)
ax3.grid(visible=True, which='major', linestyle='-')
ax3.grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
ax3.legend(loc='best')

# saving figure 
if savePlot is True:
    fig15.savefig("xStaPar_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig15.savefig('xStaPar_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)


#%% Fraction and Molar holdup profiles along the column at a given time step
fig16, axs = plt.subplots(1,3, sharey=True)
fig16.suptitle('Fraction of A profile')
# Set the ticks and ticklabels for all axes 
plt.setp(axs, yticks=range(1,par['NT']+1,5))

columnTray = range(1,par['NT']+1)

timePlot =30*12 

for ii in range(nData): 
    axs[ii].plot(dataBundle[ii][12][0:par['NT'],int(timePlot/12)],columnTray,colorMethod[ii],linewidth=2)
    axs[ii].plot(dataBundle[ii][4][0:par['NT'],timePlot],columnTray,'k:',linewidth=2)

    axs[ii].minorticks_on()
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(visible=True, which='major', linestyle='-')
    axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[ii].set(title=methodLabel[ii])
    axs[ii].set(xlabel='$x_A$ [-]')
     
axs[0].set(ylabel='# tray') 

# saving figure 
if savePlot is True:
    fig16.savefig("x_profile_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig16.savefig('x_profile_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)

fig17, axs = plt.subplots(1,3, sharey=True)
fig17.suptitle('Molar holdup profile')
# Set the ticks and ticklabels for all axes 
plt.setp(axs, yticks=range(1,par['NT']+1,5),xticks=[0.4995, 0.501])

for ii in range(nData): 
    axs[ii].plot(dataBundle[ii][12][par['NT']:-1,int(timePlot/12)],columnTray[:],colorMethod[ii],linewidth=2)
    axs[ii].plot(dataBundle[ii][4][par['NT']:-1,timePlot],columnTray[:],'k:',linewidth=2)

    axs[ii].minorticks_on()
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(visible=True, which='major', linestyle='-')
    axs[ii].grid(visible=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    axs[ii].set_xlim([0.499, 0.502])
    
    axs[ii].set(title=methodLabel[ii])
    axs[ii].set(xlabel='$M$ [kmol]')
     
axs[0].set(ylabel='# Tray') 
 
if savePlot is True:
    fig17.savefig("m_profile_rampDisturbances.pdf", format="pdf", bbox_inches="tight")
    fig17.savefig('m_profile_rampDisturbances.tif', format='tif', bbox_inches="tight", dpi=600)
