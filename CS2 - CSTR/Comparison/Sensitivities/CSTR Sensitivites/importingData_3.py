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
from matplotlib import cm 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap 
from matplotlib.collections import PolyCollection


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
ax2.set(ylabel=r'd $C_A$/d $\Psi$', xlabel='time [h]')
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
ax2.set(ylabel=r'd $T$/d $\Psi$', xlabel='time [h]')
ax2.minorticks_on()
#ax2.set_ylim([-0.01, 0.01])
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

fig2.savefig("sensT.pdf", format="pdf", bbox_inches="tight")
fig2.savefig('sensT.tif', format='tif', bbox_inches="tight", dpi=600)
    

#%% PARAMETER DISTRIBUTION
# Slices
ppLabel = ['$\Delta U$ [1e3 kcal/kmol]','$\Psi$ [-]']

#color scheme
plasma=cm.get_cmap('plasma', 50*(int(a['N'])+1))


# for Bottom Fraction
fig3, axs = plt.subplots(1,2, sharey=True)

for pp in range(2): # parameters
    axs[pp].plot(a['thetaLin'][pp,:],a['XLin_CA'][pp,:,1],color=plasma.colors[50*1],linewidth=3,label='t = 2h')
    axs[pp].plot(a['thetaLin'][pp,:],a['XLin_CA'][pp,:,5],color=plasma.colors[50*10],linewidth=3,label='t = 10h') 
    axs[pp].plot(a['thetaLin'][pp,:],a['XLin_CA'][pp,:,10 - 1],color=plasma.colors[50*20],linewidth=3,label='t = 20h') 
    axs[pp].grid()
    axs[pp].set(xlabel=ppLabel[pp])
    

axs[0].set(ylabel='$x_D$ [-]')
axs[1].legend(loc='best')

fig3.savefig("sensCA_2.pdf", format="pdf", bbox_inches="tight")
fig3.savefig('sensCA_2.tif', format='tif', bbox_inches="tight", dpi=600)

# for Bottom Fraction
fig4, axs = plt.subplots(1,2, sharey=True)

for pp in range(2): # parameters
    axs[pp].plot(a['thetaLin'][pp,:],a['XLin_T'][pp,:,1],color=plasma.colors[50*1],linewidth=3,label='t = 2h')
    axs[pp].plot(a['thetaLin'][pp,:],a['XLin_T'][pp,:,5],color=plasma.colors[50*10],linewidth=3,label='t = 10h') 
    axs[pp].plot(a['thetaLin'][pp,:],a['XLin_T'][pp,:,10 - 1],color=plasma.colors[50*20],linewidth=3,label='t = 20h') 
    axs[pp].grid()
    axs[pp].set(xlabel=ppLabel[pp])
    

axs[0].set(ylabel='$x_B$ [-]')
axs[1].legend(loc='best')

fig4.savefig("sensT_2.pdf", format="pdf", bbox_inches="tight")
fig4.savefig('sensT_2.tif', format='tif', bbox_inches="tight", dpi=600)

# 3D
#%%
ax = plt.figure().add_subplot(projection='3d')


##plots
for kk in range(1,int(a['N']),2):
    times = kk*np.ones(int(a['multLin']))
    ax.plot3D(times, a['thetaLin'][0,:], a['XLin_T'][0,:,kk],linewidth=3,color=plasma.colors[50*kk])
 
ax.set(xlabel='time [h]',ylabel='$\Delta U$ [1e3 kcal/kmol]',zlabel='$T$ [K]')

mystring = []

for digit in range(0,21,4): 
    mystring.append(str(digit))

plt.xticks(np.arange(0.0, 20.0 + 1.0, step=4.0),labels=mystring)

plt.savefig("sensT_DU_3.pdf", format="pdf", bbox_inches="tight")
plt.savefig('sensT_DU_3.tif', format='tif', bbox_inches="tight", dpi=600)

#%%
ax = plt.figure().add_subplot(projection='3d')

##plots
for kk in range(1,int(a['N']),2):
    times = kk*np.ones(int(a['multLin']))
    ax.plot3D(times, a['thetaLin'][1], a['XLin_T'][1,:,kk],linewidth=3,color=plasma.colors[50*kk])
 
ax.set(xlabel='time [h]',ylabel='$\Psi$ [-]',zlabel='$T$ [K]')

plt.xticks(np.arange(0.0, 20.0 + 1.0, step=4.0),labels=mystring)

plt.savefig("sensT_psi_3.pdf", format="pdf", bbox_inches="tight")
plt.savefig('sensT_psi_3.tif', format='tif', bbox_inches="tight", dpi=600)

#%%
ax = plt.figure().add_subplot(projection='3d')

#plasma=cm.get_cmap('plasma', 5*(int(a['N'])+1))

##plots
for kk in range(1,int(a['N']),2):
    times = kk*np.ones(int(a['multLin']))
    ax.plot3D(times, a['thetaLin'][0,:], a['XLin_CA'][0,:,kk],linewidth=3,color=plasma.colors[50*kk])
 
ax.set(xlabel='time [h]',ylabel='$\Delta U$ [1e3 kcal/kmol]',zlabel='$C_A$ [kmol]')

plt.xticks(np.arange(0.0, 20.0 + 1.0, step=4.0),labels=mystring)

plt.savefig("sensCA_DU_3.pdf", format="pdf", bbox_inches="tight")
plt.savefig('sensCA_DU_3.tif', format='tif', bbox_inches="tight", dpi=600)

#%%
ax = plt.figure().add_subplot(projection='3d')

##plots
for kk in range(1,int(a['N']),2):
    times = kk*np.ones(int(a['multLin']))
    ax.plot3D(times, a['thetaLin'][1], a['XLin_CA'][1,:,kk],linewidth=3,color=plasma.colors[50*kk])
 
ax.set(xlabel='time [h]',ylabel='$\Psi$ [-]',zlabel='$C_A$ [kmol]')

plt.xticks(np.arange(0.0, 20.0 + 1.0, step=4.0),labels=mystring)

plt.savefig("sensCA_psi_3.pdf", format="pdf", bbox_inches="tight")
plt.savefig('sensCA_psi_3.tif', format='tif', bbox_inches="tight", dpi=600)
