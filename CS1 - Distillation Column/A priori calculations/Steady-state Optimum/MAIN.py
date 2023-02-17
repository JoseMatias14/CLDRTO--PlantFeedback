#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# March 2022 (in Matlab); Last revision: 
# 
# Computing the model steady-state economic opt. results
#=======================================================

###################
# IMPORTING STUFF #
###################
# importing  all the
# functions defined in casadi.py
from casadi import *

# library that helps you with matrices "numpy":
import numpy as np

# for plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

# Import math Library
import math

# for saving variables 
import pickle

# import column model 
from ColumnModelRigorous import *

# import economic optimization routines
from EconomicOptimization import *

############################
# SIMULATION CONFIGURATION #
############################
# building column model
xdot, xvar, pvar = CreateColumnModelSteadyState()

# building solver for finding steady-state
ssOptSolver = SSEconomicOptimization(xvar,pvar,xdot) 

# Initial condition
dx0, y0, u0, p0 = InitialCondition()

# Model Parameters
par = SystemParameters()

# name of the file to be saved
name_file = 'SteadyStateOpt.pkl'

#%%  
########################################################
# NOMINAL OPTIMUM ANALYSIS #
########################################################
xPurity = 0.99
uGuess = u0[:4] # np.array([2,2,0.5,0.5], dtype = np.float64, ndmin=2).T | u0[:4]

# find SS optimum of the system 
# 1. NOMINAL CONDITION
sysMeas_nom = u0[4:]
theta_nom = p0
uOptk_nom, xOptk_nom, JOptk_nom, solverSol_nom = CallSSSolver(ssOptSolver,dx0[:par['NT']],uGuess,theta_nom,sysMeas_nom,xPurity)
SSOpt_nom = {'u':uOptk_nom, 'x':xOptk_nom, 'J':JOptk_nom}

# 2. STEP IN Z
sysMeas_z = np.vstack((u0[4],0.5*0.8,u0[6]))
theta_z = p0
uOptk_z, xOptk_z, JOptk_z, solverSol_z = CallSSSolver(ssOptSolver,dx0[:par['NT']],uGuess,theta_z,sysMeas_z,xPurity)
SSOpt_z = {'u':uOptk_z, 'x':xOptk_z, 'J':JOptk_z}

# 3. STEP IN ALPHA
sysMeas_a = np.vstack((u0[4],u0[5],u0[6]))
theta_a = p0*0.99
uOptk_a, xOptk_a, JOptk_a, solverSol_a = CallSSSolver(ssOptSolver,dx0[:par['NT']],uGuess,theta_a,sysMeas_a,xPurity)
SSOpt_a = {'u':uOptk_a, 'x':xOptk_a, 'J':JOptk_a}

#%%
########################################################
# PLOTTING THE RESULTS OF THE NOMINAL OPTIMUM ANALYSIS #
########################################################
fig1 = plt.figure(1)
fig1.suptitle("Opt Points")
ax = plt.axes(projection='3d')
ax.scatter3D(uOptk_nom[0], uOptk_nom[1], JOptk_nom, marker='x', c='black');
ax.scatter3D(uOptk_z[0], uOptk_z[1], JOptk_z, marker='o', c='blue');
ax.scatter3D(uOptk_a[0], uOptk_a[1], JOptk_a, marker='d', c='red');

ax.set_xlabel('L')
ax.set_ylabel('V')
ax.set_zlabel('OF')

plt.title('Contour Economic OF')
plt.xlabel('Reflux [kmol/min]')
plt.ylabel('Boilup [kmol/min]')    


fig2 = plt.figure(2)

# feed (+ and - 20%)
plt.plot(uOptk_nom[0], uOptk_nom[1],'kx', label ='Nom.')
plt.plot(uOptk_z[0], uOptk_z[1],'bo', label ='z')
plt.plot(uOptk_a[0], uOptk_a[1],'dr', label ='alpha')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Reflux [kmol/min]')
plt.ylabel('Boilup [kmol/min]')  

#%%
########################
# For saving variables #
########################
with open(name_file, 'wb') as file:
      
    # A new file will be created
    pickle.dump(SSOpt_nom, file)
    pickle.dump(SSOpt_z,file)
    pickle.dump(SSOpt_a,file)
