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
from numpy.linalg import matrix_rank
from numpy.linalg import eig

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

#%% Evaluating matrices @ nominal point       
pk = vertcat(u0,p0,ubiask,par['T'])

# state space dynamics      
A = Fx(x0=dx0, p=pk)['jac_xf_x0'].full()
B = Fp(x0=dx0, p=pk)['jac_xf_p'].full()

C = par['HCL']

# 16 biases are computed
Cd = par['b2m']

Dtemp1 = np.concatenate((A - np.identity(86), np.zeros((86,16))), axis=1)
Dtemp2 = np.concatenate((C, Cd), axis=1)
Detec = np.concatenate((Dtemp1, Dtemp2), axis=0)

rankDetec = matrix_rank(Detec)

if rankDetec == 86+16:
    display("System is detectable")
else:
    display("Nominal system is not detectable")
    
rankA = matrix_rank(A - np.identity(86))
eigA = eig(A)
