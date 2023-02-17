#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# April 2022; Last revision: 15-06-2022
# 
# Code runs DRTO cycle (Open-loop): 
# modelUpdate = 0 (perfect information) 
#               1 (bias update)
#               2 (state estimation via MHE)
#               3 (state and paramter estimation via MHE)
#
#=======================================================

#%% IMPORTING PACKAGES AND MODULES + LOADING FILES
# importing  all the
# functions defined in casadi.py
from casadi import *

# library that helps you with matrices "numpy":
import numpy as np
import numpy.matlib

# for plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# for saving variables 
import pickle

# for timing stuff
import time as tt

# making a beep when simulation ends
import winsound

# import column model 
from ColumnModelRigorous import *

# import economic otimization module
from EconomicOptimization import *

# import estimation module
from StateEstimation import *

# loading previously built disturbances array and zF/alpha forecasts
# Previously computed in:
# ...\A priori calculations\{Step,Ramp,Sinusoidal}
# ORDER:
# timeArray, dist1, dist1Approx, dist2, dist2Approx
distArray = []

# STEP oisturbance: disturbance_step.pkl
# RAMP disturbance: disturbance_ramp.pkl
# SINUSOIDAL disturbancce: disturbance_sin.pkl
with (open('./disturbance_ramp.pkl', 'rb')) as openfile:
    while True:
        try:
            distArray.append(pickle.load(openfile))
        except EOFError:
            break

# Loading steady-state optimum
# Previously computed in:
# ...\A priori calculations\Steady-state Optimum\
# ORDER      
# nom, z, alpha
ssOptArray = []
with (open('./SteadyStateOpt.pkl', 'rb')) as openfile:
    while True:
        try:
            ssOptArray.append(pickle.load(openfile))
        except EOFError:
            break

#%% SIMULATION CONFIGURATION 

# name of the file to be saved
name_file = 'Results_CLDRTO_bias03_ramp.pkl'

paradigm = 1 # 0 (Open-loop) | 1 (closed-loop) --> CLDRTO is not used in the paper!
modelUpdate = 1 # 0 (perfect information) 
                # 1 (bias update)
                # 2 (state estimation via MHE)
                # 3 (state and paramter estimation via MHE)


#%% BUILDING FUNCTIONS 
# building open-loop dynamic column model 
# used as plant, for estimation [EKF + MHE], and for OL-DRTO
Integ, Fx, Fp, xdot, xvar, pvar = CreateColumnModel()

# building dynamic column model (used inside DRTO)
if paradigm == 1: IntegCL, FxCL, FuCL, xdotCL, xvarCL, pvarCL = CreateColumnModelClosedLoop()
    
#  building DRTO solver
if paradigm == 0: DRTOsolver = DynamicEconomicOptimization(xvar,pvar,xdot)
if paradigm == 1: DRTOsolver = CLDynamicEconomicOptimization(xvarCL,pvarCL,xdotCL)

# building MHE solver 
if modelUpdate == 2: MHEsolver = MovingHorizonEstimatorStates(xvar,pvar,xdot)
if modelUpdate == 3: MHEsolver = MovingHorizonEstimatorStaPar(xvar,pvar,xdot)

# Initial condition
dx0, y0, u0, p0 = InitialCondition()

# Model Parameters
par = SystemParameters()

# Control parameters
ctrlPar = ControlTuning()
  
# Econominc parameters
ecoPar = EconomicSystemParameters()

# Estimation (EKF and MHE) filter parameters
est = EstimationParameters()
 
#%% STARTING THE SIMULATION 
# simulation time (from disturbance file)
simTime = distArray[0].shape[0] # minutes

# DRTO sampling time (in terms of simulation sampling time)
drtoT = ecoPar['T']/ecoPar['N']/par['T']

# Preparing time array
timeArray = np.linspace(0,simTime*par['T'],simTime)

# Manipulated variable array
MVArray = u0*np.ones([u0.size, simTime])

# loading zF disturbance
MVArray[5,:] = distArray[1]

arr = np.vstack((MVArray[0,-1],MVArray[1,-1],MVArray[2,-1],MVArray[3,-1],MVArray[4,-1],MVArray[5,-1],MVArray[6,-1]))
MVArray = np.concatenate((MVArray,np.matlib.repmat(arr, 1, 100*ecoPar['execPer'])), axis = 1)
# N.B.: here I need to extend the size of the MVarray due to the way I implement the DRTO solution. 
# I have some extra entries but they are not going to be used. The simulation runs only until simTime

# Parameter array (relative volality)
PArray = distArray[3]
PArray = np.concatenate((PArray,PArray[-1]*np.ones(100*ecoPar['execPer'],)), axis = 0)

# building setpoint array for the controllers
setpointArray = np.array([ctrlPar['MBs'],ctrlPar['MDs'],ctrlPar['xDs']], ndmin=2).T*np.ones([ctrlPar['crtlNum'], simTime + 100*ecoPar['execPer']])

#############################################################################
# UNCOMMENT AND MODIFY HERE IF YOU WANT TO CHANGE THE FINAL SIMULATION TIME #
#############################################################################
# simTime = 90*12 # [min] ! should be smaller than 800*12 due to preloaded disturbance file
# N.B.: since the simulation sampling time is 5s, by multiplying the number by 12, we change the
# units of simTime to minutes.
 
#%% CREATING VARIABLES FOR SAVING SIMULATION INFO                 
#########
# Plant #
#########
statesArray = []
ubiasArray = []
measOFArray = []
measArray = []
inputArray = []
forecastArray = []
ecoOFArray = []

################
# Optimization #
################
statesOptkArray = []
uOptkArray = []
ofOptkArray = []
optTimeArray = []
optSolArray = []
timeOpt = []

# for checking computed trajectories 
timeTrajArray = []
mvTrajArray = []
xTrajArray = []

##############
# Estimators #
##############
statesHatArray = []
biasHatArray = []
ofEstArray = []
wHatArray = []
thetaHatArray = []
arrivalCostArray = []
PkSTD = []
PkFrob = []
PkCond = []
estTimeArray = []
estSolArray = []
timeEst = []

pmHatTrajArray = []
xHatTrajArray = []

#%% INITIALIZING SIMULATION
#########
# Plant #
#########
# states
xk = dx0
statesArray.append(xk)

# System measurements (for estimation)
yk = par['HOL'].dot(xk)
measArray.append(yk)

# Objective Function measurements (D*xD,B*(1 - xB),V,(D + L),F)
yOFk = np.vstack((xk[par['NT'] - 1]*u0[2],(1 - xk[0])*u0[3],u0[1],(u0[0] + u0[2]),u0[4]))
measOFArray.append(yOFk)

# prices for OF function
alph = ecoPar['alph']
ecoOFArray.append(alph.dot(yOFk))

##################
# PI-controllers #
##################
# initializing bias (error term)
ubarB = ctrlPar['Bs']
ubarD = ctrlPar['Ds']
ubarL = ctrlPar['Ls']
ubiask = np.vstack((ubarB,ubarD,ubarL))
ubiasArray.append(ubiask)

########
# DRTO #
########
# execution time
timeEst.append(0.0)

# check if DRTO problem converged
solverDRTOSol = 0 # 0 (didn't) | 1 (converged)

# states are known
statesHatArray.append(dx0) 
# if modelUpdate == 1 or modelUpdate == 2 or modelUpdate == 3: 
#     # initial state estimate is known
#     xHatk = dx0
  
# initial state estimate is known  
if modelUpdate == 1:
    xHatk = dx0
    if paradigm == 1:
        xHatCLk = vertcat(xHatk,0.0,0.0,0.0)
            
if modelUpdate == 2 or modelUpdate == 3: 
    xHatk = dx0
    
# computed for all but only used in bias update
bHatk = np.zeros((len(yk),1)) # xD | xB | MD | MB
biasHatArray.append(bHatk)

# computed for all but only used in states update
wHatk = np.zeros((len(dx0),1))
wHatArray.append(wHatk)

# computed for all but only used in states + parameters update
#thetaHatk = np.array([u0[5],[p0]], dtype = np.float64, ndmin=2)
thetaHatk = np.array([[distArray[2][0]],[p0]], dtype = np.float64, ndmin=2)
thetaHatArray.append(thetaHatk)

# check if estimation converged
solverEstSol = 0 # 0 (didn't) | 1 (converged)

# initial value of the OF
of_est_kk = 0.0 
ofEstArray.append(of_est_kk) 

if modelUpdate == 2: 
    # covariance matrix
    Pk = est['P0']
    PkFrob.append(np.linalg.norm(Pk,ord='fro'))
    PkSTD.append(np.matrix.trace(Pk))
    PkCond.append(np.linalg.cond(Pk))
    
elif modelUpdate == 3: 
    # covariance matrix - extended states (states + parameters)
    Pk = est['Pe0']
    PkFrob.append(np.linalg.norm(Pk,ord='fro'))
    PkSTD.append(np.matrix.trace(Pk))
    PkCond.append(np.linalg.cond(Pk))
    

#%% SIMULATION (loop in sampling intervals)
for k in range(1,simTime): 
    print('Simulation time >> ',"{:.3f}".format(k*par['T']), '[min]')
      
    ####################
    # Simulating Plant #
    ####################
    # updating inputs using true values of zf and alpha
    pk = vertcat(MVArray[:,k-1],PArray[k-1],par['T'])
    
    if modelUpdate != 3:
        # updating inputs using forecast of zf and alpha (preloaded with distArray)
        pk_f = vertcat(MVArray[0,k-1],MVArray[1,k-1],MVArray[2,k-1],MVArray[3,k-1],MVArray[4,k-1],distArray[2][k-1],MVArray[6,k-1],distArray[4][k-1],est['dT'])
    else:
        # updating inputs using the estimated values
        pk_f = vertcat(MVArray[0,k-1],MVArray[1,k-1],MVArray[2,k-1],MVArray[3,k-1],MVArray[4,k-1],thetaHatk[0],MVArray[6,k-1],thetaHatk[1],est['dT'])
    
    # Evolving plant in time
    Ik = Integ(x0=xk,p=pk)
    xk = Ik['xf']

    # measuments --> objective function
    yOFk = np.vstack((xk[par['NT'] - 1]*pk[2],(1 - xk[0])*pk[3],pk[1],(pk[0] + pk[2]),pk[4]))
    
    # measurements --> estimation: "temperatures" (molar fraction) and "level" (molar holdup) measurements
    yk = par['HOL'].dot(xk)
    
    # adding noise
    yk = yk + np.reshape(distArray[5][:,k],(16,1))
    
    ###############################
    # Running COMPLETE DRTO CYCLE #
    ###############################
    #===================================================
    # 1. ESTIMATION -> executes every est['execPer'] min
    #===================================================
    if k%est['execPer'] == 0: 
        if modelUpdate == 0:
            tEKF = 0
            tMHE = 0
            
        if modelUpdate == 1: 
            
            # calling bias update
            t3 = tt.time()
            
            # Evolving plant in time
            if paradigm == 0:
                INomk = Integ(x0=xHatk,p=pk_f)
                xHatk = INomk['xf'].full()
            if paradigm == 1:
                pk_CL_f = np.vstack((setpointArray[2,k-1],MVArray[1,k-1],setpointArray[1,k-1],setpointArray[0,k-1],MVArray[4,k-1],distArray[2][k-1],MVArray[6,k-1],distArray[4][k-1],ctrlPar['Bs'],ctrlPar['Ds'],ctrlPar['Ls'],est['dT']))
                INomk = IntegCL(x0=xHatCLk,p=pk_CL_f)
                xHatCLk = INomk['xf'].full()
                xHatk = xHatCLk[:-3]
                
            # computing bias
            # bTemp = par['b2m'].dot(yk)
            # bhatk_0_temp = bTemp[0] - xHatk[par['NT'] - 1]     # xD
            # bhatk_1_temp = bTemp[1] - xHatk[0]                 # xB
            # bhatk_2_temp = bTemp[2] - xHatk[2*par['NT'] - 1]   # MD
            # bhatk_3_temp = bTemp[3] - xHatk[par['NT']]         # MB
            # bHatk = np.vstack((bhatk_0_temp,bhatk_1_temp,bhatk_2_temp,bhatk_3_temp)) 
            bHatkTemp = par['b2m'].dot(yk) - par['b2m'].dot(par['HOL'].dot(xHatk))               
            bHatk = (1 - est['lambdaBias'])*bHatkTemp + est['lambdaBias']*bHatk               
        
            # name of the variable is tMHE only for bookkeeping
            tMHE = tt.time() - t3  
            tEKF = 0.0
            
        if modelUpdate == 2 or modelUpdate == 3:
            if k > est['T']*est['execPer']:
                # extracting information for MHE --> list[start:stop:step]
                # T past measurements
                Y_k = np.hstack(measArray)[:,-(est['T'] - 1)*est['execPer'] - 1::est['execPer']]
                # T past inputs ---> from forecast 
                P_k = np.hstack(forecastArray)[:,-(est['T'] - 1)*est['execPer'] - 1::est['execPer']]
                # state estimate at the beginning of the MHE horizon @ the beginning of the window
                xHatk_N = statesHatArray[-(est['T'] - 1) - 1]
                # arrival cost (here calculated using an EKF)  @ the beginning of the window
                Pk = arrivalCostArray[-(est['T'] - 1) - 1]    
                # state estimate at the beginning of the MHE horizon @ the beginning of the window (dummy for only state estimation)
                thetaHatk_N = thetaHatArray[-(est['T'] - 1) - 1]
                
                # a warm-up strategy is used
                # for the first MHE iteration we need guesses for the primal and dual variables
                # also, for the iterations that MHE hasn`t converge
                if solverEstSol == 0:
                    xArray_est_k = np.hstack(statesHatArray)[:,-(est['T'] + 1):] 
                    wArray_est_k = np.zeros((len(dx0),est['T'])) # dummy for states + parameters
                    thetaArray_est_k = np.hstack(thetaHatArray)[:,-(est['T'] + 1):]
                                         
                    if modelUpdate == 2: 
                        w_est_warm, lam_w_est_warm, lam_g_est_warm = MHEGuessInitializationStates(xArray_est_k,wArray_est_k)
                    if modelUpdate == 3: 
                        w_est_warm, lam_w_est_warm, lam_g_est_warm = MHEGuessInitializationStaPar(xArray_est_k,thetaArray_est_k)
                
                # calling MHE
                t1 = tt.time()
                
                if modelUpdate == 2: 
                    xHatk, wHatk, of_est_kk, xArray_est_k, pmArray_est_k, w_est_warm, lam_w_est_warm, lam_g_est_warm, solverEstSol  = CallMovingHorizonEstimatorStates(MHEsolver,xHatk_N,Y_k,P_k,Pk,w_est_warm,lam_w_est_warm,lam_g_est_warm)
                    
                if modelUpdate == 3: 
                    xHatk, thetaHatk, of_est_kk, xArray_est_k, pmArray_est_k, w_est_warm, lam_w_est_warm, lam_g_est_warm, solverEstSol  = CallMovingHorizonEstimatorStaPar(MHEsolver,xHatk_N,thetaHatk_N,Y_k,P_k,Pk,w_est_warm,lam_w_est_warm,lam_g_est_warm)

                tMHE = tt.time() - t1        
                        
                # running EKF to update the covariance matrix        
                t2 = tt.time()           
                if modelUpdate == 2: dummy, Pk = ExtendedKalmanFilterStates(yk,pk_f,xHatk,Pk,Integ,Fx)
                if modelUpdate == 3: dummy, thetaHatk, Pk = ExtendedKalmanFilterStaPar(yk,pk_f,xHatk,thetaHatk,Pk,Integ,Fx,Fp)    
                tEKF = tt.time() - t2
                
            else:
                # running EKF until MHE window is filled         
                t2 = tt.time()
                if modelUpdate == 2: xHatk, Pk = ExtendedKalmanFilterStates(yk,pk_f,xHatk,Pk,Integ,Fx)
                if modelUpdate == 3: xHatk, thetaHatk, Pk = ExtendedKalmanFilterStaPar(yk,pk_f,xHatk,thetaHatk,Pk,Integ,Fx,Fp)
                tEKF = tt.time() - t2
                    
                # only for bookkeeping
                tMHE = 0.0
                of_est_kk = 0.0 
                xArray_est_k = 0.0
                pmArray_est_k = 0.0
                            
        ######################  
        # Saving Information #
        ######################
        # Estimation time 
        timeEst.append(k*par['T'])
        # of value
        ofEstArray.append(of_est_kk)
        # checking convergence of the solver
        estSolArray.append(solverEstSol)
        # parameters and bias update 
        # dummy for perfect information | states | states + parameters
        biasHatArray.append(bHatk)
        # dummy for perfect information | bias update | states
        thetaHatArray.append(thetaHatk)
        # dummy for perfect information | bias update | states + parameters
        wHatArray.append(wHatk)
        # CPU time for computing estimates
        timeTemp = np.vstack((tEKF,tMHE))
        estTimeArray.append(timeTemp)
            
        if modelUpdate == 0:
            # estimates states
            statesHatArray.append(np.array(xk))
        else: # modelUpdate == 1 or modelUpdate == 2 or modelUpdate == 3:
            # estimates states
            statesHatArray.append(np.array(xHatk))
            
        if modelUpdate == 2 or modelUpdate == 3:
            # arrival cost
            arrivalCostArray.append(Pk)
            # Frobenius norm of the estimate covariance matrix 
            PkFrob.append(np.linalg.norm(Pk,ord='fro'))
            # main diagonal (states variance)
            PkSTD.append(np.matrix.trace(Pk))
            # condition number of matrix
            PkCond.append(np.linalg.cond(Pk))
  
            
            # saving the parameter or bias array
            pmHatTrajArray.append(pmArray_est_k)
            # saving the state trajectory array
            xHatTrajArray.append(xArray_est_k)
                    
    #======================================================
    # 2. OPTIMIZATION -> executes every ecoPar['execPer'] min
    #======================================================
    if k%ecoPar['execPer'] == 0: # bool(0):
      
        #########################################################################
        # computing the future trajectories of the feed fraction and parameters #
        #########################################################################
        if modelUpdate == 0:
            # known by perfect DRTO
            thetaTraj = PArray[k-1:k-1 + int(drtoT*(ecoPar['N'] - 1)) + 1:int(drtoT)]
            zTraj = MVArray[5,k-1:k-1 + int(drtoT*(ecoPar['N'] - 1)) + 1:int(drtoT)]
        if modelUpdate == 1 or  modelUpdate == 2:
            # computing from forecast (fixed along DRTO prediction horizon) 
            thetaTraj = distArray[4][k-1]*np.ones((ecoPar['N'],))
            zTraj = distArray[2][k-1]*np.ones((ecoPar['N'],))
        if modelUpdate == 3:
            # computing from estimated (fixed along DRTO prediction horizon)              
            thetaTraj = thetaHatk[1]*np.ones((ecoPar['N'],))
            zTraj = thetaHatk[0]*np.ones((ecoPar['N'],))
        
        ###########################################   
        # current inputs implemented in the plant #
        ###########################################
        if paradigm == 0:
            # Uk = [L, V, D, B, F, (((zF))), qF]
            if modelUpdate == 0:
                Uk = np.array(MVArray[:,k-1], ndmin=2).T
            if modelUpdate == 1 or  modelUpdate == 2:
                Uk = np.vstack((MVArray[0,k-1],MVArray[1,k-1],MVArray[2,k-1],MVArray[3,k-1],MVArray[4,k-1],distArray[2][k-1],MVArray[6,k-1]))   
            if modelUpdate == 3:
                Uk = np.vstack((MVArray[0,k-1],MVArray[1,k-1],MVArray[2,k-1],MVArray[3,k-1],MVArray[4,k-1],thetaHatk[0],MVArray[6,k-1]))
        
        if paradigm == 1:
            # Uk = [xDs, V, MDs, MBs, F, (((zF))), qF]
            if modelUpdate == 0:
                UkCL = np.vstack((setpointArray[2,k-1],MVArray[1,k-1],setpointArray[1,k-1],setpointArray[0,k-1],MVArray[4,k-1],MVArray[5,k-1],MVArray[6,k-1]))
            if modelUpdate == 1 or  modelUpdate == 2:
                UkCL = np.vstack((setpointArray[2,k-1],MVArray[1,k-1],setpointArray[1,k-1],setpointArray[0,k-1],MVArray[4,k-1],distArray[2][k-1],MVArray[6,k-1]))
            if modelUpdate == 3:
                UkCL = np.vstack((setpointArray[2,k-1],MVArray[1,k-1],setpointArray[1,k-1],setpointArray[0,k-1],MVArray[4,k-1],thetaHatk[0],MVArray[6,k-1]))
        
        #################################################
        # extracting feed information [F, (((zF))), qF] #
        #################################################
        if modelUpdate == 0:
            sysMeas = np.array((MVArray[4,k-1],MVArray[5,k-1],MVArray[6,k-1]), ndmin=2).T 
        if modelUpdate == 1 or  modelUpdate == 2:
            sysMeas = np.array((MVArray[4,k-1],distArray[2][k-1],MVArray[6,k-1]), ndmin=2).T    
        if modelUpdate == 3:
            sysMeas = np.array((MVArray[4,k-1],thetaHatk[0],MVArray[6,k-1]), ndmin=2).T     
            
        #################
        # Bias Feedback #
        #################  
        if modelUpdate == 1:
            bHatk = bHatk    
        else:
            bHatk = np.zeros((len(yk),1)) # no bias 
            
        ##################
        # State Feedback #
        ##################             
        if modelUpdate == 0:
            # full-state feedback
            x0Hatk = np.array(xk)
        else:
            x0Hatk = xHatk
            
        if  paradigm == 1:
            # adding the integral action states (one for each controller)
            x0HatCLk = vertcat(x0Hatk,0.0,0.0,0.0)
            # computing the current bias value to initialize integral action
            biasCtrlk = ubiasArray[-1] 

        xPurity = ecoPar['xPurity']

        # a warm-up strategy is used in the DRTO
        # for the first DRTO iteration we need guesses for the primal and dual variables
        # also, for the iterations that DRTO hasn`t converge
        if  solverDRTOSol == 0:
            # initializing solver
            if paradigm == 0: w_opt_warm, lam_w_opt_warm, lam_g_opt_warm =  DRTOGuessInitialization(x0Hatk,Uk)
            if paradigm == 1: w_opt_warm, lam_w_opt_warm, lam_g_opt_warm =  CLDRTOGuessInitialization(x0HatCLk,UkCL)
        
        # computing the input trajectory
        t3 = tt.time()
        if paradigm == 0:
            uArray_opt_k, xArray_opt_k, xTemp, mvTemp, timeTemp, of_opt_sol, w_opt_warm, lam_w_opt_warm, lam_g_opt_warm, solverDRTOSol = CallDRTO(DRTOsolver,x0Hatk,Uk,thetaTraj,zTraj,bHatk,xPurity,sysMeas,w_opt_warm,lam_w_opt_warm,lam_g_opt_warm) 
        if paradigm == 1:
            uArray_opt_k, xArray_opt_k, xTemp, mvTemp, timeTemp, of_opt_sol, w_opt_warm, lam_w_opt_warm, lam_g_opt_warm, solverDRTOSol = CallCLDRTO(DRTOsolver,x0HatCLk,UkCL,thetaTraj,zTraj,bHatk,biasCtrlk,xPurity,sysMeas,w_opt_warm,lam_w_opt_warm,lam_g_opt_warm) 

        tDRTO = tt.time() - t3
        
        # updating MV array for the next ecoPar['execPer'] periods
        DRTOcounter = 0
        
        # DRTO sampling time
        DRTOsamp = ecoPar['T']/ecoPar['N']
        
        # Update inputs if simulation succeded 
        if solverDRTOSol == 1:
            for kk in range(int(ecoPar['T']/par['T'])): # range(ecoPar['execPer'] + int(DRTOsamp*12)):
                # DRTO sampling time is 6 minutes
                if kk != 0 and kk%(DRTOsamp*12) == 0: DRTOcounter += 1
                
                # update the inputs based on the DRTO solution
                MVArray[1,k + kk] = uArray_opt_k[1,DRTOcounter]
                if paradigm == 0:
                    setpointArray[2,k + kk] = xArray_opt_k[par['NT'] - 1,DRTOcounter + 1] + bHatk[0]
                    #setpointArray[1,k + kk] = xArray_opt_k[par['NT'],DRTOcounter + 1] + bHatk[3]
                    #setpointArray[0,k + kk] = xArray_opt_k[2*par['NT'] - 1,DRTOcounter + 1] + bHatk[2]
                if paradigm == 1:
                    setpointArray[2,k + kk] = uArray_opt_k[0,DRTOcounter] + bHatk[0]
                    setpointArray[1,k + kk] = uArray_opt_k[2,DRTOcounter] + bHatk[2]
                    setpointArray[0,k + kk] = uArray_opt_k[3,DRTOcounter] + bHatk[3]
                # N.B. bias is zero except when bias update method is chosen as the model updating strategy

        ######################  
        # Saving Information #
        ######################
        # Optimization time 
        timeOpt.append(k*par['T'])       
        # optimization computed state
        statesOptkArray.append(xArray_opt_k)
        # optimization MV value
        uOptkArray.append(uArray_opt_k)
        # optimization value array
        ofOptkArray.append(of_opt_sol)
        # CPU time for computing DRTO
        optTimeArray.append(tDRTO)
        # Converged?
        optSolArray.append(solverDRTOSol)
        
        # checking collocation
        timeTrajArray.append(k*par['T'] + timeTemp) 
        mvTrajArray.append(mvTemp)
        xTrajArray.append(xTemp)
                  
    ############################  
    # Saving Plant Information #
    ############################
    # true states
    statesArray.append(np.array(xk))
    # OF measurements
    measOFArray.append(yOFk)
    # system measurements
    measArray.append(yk)
    # inputs 
    inputArray.append(np.array(pk))
    # system forecast
    forecastArray.append(np.array(pk_f))
    # objective function
    # setting profit to zero if top concentration is 
    # outside the band
    if statesArray[-1][par['NT'] - 1] < 0.99-0.0013:  # 0.00065 | 0.0013
        ecoOFArray.append(0.0)
    else:
        ecoOFArray.append(alph.dot(yOFk))

    #########################################################
    # PI-Controllers - reboiler, condenser levels and comp. #
    #########################################################
    ## REBOILER ##
    # Actual reboiler holdup
    MB = xk[par['NT']].full()
    # computing error
    eB = MB - setpointArray[0,k]
    # adjusting bias (accounting for integral action)
    ubarB = ubarB + ctrlPar['KcB']/ctrlPar['tauB']*eB*par['T']
    # Bottoms flor
    MVArray[3,k] = ubarB + ctrlPar['KcB']*eB
    # clipping if negative values
    if MVArray[3,k] < 0: 
        MVArray[3,k] = 0
        ubarB = - ctrlPar['KcB']*eB # resetting bias
    
    ## CONDENSER ##
    # Actual condenser holdup
    MD = xk[2*par['NT'] - 1].full() 
    # computing error
    eD = MD - setpointArray[1,k]
    # adjusting bias (accounting for integral action)
    ubarD = ubarD + ctrlPar['KcD']/ctrlPar['tauD']*eD*par['T']
    # Distillate flow
    MVArray[2,k] = ubarD + ctrlPar['KcD']*eD           
    # clipping 
    if MVArray[2,k] < 0: 
        MVArray[2,k] = 0
        ubarD = - ctrlPar['KcD']*eD
        
    ## TOP COMP. ##
    # Actual top composition
    xD = xk[2*par['NT']].full() # xk[par['NT'] - 1].full() 
    # computing error
    eL = xD - setpointArray[2,k]
    # adjusting bias (accounting for integral action)
    ubarL = ubarL + ctrlPar['KcL']/ctrlPar['tauL']*eL*par['T']
    # Distillate flow
    MVArray[0,k] = ubarL + ctrlPar['KcL']*eL    
    # clipping 
    if MVArray[0,k] < 0: 
        MVArray[0,k] = 0
        ubarL = - ctrlPar['KcL']*eL

    #################################  
    # Saving Controller Information #
    #################################    
    ubiask = np.vstack((ubarB,ubarD,ubarL))
    ubiasArray.append(ubiask)
            
#%%   
######################    
# Preparing for plot #
######################
# System
statesArray = np.hstack(statesArray)
ubiasArray = np.hstack(ubiasArray)
measOFArray = np.hstack(measOFArray)
measArray = np.hstack(measArray)
ecoOFArray = np.hstack(ecoOFArray)
inputArray = np.hstack(inputArray)

# Estimation
statesHatArray = np.hstack(statesHatArray)
biasHatArray = np.hstack(biasHatArray)
thetaHatArray = np.hstack(thetaHatArray)

if modelUpdate == 2 or modelUpdate == 3:
    PkSTD = np.hstack(PkSTD)
    PkFrob = np.hstack(PkFrob)
    PkCond = np.hstack(PkCond)

estTimeArray = np.hstack(estTimeArray)
estSolArray = np.hstack(estSolArray)
    
# optimization
ofOptkArray = np.hstack(ofOptkArray)
optTimeArray = np.hstack(optTimeArray)
optSolArray = np.hstack(optSolArray)

#%%
########################
# For saving variables #
########################
with open(name_file, 'wb') as file:
      
    # A new file will be created
    #PLANT
    pickle.dump(MVArray, file)
    pickle.dump(PArray,file)
    pickle.dump(forecastArray,file)
    pickle.dump(setpointArray,file)
    pickle.dump(statesArray, file)
    pickle.dump(measArray,file)
    pickle.dump(inputArray,file)
    pickle.dump(ubiasArray, file)
    pickle.dump(ecoOFArray,file)
    pickle.dump(timeArray,file)
    #DRTO
    pickle.dump(timeEst,file)
    pickle.dump(timeOpt,file)
    #estimation
    pickle.dump(statesHatArray,file)
    pickle.dump(biasHatArray,file)
    pickle.dump(ofEstArray,file)
    pickle.dump(wHatArray,file)
    pickle.dump(thetaHatArray,file)   
    pickle.dump(arrivalCostArray,file)
    pickle.dump(PkSTD,file)
    pickle.dump(PkFrob,file)
    pickle.dump(PkCond,file)
    pickle.dump(estTimeArray,file)
    pickle.dump(estSolArray,file)
    pickle.dump(pmHatTrajArray,file)
    pickle.dump(xHatTrajArray,file)    
    #optimization
    pickle.dump(statesOptkArray,file)
    pickle.dump(uOptkArray,file)
    pickle.dump(ofOptkArray,file)
    pickle.dump(optTimeArray,file)
    pickle.dump(optSolArray,file)
    pickle.dump(timeTrajArray,file)
    pickle.dump(mvTrajArray,file)
    pickle.dump(xTrajArray,file)

########
# Plot #
########

#%% 1. MVS
majorR = 60
minorR = 30

deltaPlot = 5*12 # plot one point every 5 minutes

fig1, axs = plt.subplots(4, sharex=True)
fig1.suptitle('Inputs [kmol/min]')

for ii in range(4):
    axs[ii].plot(timeArray[0:simTime:deltaPlot],MVArray.T[0:simTime:deltaPlot,ii],'b', label ='PI', linewidth=4)
    # Nominal
    axs[ii].hlines(ssOptArray[0]['u'][ii],0,100,color='gray', linestyles='dotted', linewidth=2)
    # zStep
    axs[ii].hlines(ssOptArray[1]['u'][ii],100,simTime/12,color='gray', linestyles='dotted', linewidth=2,label='SS Opt')
    
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
axs[0].set(ylabel='L') # [kmol/min]    
axs[1].set(ylabel='V') # [kmol/min]  
axs[2].set(ylabel='D') # [kmol/min]  
axs[3].set(ylabel='B',xlabel='t [min]') # [kmol/min]    

#%% 2. Objective function
fig2, (ax1, ax2) = plt.subplots(2, sharex=True)

ax2.set_title('Top Purity Constraint')
ax2.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,40] ,'b:', linewidth=2)
ax2.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,82] ,'b', linewidth=4)
ax2.plot(timeArray[0:simTime:deltaPlot], setpointArray.T[0:simTime:deltaPlot,2],'k--', linewidth=3,label='SP')
#ax2.hlines(0.99-0.0013,0,simTime/12,color='gray', linestyles='dotted', linewidth=2,label='low lim')
ax2.set_ylim([0.988, 0.992]) 
ax2.set(ylabel='x_D [-]',xlabel='t [min]')
#ax2.legend(loc='best')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax1.set_title('Economic Objective Function')
ax1.plot(timeArray[0:simTime:deltaPlot], ecoOFArray.T[0:simTime:deltaPlot],'b', linewidth=4,label='Plant')
ax1.hlines(ssOptArray[0]['J'],0,100,color='gray', linestyles='dotted', linewidth=2,label='SS Opt')
ax1.hlines(ssOptArray[1]['J'],100,simTime/12,color='gray', linestyles='dotted', linewidth=2)

ax1.set(ylabel='OF [$/min]') # [kmol/min]
ax1.set_ylim([0.4, 0.8]) 
ax1.legend(loc='best')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

#%% 3. CONCENTRATION STATES
fig3, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig3.suptitle('Controlled States')

ax1.plot(timeArray[0:simTime:deltaPlot], setpointArray.T[0:simTime:deltaPlot,0],'k--', linewidth=4)
ax1.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,par['NT']],'b', linewidth=4)
ax1.set(ylabel='Sump [kmol]')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray[0:simTime:5*12], setpointArray.T[0:simTime:5*12,1],'k--', linewidth=4)
ax2.plot(timeArray[0:simTime:5*12], statesArray.T[0:simTime:5*12,2*par['NT'] - 1],'b', linewidth=4)
ax2.set(ylabel='Condenser [kmol]',xlabel='t [min]')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax3.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,40],'b:', linewidth=2)
ax3.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,82],'b', linewidth=4)
ax3.plot(timeArray[0:simTime:deltaPlot], setpointArray.T[0:simTime:deltaPlot,2],'k--', linewidth=4)
ax3.set(ylabel='x top [-]',xlabel='t [min]')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(majorR))
ax3.xaxis.set_minor_locator(MultipleLocator(minorR))
ax3.yaxis.set_tick_params(which='minor', bottom=False)
ax3.grid(b=True, which='major', linestyle='-')
ax3.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

## HOLDUP STATES
fig4, (ax1, ax2) = plt.subplots(2, sharex=True)
fig4.suptitle('Uncontrolled States: xfrac (selected)')

ax1.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,0],'b', linewidth=4)
ax1.set(ylabel='x bottom [-]')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,21],'b', linewidth=4)
ax2.set(ylabel='x above feed [-]',xlabel='t [min]')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

#%% 5. Disturbances
fig5, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(timeArray[:simTime],distArray[1][:simTime],'r',linewidth=4,label='True')

if modelUpdate == 3:
    ax1.plot(timeEst,thetaHatArray[0],'bx', markersize =2,label='Est.')
else:
    ax1.plot(timeArray[0:simTime:deltaPlot],distArray[2][0:simTime:deltaPlot],'b', linewidth=4,label='Forecast')

ax1.set(ylabel='zF (A) [-]')  
ax1.set_title('Column feed (A fraction) disturbance')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray[:simTime],distArray[3][:simTime],'r',linewidth=4,label='True')

if modelUpdate == 3:
    ax2.plot(timeEst,thetaHatArray[1],'bx', markersize=2,label='Est.')
else:
    ax2.plot(timeArray[0:simTime:deltaPlot],distArray[4][0:simTime:deltaPlot],'b',linewidth=4,label='Forecast')

ax2.set(xlabel='t [min]',ylabel='alpha [-]')     
ax2.set_title('Pressure disturbance')
ax2.legend(loc='best')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

#%% 6, 7. State Estimation
fig6, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig6.suptitle('States (mol fractions [-]): estimated vs. true')

ax1.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,40],'r', label ='True', linewidth=4)
ax1.plot(timeEst, statesHatArray.T[:,40],'bx',markersize=3, label ='Est.')
ax1.set(ylabel='x_D')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,21],'r', label ='True', linewidth=4)
ax2.plot(timeEst, statesHatArray.T[:,21],'bx',markersize=3, label ='Est.')
ax2.set(ylabel='above feed')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax3.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,3],'r', label ='True', linewidth=4)
ax3.plot(timeEst, statesHatArray.T[:,3],'bx',markersize=3,label ='Est.')
ax3.legend(loc='best')
ax3.set(xlabel='t [min]',ylabel='near bottom')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(majorR))
ax3.xaxis.set_minor_locator(MultipleLocator(minorR))
ax3.yaxis.set_tick_params(which='minor', bottom=False)
ax3.grid(b=True, which='major', linestyle='-')
ax3.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

## HOLDUP STATES
fig7, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig7.suptitle('States (holdups [kmol] ): estimated vs. true')

ax3.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,44],'r', label ='True', linewidth=4)
ax3.plot(timeEst, statesHatArray.T[:,44],'bx',markersize=3, label ='Est.')
ax3.legend(loc='best')
ax3.set(xlabel='t [min]',ylabel='near bottom')
ax3.minorticks_on()
ax3.xaxis.set_major_locator(MultipleLocator(majorR))
ax3.xaxis.set_minor_locator(MultipleLocator(minorR))
ax3.yaxis.set_tick_params(which='minor', bottom=False)
ax3.grid(b=True, which='major', linestyle='-')
ax3.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax2.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,64],'r', label ='True', linewidth=4)
ax2.plot(timeEst, statesHatArray.T[:,64],'bx',markersize=3, label ='Est.')
ax2.set(ylabel='above feed')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MultipleLocator(majorR))
ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
ax2.yaxis.set_tick_params(which='minor', bottom=False)
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

ax1.plot(timeArray[0:simTime:deltaPlot], statesArray.T[0:simTime:deltaPlot,77],'r', label ='True', linewidth=4)
ax1.plot(timeEst, statesHatArray.T[:,77],'bx',markersize=3, label ='Est.')
ax1.set(ylabel='near top')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MultipleLocator(majorR))
ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
ax1.yaxis.set_tick_params(which='minor', bottom=False)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

#%% 8. DRTO Solution time
fig7, axs = plt.subplots(2, sharex=True)
fig7.suptitle('DRTO Solution time')

axs[0].plot(timeOpt,optTimeArray,'kx',markersize=4)
axs[0].set(ylabel='sol time [ms]') # [kmol/min]  

axs[1].plot(timeOpt,optSolArray,'kx',markersize=4)
axs[1].set_yticks([0,1])
axs[1].set_yticklabels(['No','Yes'])
axs[1].set(xlabel='t [min]',ylabel='sol?') # [kmol/min]  

for ii in range(2):
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

#%% 8. MHE Solution time and 9. Filter information
if modelUpdate == 2 or modelUpdate == 3:
    fig8, axs = plt.subplots(2, sharex=True)
    fig8.suptitle('MHE Solution time')
    
    axs[0].plot(timeEst[1:],estTimeArray.T[:,0],'kx',markersize=4, label ='EKF')
    axs[0].plot(timeEst[1:],estTimeArray.T[:,1],'bo',markersize=4, label ='MHE')
    axs[0].set(ylabel='sol time [ms]') # [kmol/min]  
    axs[0].legend(loc='best')
    
    axs[1].plot(timeEst[1:],estSolArray,'kx',markersize=4)
    axs[1].set_yticks([0,1])
    axs[1].set_yticklabels(['No','Yes'])
    axs[1].set(xlabel='t [min]',ylabel='sol?') # [kmol/min]  
    
    for ii in range(2):
        axs[ii].minorticks_on()
        axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
        axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
        axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
        axs[ii].grid(b=True, which='major', linestyle='-')
        axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    
    fig9, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig9.suptitle('Covariance Matrix Information')
    
    ax1.plot(timeEst, PkFrob.T,'b', linewidth=4)
    #ax1.set_xlim([50*12/60, timeEst[-1]])
    ax1.set(ylabel='Frob. Norm')
    ax1.minorticks_on()
    ax1.xaxis.set_major_locator(MultipleLocator(majorR))
    ax1.xaxis.set_minor_locator(MultipleLocator(minorR))
    ax1.yaxis.set_tick_params(which='minor', bottom=False)
    ax1.grid(b=True, which='major', linestyle='-')
    ax1.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    ax2.plot(timeEst, PkSTD.T,'b', linewidth=4)
    ax2.set(xlabel='t [min]',ylabel='Trace')
    ax2.minorticks_on()
    ax2.xaxis.set_major_locator(MultipleLocator(majorR))
    ax2.xaxis.set_minor_locator(MultipleLocator(minorR))
    ax2.yaxis.set_tick_params(which='minor', bottom=False)
    ax2.grid(b=True, which='major', linestyle='-')
    ax2.grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
#%% 10. Computed setpoints
fig10, axs = plt.subplots(4, sharex=True)
fig10.suptitle('DRTO Decisions')

DRTOsamp = ecoPar['T']/ecoPar['N']

if paradigm == 0:
    for kk in range(len(timeOpt)):
        time_kk = list(range(int(timeOpt[kk]),int(timeOpt[kk] + ecoPar['T'] + DRTOsamp),int(DRTOsamp)))
        axs[0].plot(time_kk,statesOptkArray[kk][par['NT'] - 1,:],'b',alpha=0.5, linewidth=3)  
        axs[1].plot(time_kk[1:],uOptkArray[kk][1,:],'b',alpha=0.5, linewidth=3)    

    axs[0].plot(timeArray[0:simTime:deltaPlot],setpointArray.T[0:simTime:deltaPlot,2],'r', linewidth=3, label ='Impl.')
    axs[1].plot(timeArray[0:simTime:deltaPlot],MVArray.T[0:simTime:deltaPlot,1],'r', linewidth=3, label ='Impl.')
    axs[2].plot(timeArray[0:simTime:deltaPlot],setpointArray.T[0:simTime:deltaPlot,0],'r:', linewidth=3, label ='Impl.')
    axs[3].plot(timeArray[0:simTime:deltaPlot],setpointArray.T[0:simTime:deltaPlot,1],'r:', linewidth=3, label ='Impl.')    


if paradigm == 1:
    for kk in range(len(timeOpt)):
        time_kk = list(range(int(timeOpt[kk]),int(timeOpt[kk] + ecoPar['T'] + DRTOsamp),int(DRTOsamp)))
        axs[0].plot(time_kk[1:],uOptkArray[kk][0,:],'b',alpha=0.5, linewidth=3)  
        axs[1].plot(time_kk[1:],uOptkArray[kk][1,:],'b',alpha=0.5, linewidth=3)
        axs[2].plot(time_kk[1:],uOptkArray[kk][2,:],'b',alpha=0.5, linewidth=3)
        axs[3].plot(time_kk[1:],uOptkArray[kk][3,:],'b',alpha=0.5, linewidth=3)

    axs[0].plot(timeArray[0:simTime:deltaPlot],setpointArray.T[0:simTime:deltaPlot,2],'r', linewidth=3, label ='Impl.')
    axs[1].plot(timeArray[0:simTime:deltaPlot],MVArray.T[0:simTime:deltaPlot,1],'r', linewidth=3, label ='Impl.')
    axs[2].plot(timeArray[0:simTime:deltaPlot],setpointArray.T[0:simTime:deltaPlot,0],'r', linewidth=3, label ='Impl.')
    axs[3].plot(timeArray[0:simTime:deltaPlot],setpointArray.T[0:simTime:deltaPlot,1],'r', linewidth=3, label ='Impl.')  
    

for ii in range(4):
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

axs[0].set(ylabel='xDs') # [kmol/min]    
axs[1].set(ylabel='V') # [kmol/min]  
axs[2].set(ylabel='MDs') # [kmol/min] 
axs[3].set(xlabel='t [min]', ylabel='MBs') # [kmol/min] 

axs[0].set_ylim([0.988, 0.994])
# axs[0].set_ylim([0.95, 1.0])
axs[1].set_ylim([2.7, 3.8])
# axs[1].set_ylim([2.5, 3.8])
axs[2].set_ylim([0.0, 0.8])
axs[3].set_ylim([0.0, 0.8])


#%% 11. Computed outputs
fig11, axs = plt.subplots(3, sharex=True)
fig11.suptitle('DRTO States (controlled)')

DRTOsamp = ecoPar['T']/ecoPar['N']

for kk in range(len(timeOpt)):
    time_kk = list(range(int(timeOpt[kk]),int(timeOpt[kk] + ecoPar['T'] + DRTOsamp),int(DRTOsamp)))
    
    axs[0].plot(time_kk,statesOptkArray[kk][par['NT'],:],'b',alpha=0.5, linewidth=3)
    if kk == 1:
        axs[1].plot(time_kk,statesOptkArray[kk][2*par['NT'] - 1,:],'b',alpha=0.5, linewidth=3, label ='Comp.')
    else:
        axs[1].plot(time_kk,statesOptkArray[kk][2*par['NT'] - 1,:],'b',alpha=0.5, linewidth=3)
    
    axs[2].plot(time_kk,statesOptkArray[kk][par['NT'] - 1,:],'b',alpha=0.5, linewidth=3)
     
for ii in range(3):
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

axs[0].plot(timeArray[0:simTime:deltaPlot],statesArray.T[0:simTime:deltaPlot,par['NT']],'r', linewidth=4, label ='Impl.')
axs[1].plot(timeArray[0:simTime:deltaPlot],statesArray.T[0:simTime:deltaPlot,2*par['NT'] - 1],'r', linewidth=4, label ='Impl.')
axs[2].plot(timeArray[0:simTime:deltaPlot],statesArray.T[0:simTime:deltaPlot,par['NT'] - 1],'r', linewidth=4, label ='Impl.')

axs[1].legend(loc='best')

axs[0].set(ylabel='MB') # [kmol/min]    
axs[1].set(ylabel='MD') # [kmol/min]  
axs[2].set(xlabel='t [min]', ylabel='xD') # [kmol/min]  

axs[0].set_ylim([-0.1, 1.0])
axs[1].set_ylim([-0.1, 1.0])
axs[2].set_ylim([0.988, 0.992])
#axs[2].set_ylim([0.98, 1.0])

#%% 12. Selected holdups inputs
fig12, axs = plt.subplots(5, sharex=True)
fig12.suptitle('DRTO States (holdups)')

DRTOsamp = ecoPar['T']/ecoPar['N']

for kk in range(len(timeOpt)):
    time_kk = list(range(int(timeOpt[kk]),int(timeOpt[kk] + ecoPar['T'] + DRTOsamp),int(DRTOsamp)))
    axs[0].plot(time_kk,statesOptkArray[kk][par['NT'] + 1,:],'b',alpha=0.5, linewidth=3)
    axs[1].plot(time_kk,statesOptkArray[kk][par['NT'] + 9,:],'b',alpha=0.5, linewidth=3)
    axs[2].plot(time_kk,statesOptkArray[kk][par['NT'] + 17,:],'b',alpha=0.5, linewidth=3)
    axs[3].plot(time_kk,statesOptkArray[kk][2*par['NT'] - 3,:],'b',alpha=0.5, linewidth=3)
    axs[4].plot(time_kk,statesOptkArray[kk][2*par['NT'] - 2,:],'b',alpha=0.5, linewidth=3)

        
for ii in range(5):
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

axs[0].set(ylabel='M(2)') # [kmol/min]    
axs[1].set(ylabel='M(10)') # [kmol/min]  
axs[2].set(ylabel='M(18)') # [kmol/min]  
axs[3].set(ylabel='M(39)') # [kmol/min]  
axs[4].set(xlabel='t [min]', ylabel='M(40)') # [kmol/min]  


#%% 13. Computed inputs
fig13, axs = plt.subplots(3, sharex=True)
fig13.suptitle('DRTO MVs (computed vs. implemented)')

DRTOsamp = ecoPar['T']/ecoPar['N']

for kk in range(len(timeOpt)):
    for ii in range(3):
        if kk == 1:
            axs[ii].plot(timeTrajArray[kk][1:],mvTrajArray[kk][ii,:],'b',alpha=0.5, linewidth=3, label ='Comp.')
        else:
            axs[ii].plot(timeTrajArray[kk][1:],mvTrajArray[kk][ii,:],'b',alpha=0.5, linewidth=3)
       
        
for ii in range(3):
    axs[ii].minorticks_on()
    axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
    axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
    axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
    axs[ii].grid(b=True, which='major', linestyle='-')
    axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)

axs[0].plot(timeArray[0:simTime:deltaPlot],MVArray.T[0:simTime:deltaPlot,0],'r', linewidth=4, label ='Impl.')
axs[1].plot(timeArray[0:simTime:deltaPlot],MVArray.T[0:simTime:deltaPlot,2],'r', linewidth=4, label ='Impl.')
axs[2].plot(timeArray[0:simTime:deltaPlot],MVArray.T[0:simTime:deltaPlot,3],'r', linewidth=4, label ='Impl.')


# # Nominal
# axs[0].hlines(ssOptArray[0]['u'][0],0,100,color='gray', linestyles='dotted', linewidth=2)
# axs[1].hlines(ssOptArray[0]['u'][2],0,100,color='gray', linestyles='dotted', linewidth=2)
# axs[2].hlines(ssOptArray[0]['u'][3],0,100,color='gray', linestyles='dotted', linewidth=2)

# # zStep
# axs[0].hlines(ssOptArray[1]['u'][0],100,simTime/12,color='gray', linestyles='dotted', linewidth=2,label='SS Opt')
# axs[1].hlines(ssOptArray[1]['u'][2],100,simTime/12,color='gray', linestyles='dotted', linewidth=2)
# axs[2].hlines(ssOptArray[1]['u'][3],100,simTime/12,color='gray', linestyles='dotted', linewidth=2)

axs[0].set(ylabel='L') # [kmol/min]    
axs[1].set(ylabel='D') # [kmol/min]  
axs[2].set(xlabel='t [min]', ylabel='B') # [kmol/min]  

axs[0].set_ylim([2.4, 3.2])
# axs[0].set_ylim([2.0, 3.6])
axs[1].set_ylim([0.3, 0.8])
axs[2].set_ylim([0.3, 0.8])

axs[0].legend(loc='best')

#%% 14. Computed Bias
if modelUpdate == 2:
    fig14, axs = plt.subplots(3, sharex=True)
    fig14.suptitle('Computed Bias xFrac (selected)')
    
    for kk in range(1,int(simTime/12 - 1),10):
        time_kk = list(range(int(kk - est['N'] + 1),int(kk + 1)))
        if kk > est['T']*est['execPer']/12:
            axs[0].plot(time_kk,pmHatTrajArray[kk][0,:],'b',alpha=0.5, linewidth=3)
            axs[1].plot(time_kk,pmHatTrajArray[kk][21,:],'b',alpha=0.5, linewidth=3)
            axs[2].plot(time_kk,pmHatTrajArray[kk][40,:],'b',alpha=0.5, linewidth=3)
            
    for ii in range(3):
        axs[ii].minorticks_on()
        axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
        axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
        axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
        axs[ii].grid(b=True, which='major', linestyle='-')
        axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[0].hlines(0,0,simTime/12,color='gray', linestyles='dashed', linewidth=1)
    axs[1].hlines(0,0,simTime/12,color='gray', linestyles='dashed', linewidth=1)
    axs[2].hlines(0,0,simTime/12,color='gray', linestyles='dashed', linewidth=1)
    

    axs[0].set(ylabel='w: x_B') # [kmol/min]    
    axs[1].set(ylabel='w: x_{22}') # [kmol/min]  
    axs[2].set(xlabel='t [min]', ylabel='w: x_D') # [kmol/min]  

#%% 15. Computed Parameters
if modelUpdate == 3:
    fig15, axs = plt.subplots(2, sharex=True)
    fig15.suptitle('Computed Parameters')
    
    for kk in range(1,int(simTime/12 - 1),10):
        time_kk = list(range(int(kk - est['N'] + 1),int(kk)))
        if kk > est['T']*est['execPer']/12:
            axs[0].plot(time_kk,pmHatTrajArray[kk][1,:],'b',alpha=0.5, linewidth=3)
            axs[1].plot(time_kk,pmHatTrajArray[kk][0,:],'b',alpha=0.5, linewidth=3)
            
    for ii in range(2):
        axs[ii].minorticks_on()
        axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
        axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
        axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
        axs[ii].grid(b=True, which='major', linestyle='-')
        axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
    
    axs[0].plot(timeArray[0:simTime:deltaPlot],distArray[3][0:simTime:deltaPlot],'k--',alpha=0.7,linewidth=1,label='True')
    axs[1].plot(timeArray[0:simTime:deltaPlot],distArray[1][0:simTime:deltaPlot],'k--',alpha=0.7,linewidth=1,label='True')
    
    axs[0].set(ylabel='z_f') # [kmol/min]    
    axs[1].set(xlabel='t [min]', ylabel='alpha') # [kmol/min]  

#%% 16. Computed Past States Trajectories   
if modelUpdate == 2 or modelUpdate == 3: 
    fig15, axs = plt.subplots(3, sharex=True)
    fig15.suptitle('Computed States Trajectory (selected)')
    
    if modelUpdate == 2 or modelUpdate == 3:
        for kk in range(1,int(simTime/12 - 1),10):
            time_kk = list(range(int(kk - est['N']),int(kk + 1)))
            if kk > est['T']*est['execPer']/12:
                axs[0].plot(time_kk,xHatTrajArray[kk][0,:],'b',alpha=0.5, linewidth=3)
                axs[1].plot(time_kk,xHatTrajArray[kk][21,:],'b',alpha=0.5, linewidth=3)
                axs[2].plot(time_kk,xHatTrajArray[kk][40,:],'b',alpha=0.5, linewidth=3)
                
        for ii in range(3):
            axs[ii].minorticks_on()
            axs[ii].xaxis.set_major_locator(MultipleLocator(majorR))
            axs[ii].xaxis.set_minor_locator(MultipleLocator(minorR))
            axs[ii].yaxis.set_tick_params(which='minor', bottom=False)
            axs[ii].grid(b=True, which='major', linestyle='-')
            axs[ii].grid(b=True, axis='x',which='minor',linestyle='-', alpha=0.3)
        
        axs[0].plot(timeArray[0:simTime:deltaPlot],statesArray.T[0:simTime:deltaPlot,0],'r', linewidth=3, label ='Impl.')
        axs[1].plot(timeArray[0:simTime:deltaPlot],statesArray.T[0:simTime:deltaPlot,21],'r', linewidth=3, label ='Impl.')
        axs[2].plot(timeArray[0:simTime:deltaPlot],statesArray.T[0:simTime:deltaPlot,40],'r', linewidth=3, label ='Impl.')
            
        axs[0].set(ylabel='x_B') # [kmol/min]    
        axs[1].set(ylabel='x_{22}') # [kmol/min]  
        axs[2].set(xlabel='t [min]', ylabel='x_D') # [kmol/min]  

#%%
# Beeping when simulation ends
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)