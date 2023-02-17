#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# March 2018 (in Matlab); Last revision: 15-Jun-2022
#=======================================================

import numpy as np

# importing  all the
# functions defined in casadi.py
from casadi import *

#=======================================================================
def CreateColumnModel():
    #=======================================================
    # builds dynamic model of a distillation columns in CasADi 
    # separating a stream of A + B
    #=======================================================
    
    # Inputs: par - system parameters
    par = SystemParameters()
    
    #############################
    # Creating CasADi variables #
    #############################
    ## Differential states 
    # mole fractions 
    x_A = MX.sym('x_A',par['NT']) # [molA/mol]

    # Liquid holdup
    M = MX.sym('M',par['NT']) # [kmol] 
    # N.B.: stages numbers from btm to top
    
    ## System inputs 
    # Reflux
    LT = MX.sym('LT',1) # [kmol/min]
    # Boilup
    VB = MX.sym('VB',1) # [kmol/min]
    # Distillate
    D = MX.sym('D',1) # [kmol/min]
    # Bottoms
    B = MX.sym('B',1) # [kmol/min]

    ## Disturbances
    # Feedrate
    F = MX.sym('F',1) # [kmol/min]
    # Feed composition
    zF = MX.sym('zF',1) # [kmol A/kmol total]
    # Feed liquid fraction
    qF = MX.sym('qF',1) # [-]                            
    
    # Parameters 
    # relative volatility (as a function of pressure)
    alpha = MX.sym('alpha',1)  # [-]
    
    # time transformation: CASADI integrates always from 0 to 1 
    # and the USER specifies the time
    # scaling with T.
    T = MX.sym('T',1) # [min]
       
    #############################
    # System Equations          #
    #############################
    # Vapor-liquid equilibria
    y_A = []
    for ii in range(0,par['NT'] - 1):
        y_A.append(alpha*x_A[ii]/(1 + (alpha - 1)*x_A[ii])) # until NT - 2 because condenser is not an eq. stage
    
    # Vapor Flows assuming constant molar flows
    V = []
    for ii in range(0,par['NT'] - 1):
        V.append(VB) # until NT - 2 because condenser is not an eq. stage
        
    for ii in range(par['NF'] - 1,par['NT'] - 1):
        V[ii] = V[ii] + (1-qF)*F
              
    # Liquid flows assuming linearized tray hydraulics with time constant taul
    # Also includes coefficient lambda for effect of vapor flow ("K2-effect")
    L = []
    
    # appending the liquid from the reboiler 
    L.append(B)
    
    # looping through the stages
    for ii in range(1,par['NF']):
        L.append(par['L0b'] + (M[ii] - par['M0'])/par['taul'] + par['lambda']*(V[ii - 1]- par['V0']))
        
    for ii in range(par['NF'],par['NT'] - 1):
        L.append(par['L0'] + (M[ii] - par['M0'])/par['taul'] + par['lambda']*(V[ii - 1]- par['V0t']))
            
    # condenser    
    L.append(LT);
        
    # Time derivatives from  material balances for 
    # 1) total holdup and 2) component holdup
    
    # Column
    dMdt =[]
    dMxdt =[]
    
    # Reboiler (assumed to be an equilibrium stage)
    # stage 1 (or counter 0)
    dMdt.append(L[1] - V[0] - B)
    dMxdt.append(L[1]*x_A[1] - V[0]*y_A[0] - B*x_A[0])
    
    # stage 2 to NT - 1 (or counter 1 to NT - 2)
    for ii in range(1,par['NT'] - 1):
        dMdt.append(L[ii + 1] - L[ii] + V[ii - 1] - V[ii])
        dMxdt.append(L[ii + 1]*x_A[ii + 1] - L[ii]*x_A[ii] + V[ii - 1]*y_A[ii - 1] - V[ii]*y_A[ii])
    
    # Correction for feed at the feed stage
    # The feed is assumed to be mixed into the feed stage
    dMdt[par['NF'] - 1] = dMdt[par['NF'] - 1]  + F
    dMxdt[par['NF'] - 1] = dMxdt[par['NF'] - 1]  + F*zF
    
    # Total condenser (no equilibrium stage)
    # stage NT (or counter NT - 1)
    dMdt.append(V[par['NT'] - 2] - LT - D)
    dMxdt.append(V[par['NT'] - 2]*y_A[par['NT'] - 2] - LT*x_A[par['NT'] - 1] - D*x_A[par['NT'] - 1])
    
    # Compute the derivative for the mole fractions from d(Mx) = x dM + M dx
    dxdt =  []
    for ii in range(0,par['NT']):
       dxdt.append((dMxdt[ii] - x_A[ii]*dMdt[ii])/M[ii])

    # ! end modeling
    
    #############################
    # Creating integrator       #
    #############################
    # Form the DAE system
    rhs = vertcat(*dxdt,*dMdt)
    states = vertcat(x_A,M)
    param = vertcat(LT, VB, D, B, F, zF, qF, alpha, T)
           
    sys_model = {'x':states, 'p':param, 'ode':T*rhs}
    
    # building the integrator
    Integ = integrator('Integ','cvodes',sys_model)
   
    # calculating sensitivities for EKF
    Fx = Integ.factory('Fx_fwd', ['x0','p'], ['jac:xf:x0'])
    Fu = Integ.factory('Fu_fwd', ['x0','p'], ['jac:xf:p'])
          
    return Integ, Fx, Fu, rhs, states, param

#=======================================================================
def CreateColumnModelClosedLoop():
    #=======================================================
    # builds dynamic model of a distillation columns in CasADi (+ controllers)
    # separating a stream of A + B
    #=======================================================
        
    # Inputs: par - system parameters
    par =  SystemParameters()
    
    # controller parameters
    ctrlPar = ControlTuning()
    
    #############################
    # Creating CasADi variables #
    #############################
    ## Differential states 
    # mole fractions 
    x_A = MX.sym('x_A',par['NT']) # [molA/mol]

    # Liquid holdup
    M = MX.sym('M',par['NT']) # [kmol] 
    # N.B.: stages numbers from btm to top
    
    # integral states 
    x_E = MX.sym('x_E',ctrlPar['crtlNum']) # integral error controllers
    
    ## System inputs 
    # Top composition controller setpoint
    xDs = MX.sym('xDs',1) # [-]
    # Boilup
    VB = MX.sym('VB',1) # [kmol/min]
    # Condenser holdup setpoint
    MDs = MX.sym('MDs',1) # [kmol]
    # Reboiler holdup setpoint
    MBs = MX.sym('MBs',1) # [kmol]

    ## Disturbances
    # Feedrate
    F = MX.sym('F',1) # [kmol/min]
    # Feed composition
    zF = MX.sym('zF',1) # [kmol A/kmol total]
    # Feed liquid fraction
    qF = MX.sym('qF',1) # [-]                         
    
    # Parameters 
    # relative volatility (as a function of pressure)
    alpha = MX.sym('alpha',1)  # [-]
    
    # controller parameters
    biasCtrl = MX.sym('biasCtrl',ctrlPar['crtlNum'])  # [kmol/min]
    
    # time transformation: CASADI integrates always from 0 to 1 
    # and the USER specifies the time
    # scaling with T.
    T = MX.sym('T',1) # [min]
       
    #############################
    # Controllers Equations     #
    #############################
    ## REBOILER ##
    # Actual reboiler holdup
    MB = M[0]
    # computing error
    eB = MB - MBs
    # Bottoms flowrate # [kmol/min]
    B = biasCtrl[0] + ctrlPar['KcB']*(eB + x_E[0]/ctrlPar['tauB']) 
    
    ## CONDENSER ##
    # Actual condenser holdup
    MD = M[par['NT'] - 1] 
    # computing error
    eD = MD - MDs
    # Distillate flowrate # [kmol/min]
    D = biasCtrl[1] + ctrlPar['KcD']*(eD  + x_E[1]/ctrlPar['tauD'])         
    
    ## TOP COMP. ##
    # Actual top composition
    xD = x_A[par['NT'] - 1]
    # computing error
    eL = xD - xDs
    # Reflux flow # [kmol/min]
    LT = biasCtrl[2] + ctrlPar['KcL']*(eL + x_E[2]/ctrlPar['tauL'])  
    
    # Computing integral error
    dxEdt = []
    dxEdt.append(eB) # [kmol]
    dxEdt.append(eD) # [kmol]
    dxEdt.append(eL) # [-]
    
    #############################
    # System Equations          #
    #############################    
    # Vapor-liquid equilibria
    y_A = []
    for ii in range(0,par['NT'] - 1):
        y_A.append(alpha*x_A[ii]/(1 + (alpha - 1)*x_A[ii])) # until NT - 2 because condenser is not an eq. stage
    
    # Vapor Flows assuming constant molar flows
    V = []
    for ii in range(0,par['NT'] - 1):
        V.append(VB) # until NT - 2 because condenser is not an eq. stage
        
    for ii in range(par['NF'] - 1,par['NT'] - 1):
        V[ii] = V[ii] + (1-qF)*F
              
    # Liquid flows assuming linearized tray hydraulics with time constant taul
    # Also includes coefficient lambda for effect of vapor flow ("K2-effect")
    L = []
    
    # appending the liquid from the reboiler 
    L.append(B)
    
    # looping through the stages
    for ii in range(1,par['NF']):
        L.append(par['L0b'] + (M[ii] - par['M0'])/par['taul'] + par['lambda']*(V[ii - 1]- par['V0']))
        
    for ii in range(par['NF'],par['NT'] - 1):
        L.append(par['L0'] + (M[ii] - par['M0'])/par['taul'] + par['lambda']*(V[ii - 1]- par['V0t']))
            
    # condenser    
    L.append(LT);
        
    # Time derivatives from  material balances for 
    # 1) total holdup and 2) component holdup
    
    # Column
    dMdt =[]
    dMxdt =[]
    
    # Reboiler (assumed to be an equilibrium stage)
    # stage 1 (or counter 0)
    dMdt.append(L[1] - V[0] - B)
    dMxdt.append(L[1]*x_A[1] - V[0]*y_A[0] - B*x_A[0])
    
    # stage 2 to NT - 1 (or counter 1 to NT - 2)
    for ii in range(1,par['NT'] - 1):
        dMdt.append(L[ii + 1] - L[ii] + V[ii - 1] - V[ii])
        dMxdt.append(L[ii + 1]*x_A[ii + 1] - L[ii]*x_A[ii] + V[ii - 1]*y_A[ii - 1] - V[ii]*y_A[ii])
    
    # Correction for feed at the feed stage
    # The feed is assumed to be mixed into the feed stage
    dMdt[par['NF'] - 1] = dMdt[par['NF'] - 1]  + F
    dMxdt[par['NF'] - 1] = dMxdt[par['NF'] - 1]  + F*zF
    
    # Total condenser (no equilibrium stage)
    # stage NT (or counter NT - 1)
    dMdt.append(V[par['NT'] - 2] - LT - D)
    dMxdt.append(V[par['NT'] - 2]*y_A[par['NT'] - 2] - LT*x_A[par['NT'] - 1] - D*x_A[par['NT'] - 1])
    
    # Compute the derivative for the mole fractions from d(Mx) = x dM + M dx
    dxdt =  []
    for ii in range(0,par['NT']):
       dxdt.append((dMxdt[ii] - x_A[ii]*dMdt[ii])/M[ii])
       
    # ! end modeling
    
    #############################
    # Creating integrator       #
    #############################
    # Form the DAE system
    rhs = vertcat(*dxdt,*dMdt,*dxEdt)
    states = vertcat(x_A,M,x_E)
    param = vertcat(xDs, VB, MDs, MBs, F, zF, qF, alpha, biasCtrl, T)
    
    sys_model = {'x':states, 'p':param, 'ode':T*rhs}
    
    # building the integrator
    Integ = integrator('Integ','cvodes',sys_model)
          
    # calculating sensitivities for EKF
    Fx = Integ.factory('Fx_fwd', ['x0','p'], ['jac:xf:x0'])
    Fu = Integ.factory('Fu_fwd', ['x0','p'], ['jac:xf:p'])
          
    return Integ, Fx, Fu, rhs, states, param

#=======================================================================
def CreateColumnModelSteadyState():
    #=======================================================
    # builds the SS model of a distillation columns in CasADi 
    # separating a stream of A + B 
    # ! NOT USED HERE
    #=======================================================
    
    # Inputs: par - system parameters
    par = SystemParameters()
    
    #############################
    # Creating CasADi variables #
    #############################
    ## Differential states 
    # mole fractions 
    x_A = MX.sym('x_A',par['NT']) # [molA/mol]
    # N.B.: stages numbers from btm to top
    
    ## System inputs 
    # Reflux
    LT = MX.sym('LT',1) # [kmol/min]
    # Boilup
    VB = MX.sym('VB',1) # [kmol/min]
    # Distillate
    D = MX.sym('D',1) # [kmol/min]
    # Bottoms
    B = MX.sym('B',1) # [kmol/min]

    ## Disturbances
    # Feedrate
    F = MX.sym('F',1) # [kmol/min]
    # Feed composition
    zF = MX.sym('zF',1) # [kmol A/kmol total]
    # Feed liquid fraction
    qF = MX.sym('qF',1) # [-]                        
    
    # Parameters 
    # relative volatility (as a function of pressure)
    alpha = MX.sym('alpha',1)  # [-]
    
    #############################
    # System Equations          #
    #############################
    # Vapor-liquid equilibria
    y_A = []
    for ii in range(0,par['NT'] - 1):
        y_A.append(alpha*x_A[ii]/(1 + (alpha - 1)*x_A[ii])) # until NT - 2 because condenser is not an eq. stage
    
    # Material balances
    # Vapor Flows assuming constant molar flows
    V = []
    for ii in range(0,par['NT'] - 1):
        V.append(VB) # until NT - 2 because condenser is not an eq. stage
        
    for ii in range(par['NF'] - 1,par['NT'] - 1):
        V[ii] = V[ii] + (1-qF)*F
          
    # Liquid flows assuming constant molar flows at steady-state (due to vapor flow assumption)
    L = []
    for ii in range(0,par['NT']):
        L.append(LT) # until NT - 2 because condenser is not an eq. stage
            
    for ii in range(0,par['NF']):
        L[ii] = L[ii] + qF*F    
    
    # Column
    dMdt =[]
    # Reboiler (perfect control assumption)
    dMdt.append(L[1] - V[0] - B)
    
    # Condenser (perfect control assumption)
    dMdt.append(V[par['NT'] - 2] - LT - D)
    
    # component balances   
    # Column
    dMxdt =[]
    
    # Reboiler (assumed to be an equilibrium stage)
    # stage 1 (or counter 0)
    dMxdt.append(L[1]*x_A[1] - V[0]*y_A[0] - B*x_A[0])
    
    # stage 2 to NT - 1 (or counter 1 to NT - 2)
    for ii in range(1,par['NT'] - 1):
        dMxdt.append(L[ii + 1]*x_A[ii + 1] - L[ii]*x_A[ii] + V[ii - 1]*y_A[ii - 1] - V[ii]*y_A[ii])
    
    # Correction for feed at the feed stage
    # The feed is assumed to be mixed into the feed stage
    dMxdt[par['NF'] - 1] = dMxdt[par['NF'] - 1]  + F*zF
    
    # Total condenser (no equilibrium stage)
    # stage NT (or counter NT - 1)
    dMxdt.append(V[par['NT'] - 2]*y_A[par['NT'] - 2] - LT*x_A[par['NT'] - 1] - D*x_A[par['NT'] - 1])
    
    # ! end modeling
    
    #############################
    # Creating integrator       #
    #############################
    # Form the DAE system
    rhs = vertcat(*dMxdt,*dMdt)
    states = vertcat(x_A)
    param = vertcat(LT, VB, D, B, F, zF, qF, alpha)
           
    # calculating Jacobians?
          
    return rhs, states, param

#=======================================================================
def MeasurementModel(xArray,TrayNumberCol,LevelNumberCol):
    #=======================================================
    # Correlated mol fractions to tray temperatures and 
    # 'extracts' measurements from state vector
    # ! NOT USED HERE
    #=======================================================
    # Compute column temperatures based on the boiling points of the pure components
    # Crude approximation
    
    # Inputs: par - system parameters
    par =  systemParameters()
    
    # Number of temperature measurements in column
    TCol = len(TrayNumberCol)
    TArray = np.zeros((TCol,1))
    
    for ii in range(TCol):
        TArray[ii] = xArray[TrayNumberCol[ii] - 1]*par['T_bL'] + (1 - xArray[TrayNumberCol[ii] - 1])*par['T_bH']
    
    # Number of level (holdup) measurements in column
    MCol = len(LevelNumberCol)
    MArray = np.zeros((MCol,1))
    
    for ii in range(MCol):
        MArray[ii] = xArray[LevelNumberCol[ii] - 1 + par['NT']]

    # measurements --> temperatures and "level" 
    yk = np.vstack((TArray,MArray))

    return yk

#=======================================================================
def InitialCondition():
    #=======================================================
    # specify initial conditions for open loop model 
    #=======================================================
    # Inputs: par - system parameters
    par =  SystemParameters()
    
    # States
    # mole fractions of A [kmol A/kmol total]
    xA0 = [0.016447,
          0.0233373,
          0.0320404,
          0.0429489,
          0.0564904,
          0.0731004,
          0.0931782,
          0.117022,
          0.144752,
          0.176224,
          0.210973,
          0.24819,
          0.286774,
          0.325451,
          0.362932,
          0.398084,
          0.430052,
          0.458322,
          0.48271,
          0.503302,
          0.520376,
          0.547693,
          0.578158,
          0.611371,
          0.646698,
          0.683301,
          0.720207,
          0.756416,
          0.790996,
          0.823185,
          0.852438,
          0.878452,
          0.901139,
          0.920595,
          0.937037,
          0.950764,
          0.962106,
          0.971398,
          0.978958,
          0.985075,
          0.99]
    
    xA0_arr = np.array(xA0, ndmin=2)
    
    # stage holdup [kmol]
    M0 = 0.5*np.ones((par['NT'],1)) 
    # for more realistic studies: M0_1=10 [kmol] (reboiler) 
    #                         and M0_NT=32.1 [kmol] (condenser)
    
    dx0 = np.concatenate((np.transpose(xA0_arr),M0))
    
    # Inputs
    # Reflux
    LT0 = 2.5669  # [kmol/mim]
    # Boilup
    VB0 = 3.06359    # [kmol/min]
    # Distillate
    D0 = 0.496689          # [kmol/min]
    # Bottom flowrate
    B0 = 0.503311         # [kmol/min]    

    # Feed flowrate                
    F0 = 1.0       # [kmol/min]  1.0 | 1.01                 
    # Feed composition
    zF0 = 0.5        # [kmol/min] 
    # Feed liquid fraction                        
    qF0 = 1.0        # [kmol/min]
    
    u0 = np.vstack((LT0,VB0,D0,B0,F0,zF0,qF0))
        
    # Parameters
    # relative volality
    alpha0 = 1.5  # [-] 
    
    p0 = alpha0
    
    # system measurements
    # y0 = np.vstack((dx0[0],dx0[6],dx0[23],dx0[34],dx0[40],dx0[41],dx0[81])) # molar fraction of internal stages + reboiler and condenser holdups
    y0 = par['HOL'].dot(dx0)
    
    return dx0, y0, u0, p0

#=======================================================================
def InitialConditionCL():
    #=======================================================
    # specify initial conditions for closed loop model
    #=======================================================
    
    # Inputs: par - system parameters
    par =  SystemParameters()
    
    # Controller tuning parameters
    ctrlPar = ControlTuning()
    
    # States
    # mole fractions of A [kmol A/kmol total]
    xA0 = [0.016447,
          0.0233373,
          0.0320404,
          0.0429489,
          0.0564904,
          0.0731004,
          0.0931782,
          0.117022,
          0.144752,
          0.176224,
          0.210973,
          0.24819,
          0.286774,
          0.325451,
          0.362932,
          0.398084,
          0.430052,
          0.458322,
          0.48271,
          0.503302,
          0.520376,
          0.547693,
          0.578158,
          0.611371,
          0.646698,
          0.683301,
          0.720207,
          0.756416,
          0.790996,
          0.823185,
          0.852438,
          0.878452,
          0.901139,
          0.920595,
          0.937037,
          0.950764,
          0.962106,
          0.971398,
          0.978958,
          0.985075,
          0.99]
    
    xA0_arr = np.array(xA0, ndmin=2)
    
    # stage holdup [kmol]
    M0 = 0.5*np.ones((par['NT'],1)) 
    # for more realistic studies: M0_1=10 [kmol] (reboiler) 
    #                         and M0_NT=32.1 [kmol] (condenser)
    
    # integral action [kmol,kmol,fraction]
    xE0 = np.zeros((ctrlPar['crtlNum'],1)) 
    
    dx0 = np.concatenate((np.transpose(xA0_arr),M0,xE0))
    
    # Inputs
    # Reflux - xD (pair) controller SP
    xD_SP = ctrlPar['xDs']  # [-]
    # Boilup
    VB0 = 3.06359           # [kmol/min]
    # Distillate  - MD (pair) controller SP
    MD_SP = ctrlPar['MDs']  # [kmol]
    # Bottoms  - MB (pair) controller SP
    MB_SP = ctrlPar['MBs']  # [kmol]   

    # Feed flowrate                
    F0 = 1.0       # [kmol/min]  1.0 | 1.01                 
    # Feed composition
    zF0 = 0.5        # [kmol/min] 
    # Feed liquid fraction                        
    qF0 = 1.0        # [kmol/min]
    
    u0 = np.vstack((xD_SP,VB0,MD_SP,MB_SP,F0,zF0,qF0))
        
    # Parameters
    # relative volality
    alpha0 = 1.5  # [-] 
    
    p0 = alpha0
    
    # system measurements
    y0 = par['HCL'].dot(dx0) # molar fraction of internal stages + reboiler and condenser holdups

    return dx0, y0, u0, p0
    
#=======================================================================
def SystemParameters():
    #============================================================
    # specify plant parameters and some operation constraints values
    #================================================================
    
    # Number of stages (including reboiler and total condenser)
    NT = 41 
    
    # Location of feed stage (stages are counted from the bottom)
    NF = 21
    
    ################
    # Measurements #
    ################
    # Boiling point of the substances (for MeasurementModel function) - not used now
    T_bL = 341.9
    T_bH = 355.4
    
    # States <-> outputs mapping matrix (selector)
    # original set= dx0[0],dx0[6],dx0[23],dx0[34],dx0[40],dx0[41],dx0[81]
    HOL = np.zeros((7,82))
    HCL = np.zeros((7,85))
    
    # # concentration in some of the trays (temperatures)
    # HOL[0,0] = 1
    # HOL[1,6] = 1
    # HOL[2,23] = 1
    # HOL[3,34] = 1
    # HOL[4,40] = 1
    # # reboiler and condenser holdups are known (levels)
    # HOL[5,41] = 1
    # HOL[6,81] = 1
    
    # # concentration in some of the trays (temperatures)
    # HCL[0,0] = 1
    # HCL[1,6] = 1
    # HCL[2,23] = 1
    # HCL[3,34] = 1
    # HCL[4,40] = 1
    # # reboiler and condenser holdups are known (levels)
    # HCL[5,41] = 1
    # HCL[6,81] = 1
    
    # States <-> outputs mapping matrix (selector)
    # original set= dx0[0],dx0[6],dx0[23],dx0[34],dx0[40],dx0[41],dx0[81]
    HOL = np.zeros((8,82))
    HCL = np.zeros((8,85))
    
    # concentration in some of the trays (temperatures)
    HOL[0,0] = 1
    HOL[1,6] = 1
    HOL[2,12] = 1
    HOL[3,23] = 1
    HOL[4,34] = 1
    HOL[5,40] = 1
    # reboiler and condenser holdups are known (levels)
    HOL[6,41] = 1
    HOL[7,81] = 1
    
    # concentration in some of the trays (temperatures)
    HCL[0,0] = 1
    HCL[1,6] = 1
    HCL[2,12] = 1
    HCL[3,23] = 1
    HCL[4,34] = 1
    HCL[5,40] = 1
    # reboiler and condenser holdups are known (levels)
    HCL[6,41] = 1
    HCL[7,81] = 1
        
    #######################################################################################
    # Data for linearized liquid flow dynamics (does not apply to reboiler and condenser) #
    #######################################################################################
    # time constant for liquid dynamics [min]
    taul = 0.063
    
    # Nominal liquid holdups
    M0 = 0.5
    
    # Nominal feed rate [kmol/min]  	
    F0 = 1
    
    # Nominal fraction of liquid in feed
    qF0 = 1

 	# Nominal reflux flow [kmol/min] 	 
    L0 = 2.5669   

    # Nominal liquid flow below feed [kmol/min]  	
    L0b = L0 + qF0*F0	
    
    # Effect of vapor flow on liquid flow ("K2-effect")
    lamb = 0
	
    # Nominal boilup flow [kmol/min] 
    V0 = 3.06359 
    
    # Nominal vapor flows - only needed if lambda is nonzero 
    V0t = V0 + (1 - qF0)*F0
    
    # sampling time
    T_samp = 5/60          # [min]
    
    par = {'NT':NT, 
           'NF':NF, 
           'taul':taul, 
           'M0':M0, 
           'F0':F0,
           'qF0':qF0,
           'L0':L0, 
           'L0b':L0b, 
           'lambda':lamb, 
           'V0':V0, 
           'V0t':V0t,
           'T_bL':T_bL,
           'T_bH':T_bH,
           'T':T_samp,
           'HOL':HOL,
           'HCL':HCL}
    
    return par

#=======================================================================
def ControlTuning():
    #====================================
    # specify the PI control parameters
    #====================================
    
    # number of active controllers
    crtlNum = 3 
    
    # P-Controller Parameters
    # 1. Reboiler level using B #
    # Proportional Gain
    KcB = 1.0 # trial and error: 1.0 | SIMC = 2.0 | matlab = 0.862,
    # Integral Constant
    tauB = 0.5 # trial and error: 0.5 | SIMC = 2.0 | matlab = 1.7,
    # Nominal holdup 
    MBs = 0.5   # [kmol]
    # Nominal flow
    Bs = 0.503311    # [kmol/min] | 0.5  

    # 2. Condenser level using D #
    # Proportional Gain
    KcD = 1.0 # trial and error: 1.0 | SIMC = 2.0 | matlab = 0.862,
    # Integral Constant
    tauD = 0.5 # trial and error: 0.5 | SIMC = 2.0 | matlab = 1.7,
    # Nominal holdup 
    MDs = 0.5   # [kmol]
    # Nominal flow
    Ds = 0.496689    # [kmol/min]  | 0.5

    # 3. One-point control (xD - L) #
    # Proportional Gain
    KcL = -250 # trial and error: 30 | SIMC = 30, 60 | matlab = 18,
    # Integral Constant
    tauL = 3.3  # trial and error: 10 | SIMC = 13, 13.5 | matlab = 5
    # Nominal concentration 
    xDs = 0.99   # [kmol A/kmol]
    # Nominal reflux
    Ls = 2.5669    # [kmol/min]  
    
    par = {'KcB':KcB, 
           'tauB':tauB, 
           'MBs':MBs, 
           'Bs':Bs, 
           'KcD':KcD,
           'tauD':tauD,
           'MDs':MDs, 
           'Ds':Ds, 
           'KcL':KcL, 
           'tauL':tauL, 
           'xDs':xDs,
           'Ls':Ls,
           'crtlNum':crtlNum}
    
    return par