#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# February 2022; Last revision: 15-06-2022
#=======================================================

import numpy as np

# importing  all the
# functions defined in casadi.py
from casadi import *

# import column model 
from ColumnModelRigorous import *

#=======================================================================    
def EconomicSystemParameters(): 
    #=============================#
    # specify economic parameters #
    #=============================#
    # System parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialConditionCL()
    
    # revenue (price for top and bottom streams + purity)
    alphaR = np.array([1.5,0.2]) # 1.5 | 1.2 | 0.8 | 0.6
    
    # costs associated with 
    # boil up vaporization (steam to reboiler)
    # top vapor condensation (cooling water to condenser)
    # feed costs (no purity associated)
    alphaC = np.array([0.03,0.02,0.05])
    
    alph = np.concatenate((alphaR,-alphaC))
    
    # purity of the top product
    xPurity = 0.99
    
    ########
    # DRTO #
    ########
    
    # execution period
    execPer = 30*12 # = 30*12*par['T'] = 30 [min] 
    
    # dimension of the system variables
    nu = len(u0)
    
    # time horizon (number of integration steps)
    # i.e. time [min] = T*par['T']
    T = 150 # [min] 90 | 100 
    
    # number of finite elements
    N = 25 # 15 | 25 | 30 | 50 
    
    # Degree of interpolating polynomial (orthogonal collocation)
    d = 3
    
    # model time step
    sStep = 12*par['T'] # [min] 
    
    # regularization term
    Qu = np.zeros((nu,nu))
    np.fill_diagonal(Qu, 1) # 1
    
    param = {'alph':alph,
             'T':T,
             'N':N,
             'd':d,
             'Qu':Qu,
             'execPer':execPer,
             'xPurity':xPurity,
             'simStep':sStep}
    
    return param

#=======================================================================
# DYNAMIC ECONOMIC OPTIMIZATION (CLOSED LOOP)
#=======================================================================
def CLDynamicEconomicOptimization(xvar,pvar,xdot):
    #=======================
    # Builds the CLDRTO NLP
    #=======================
    ###########################
    # Initializing parameters #
    ###########################
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # System parameters
    par = SystemParameters()
    
    # controller parameters
    ctrlPar = ControlTuning()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialConditionCL()
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0)
    ntheta = 1 + 3 # len(p0) + 3 biases from the controllers
    nbias = 4  # xD | xB | MD | MB =  xvar[par['NT'] - 1], xvar[0], xvar[par['NT']], xvar[2*par['NT'] - 1],
    # N.B. this is the bias from estimation not from the controller!
    # This bias is used only when the model update strategy 'bias update' is chosen
    # Otherwise it is set to zero
    
    ####################################
    # Building collocation polynomials #
    ####################################
    # Get collocation points
    tau_root = np.append(0, collocation_points(ecoPar['d'], 'legendre'))
    
    # Coefficients of the collocation equation
    C = np.zeros((ecoPar['d'] + 1,ecoPar['d'] + 1))
    
    # Coefficients of the continuity equation
    D = np.zeros(ecoPar['d'] + 1)
    
    # Coefficients of the quadrature function
    B = np.zeros(ecoPar['d'] + 1)
    
    # Construct polynomial basis
    for j in range(ecoPar['d'] + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(ecoPar['d'] + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
    
        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
    
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(ecoPar['d'] + 1):
            C[j,r] = pder(tau_root[r])
    
        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
        
    # Control discretization (size of the finite element)
    #h = par['T']*ecoPar['T']/ecoPar['N'] # --> not that the model sampling time par['T'] may not be 1! We need to take that into account
    h = ecoPar['simStep']*ecoPar['T']/ecoPar['N']
    
    ################################
    # Computing the objective term #
    ################################
    # reconstructing the MVS 
    ## REBOILER ##
    # Actual reboiler holdup
    MB = xvar[par['NT']]
    # computing error
    eB = MB - pvar[3]
    # Bottoms flowrate # [kmol/min]
    B_var = pvar[8] + ctrlPar['KcB']*(eB + xvar[-3]/ctrlPar['tauB']) 
     
    ## CONDENSER ##
    # Actual condenser holdup
    MD = xvar[2*par['NT'] - 1] 
    # computing error
    eD = MD - pvar[2]
    # Distillate flowrate # [kmol/min]
    D_var = pvar[9] + ctrlPar['KcD']*(eD + xvar[-2]/ctrlPar['tauD'])         
     
    ## TOP COMP. ##
    # Actual top composition
    xD = xvar[par['NT'] - 1] 
    # computing error
    eL = xD - pvar[0]
    # Reflux flow # [kmol/min]
    LT_var = pvar[10] + ctrlPar['KcL']*(eL + xvar[-1]/ctrlPar['tauL'])  
    
    # bias (estimation) variable
    b_est_temp = MX.sym('bias_est_temp', nbias)
    temp = vertcat((xvar[par['NT'] - 1] + b_est_temp[0])*D_var,(1 - (xvar[0] + b_est_temp[1]))*B_var,pvar[1],(LT_var + D_var),pvar[4])
    L =  dot(temp,ecoPar['alph'])
            
    # Define the ODE right hand side: continuous time dynamics 
    f = Function('f', [xvar, pvar, b_est_temp], [xdot, L])
    
    # Define a function that computes the MVs 
    f_MV = Function('f_MV', [xvar, pvar], [B_var, D_var, LT_var])
    # N.B.: I don`t add a bias to the controller variables because I am interested in the difference between xvar and pvar (the bias). 
    # Instead of adding a bias to both of the variables, I simply don`t modify them
    
    ##########################################################################
    # preparing the system model and parameters for implementing collocation #
    ##########################################################################
    # Estimated parameters
    thetaHat_var = MX.sym('thetaHat_var', ecoPar['N'])
    
    # Estimated parameters
    bias_est_var = MX.sym('bias_est_var', nbias)
    
    # controller parameters
    bias_contr_var = MX.sym('bias_contr_var',ctrlPar['crtlNum'])  # [-]
    
    # Initial inputs
    Uk_1 = MX.sym('Uk_1', nu)
    
    # Start with an empty NLP
    # variables
    w = []
    
    # constraints
    g = []
    
    # save initial value of the input for calculating input movement constraint
    Um1 = Uk_1
    
    # Initial conditions
    Xk = MX.sym('X0', nx)
    w.append(Xk)

    # Build the objective
    obj = 0

    # Formulate the NLP
    for k in range(ecoPar['N']):
        
        # New NLP variable for the input
        Uk = MX.sym('U_' + str(k),nu)
        w.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(ecoPar['d']):
            Xkj = MX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
    
        # Loop over collocation points
        # extrapolating (only for Legendre)
        Xk_end = D[0]*Xk
        
        for j in range(1,ecoPar['d']+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*Xk
           for r in range(ecoPar['d']): xp = xp + C[r+1,j]*Xc[r]
    
           # Append collocation equations
           temp2 = vertcat(Uk,thetaHat_var[k],bias_contr_var,ecoPar['simStep'])
           fj, qj = f(Xc[j-1],temp2,bias_est_var)          
           g.append(h*fj - xp)
    
           # Add contribution to the end state
           Xk_end = Xk_end + D[j]*Xc[j-1]
           
           # Add contribution to quadrature function
           obj += B[j]*qj*h - mtimes([(Uk - Um1).T,ecoPar['Qu'],(Uk - Um1)])
         
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        w.append(Xk)
    
        # Add equality constraint
        g.append(Xk_end - Xk)
        
        # Add input movement constraints
        g.append(Uk - Um1)

        # Update past input
        Um1 = Uk

        # Add top purity constraint
        g.append(Xk_end[par['NT'] - 1] + bias_est_var[0])
        
        # constraints on the reconstructed MV´s at the end of the interval
        B_k, D_k, LT_k = f_MV(Xk,temp2) 
        
        # Add MV´s constraints
        g.append(LT_k)
        g.append(D_k)
        g.append(B_k)

    # Concatenate vectors
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    prob = {'f':-obj, 'x': w, 'g': g, 'p':vertcat(Uk_1,bias_contr_var,thetaHat_var,bias_est_var)}
    # N.B. Minimization
    
    # Create the solver
    opts = {'ipopt.print_level':5, 
            'print_time':0, 
            'ipopt.max_iter':2000,          
            'ipopt.warm_start_init_point':'yes', # see N.B.1(end of the file)
            'calc_lam_x':True, # see N.B.2
            'ipopt.linear_solver':'mumps'}
            # 'ipopt.warm_start_init_point':'yes',
            #'ipopt.tol':1e-4, 
            #'ipopt.acceptable_tol':100*1e-4, 
            

    solver = nlpsol('solver', 'ipopt', prob, opts)
    
    return solver

#=======================================================================    
def CallCLDRTO(CLDRTOsolver,x0HatCLk,UkCL_1,thetaTraj,zTraj,biasHatk,biasCtrlk,xPurity,sysMeas,w_warm,lam_w_warm,lam_g_warm):
    #=======================================================
    # Initializes and solves the DRTO NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # System Parameters
    par = SystemParameters()
        
    # controller parameters
    ctrlPar = ControlTuning()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialConditionCL()
    
    # bounds
    lbw, ubw, lbg, ubg = SystemEcoDynOptBoundsCL(x0HatCLk,biasHatk,xPurity,sysMeas,zTraj)
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0)
    
    ###################
    # Solving the NLP #
    ###################
    sol = CLDRTOsolver(x0=w_warm, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(UkCL_1,biasCtrlk,thetaTraj,biasHatk), lam_x0=lam_w_warm, lam_g0=lam_g_warm)
    
    # saving the solution
    # primal variables
    w_warm = sol['x'].full() 
    
    # dual variables
    lam_x_warm = sol['lam_x'].full() 
    lam_g_warm = sol['lam_g'].full() 
    
    # objective function
    of_sol = sol['f'].full() 
    
    if CLDRTOsolver.stats()['success']:
        solverSol = 1
    else:
        solverSol = 0 # for all other types of errors
    
    ################################
    # Recovering states and inputs #
    ################################
    # organizing variables
    # example order for d = 3 
    # 0:81    - x0   - (0*nx + 0*nu:1*nx - 1)
    
    # 82:85  - u0   -  (1*nx + 0*nu:1*nx + 1*nu - 1)
    # 86:167 - x00  -  (1*nx + 1*nu:2*nx + 1*nu - 1)
    # 168:249 - x01  - (2*nx + 1*nu:3*nx + 1*nu - 1) 
    # 250:331 - x02  - (3*nx + 1*nu:4*nx + 1*nu - 1)  
    # 332:413 - x1   - (4*nx + 1*nu:5*nx + 1*nu - 1)  per loop: (1 + d)*nx + nu
    
    # 492:573 - u1   - (5*nx + 1*nu:5*nx + 2*nu - 1)
    # 574:655 - x10  - (5*nx + 2*nu:6*nx + 2*nu - 1)
    # 574:655 - x11  - (6*nx + 2*nu:7*nx + 2*nu - 1)
    # 574:655 - x12  - (7*nx + 2*nu:8*nx + 2*nu - 1)
    # 656:737 - x2   - (8*nx + 2*nu:9*nx + 2*nu - 1)
    # ....                              total = N*((1 + d)*nx + nu) + nx
    
    # Preparing for saving the data
    uArrayk = []
    xArrayk = []
    xArrayCol = []
    mvArrayCol = []
    
    # size of the finite element 
    h = ecoPar['simStep']*ecoPar['T']/ecoPar['N']
    # Get collocation points
    tau = np.append(collocation_points(ecoPar['d'], 'legendre'),1)
    timeArrayCol = []

    # x0
    xTemp = w_warm[0:nx]
    xArrayk.append(xTemp)
    xArrayCol.append(xTemp)
    timeArrayCol.append(0.0)
    
    # reconstructing the MVS 
    ## REBOILER ##
    # Reboiler holdup
    MB = xTemp[par['NT']]
    # computing error
    eB = MB - UkCL_1[3]
    # Bottoms flowrate # [kmol/min]
    B = biasCtrlk[0] + ctrlPar['KcB']*(eB + xTemp[-3]/ctrlPar['tauB']) 
     
    ## CONDENSER ##
    # Condenser holdup
    MD = xTemp[2*par['NT'] - 1] 
    # computing error
    eD = MD - UkCL_1[2]
    # Distillate flowrate # [kmol/min]
    D = biasCtrlk[1] + ctrlPar['KcD']*(eD + xTemp[-2]/ctrlPar['tauD'])         
     
    ## TOP COMP. ##
    # Top composition
    xD = xTemp[par['NT'] - 1] 
    # computing error
    eL = xD - UkCL_1[0]
    # Reflux flow # [kmol/min]
    LT = biasCtrlk[2] + ctrlPar['KcL']*(eL + xTemp[-1]/ctrlPar['tauL'])  
    
    # saving values
    MVk_CL = np.vstack((LT,D,B))
    #mvArrayCol.append(MVk_CL)
    
    for ii in range(1,ecoPar['N'] + 1):
        helpvar2 = ecoPar['d'] + 1 
        #inputs
        uTemp = w_warm[(1 + (ii - 1)*helpvar2)*nx + (ii - 1)*nu:(1 + (ii - 1)*helpvar2)*nx + ii*nu]
        uArrayk.append(uTemp)
        
        #xk1
        xTemp2 = w_warm[(4 + (ii - 1)*helpvar2)*nx + ii*nu:(5 + (ii - 1)*helpvar2)*nx + ii*nu]
        xArrayk.append(xTemp2)
           
        #xk1
        for jj in range(4):
            # extracting the collocation point
            xTemp = w_warm[(1 + jj + (ii - 1)*helpvar2)*nx + ii*nu:(2 + jj + (ii - 1)*helpvar2)*nx + ii*nu]
            xArrayCol.append(xTemp)
            
            # reconstructing the MVS 
            ## REBOILER ##
            # Actual reboiler holdup
            MB = xTemp[par['NT']]
            # computing error
            eB = MB - uTemp[3]
            # Bottoms flowrate # [kmol/min]
            B = biasCtrlk[0] + ctrlPar['KcB']*(eB + xTemp[-3]/ctrlPar['tauB']) 
             
            ## CONDENSER ##
            # Actual condenser holdup
            MD = xTemp[2*par['NT'] - 1] 
            # computing error
            eD = MD - uTemp[2]
            # Distillate flowrate # [kmol/min]
            D = biasCtrlk[1] + ctrlPar['KcD']*(eD + xTemp[-2]/ctrlPar['tauD'])         
             
            ## TOP COMP. ##
            # Actual top composition
            xD = xTemp[par['NT'] - 1] 
            # computing error
            eL = xD - uTemp[0]
            # Reflux flow # [kmol/min]
            LT = biasCtrlk[2] + ctrlPar['KcL']*(eL + xTemp[-1]/ctrlPar['tauL'])  
            
            # saving values
            MVk_CL = np.vstack((LT,D,B))
            mvArrayCol.append(MVk_CL)
                        
            # t = t_{ii - 1} + h_i*tau
            timeArrayCol.append((ii - 1)*h + h*tau[jj])
                    
    uArrayk = np.hstack(uArrayk)
    xArrayk = np.hstack(xArrayk)
    xArrayCol = np.hstack(xArrayCol)
    mvArrayCol = np.hstack(mvArrayCol)
    timeArrayCol = np.hstack(timeArrayCol)
    
    return uArrayk, xArrayk,xArrayCol,mvArrayCol,timeArrayCol,of_sol, w_warm, lam_x_warm, lam_g_warm, solverSol

#=======================================================================
def SystemEcoDynOptBoundsCL(x0CLHat,bias0Hat,xPurity,sysMeas,zTraj):
    #============================#
    # specify bounds for MHE NLP # 
    #============================#
    # Initial condition
    dx0, y0, u0, p0 = InitialConditionCL()
    
    # System parameters
    par = SystemParameters()
    
    # controller parameters
    ctrlPar = ControlTuning()
    
    # dimension of the system variables
    nx = len(dx0)
    nbias = len(bias0Hat)
    
    ##########
    # states #
    ##########
    # upper bounds
    x_fraction_LB = np.zeros((int((dx0.size - ctrlPar['crtlNum'])/2),1))
    x_holdup_LB = np.zeros((int((dx0.size - ctrlPar['crtlNum'])/2),1))
    x_E_LB = -100*np.ones((int(ctrlPar['crtlNum']),1))
    xLB = np.concatenate((x_fraction_LB,x_holdup_LB,x_E_LB))
    
    # upper bounds
    x_fraction_UB = np.ones((int((dx0.size - ctrlPar['crtlNum'])/2),1))
    x_holdup_UB = 100*np.ones((int((dx0.size - ctrlPar['crtlNum'])/2),1))
    x_E_UB = 100*np.ones((int(ctrlPar['crtlNum']),1))
    xUB = np.concatenate((x_fraction_UB,x_holdup_UB,x_E_UB))
    
    ##########
    # inputs #
    ##########
    # Lower bounds
    # Reflux
    LT_LB = 2.0  # [kmol/mim]
    # (Reflux - xD) pair setpoint
    xDs_LB = 0.9  # [-]
    # Boilup
    VB_LB = 2.0    # [kmol/min]
    # Distillate
    D_LB = 1e-6  # [kmol/mim] 
    # (Distillate - MD) pair setpoint
    MDs_LB = 0.1  # [kmol] # 1e-6
    # Bottoms
    B_LB =  1e-6   # [kmol/min] 1e-6
    # (Bottoms - MB) pair setpoint
    MBs_LB = 0.1  # [kmol] # 1e-6
    
    #uLB = np.array([LT_LB,VB_LB,D_LB,B_LB,sysMeas[0],sysMeas[1],sysMeas[2]], dtype = np.float64, ndmin=2).T
    uLB = np.array([xDs_LB,VB_LB,MDs_LB,MBs_LB,1e-6,1e-6,1e-6], dtype = np.float64, ndmin=2).T
    MVLB = np.array([LT_LB,D_LB,B_LB], dtype = np.float64, ndmin=2).T
    
    # upper bounds
    # Reflux
    LT_UB = 6.0  # [kmol/mim] 4.5
    # (Reflux - xD) pair setpoint
    xDs_UB = 1.0  # [-]
    # Boilup
    VB_UB = 6.0    # [kmol/min] 4.5
    # Distillate
    D_UB = 3.0  # [kmol/mim]  sysMeas[0]
    # (Distillate - MD) pair setpoint
    MDs_UB = 100.0 # [kmol] 
    # Bottoms
    B_UB = 3.0   # [kmol/min] sysMeas[0]
    # (Bottoms - MB) pair setpoint
    MBs_UB = 100.0  # [kmol] 
    
    #uUB = np.array([LT_UB,VB_UB,D_UB,B_UB,sysMeas[0],sysMeas[1],sysMeas[2]], dtype = np.float64, ndmin=2).T
    uUB = np.array([xDs_UB,VB_UB,MDs_UB,MBs_UB,3.0,1.0001,1.0001], dtype = np.float64, ndmin=2).T
    MVUB = np.array([LT_UB,D_UB,B_UB], dtype = np.float64, ndmin=2).T
    # N.B.: fix F and qF by specifying Delta u = 0. Here the upper and lower bounds assume dummy values
    
    ###################
    # input movements #
    ###################
    # Lower bounds
    # Reflux
    xDs_LT_max = 0.01  # [frac] 0.01
    # Boilup
    D_VB_max = 0.1    # [kmol/min] 0.3
    # Distillate
    MD_s_max = 0.05 # [kmol] 0.05
    # Bottoms
    MB_s_max = 0.05   # [kmol] 0.05
    
    D_max = np.array([xDs_LT_max,D_VB_max,MD_s_max,MB_s_max,0,10,0], dtype = np.float64, ndmin=2).T
    #N.B.: allowed input movement is zero! SysMeas doesn't change from values stored in Uk_1
    # except from zF, which is estimated from data

    ###################
    # bias correction #
    ###################
    # mapping bias to states
    Hbias = np.zeros((nx,nbias))
    Hbias[par['NT'] - 1,0] = 1 # xD
    Hbias[0,1] = 1 # xB
    Hbias[2*par['NT'] - 1,2] = 1 # MD
    Hbias[par['NT'],3] = 1 # MB
    
    ################################
    # Building the bounds for DRTO #
    ################################
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # variables
    lbw = []
    ubw = []
    
    # constraints
    lbg = []
    ubg = []
    
    # "Lift" initial conditions
    lbw.append(x0CLHat)
    ubw.append(x0CLHat)
    
    for k in range(ecoPar['N']):
        
        # Input variables
        temp1 = vertcat(uLB[0],uLB[1],uLB[2],uLB[3],uLB[4],zTraj[k],uLB[6])
        lbw.append(temp1)
        temp2 = vertcat(uUB[0],uUB[1],uUB[2],uUB[3],uUB[4],zTraj[k],uUB[6])
        ubw.append(temp2)
    
        # State at collocation points
        for j in range(ecoPar['d']):
            lbw.append(xLB - Hbias.dot(bias0Hat))
            ubw.append(xUB - Hbias.dot(bias0Hat))
            #lbw.append(xLB)
            #ubw.append(xUB)
            
        # Loop over collocation points
        for j in range(1,ecoPar['d'] + 1):
           # Append collocation equations        
           lbg.append(np.zeros((len(dx0),1)))
           ubg.append(np.zeros((len(dx0),1)))
        
        # New NLP variable for state at end of interval
        lbw.append(xLB - Hbias.dot(bias0Hat))
        ubw.append(xUB - Hbias.dot(bias0Hat))
        # lbw.append(xLB)
        # ubw.append(xUB)
            
        # Add equality constraint
        lbg.append(np.zeros((len(dx0),1)))
        ubg.append(np.zeros((len(dx0),1)))
        
        # Add input movement constraints
        lbg.append(-D_max)
        ubg.append(D_max)
        
        # Add top purity constraint 
        lbg.append(np.array(xPurity, dtype = np.float64, ndmin=2))
        ubg.append(np.array(1, dtype = np.float64, ndmin=2))
        
        # Add MV´s constraints
        lbg.append(MVLB)
        ubg.append(MVUB)
               

    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)
    
    return lbw, ubw, lbg, ubg   

#=======================================================================
def CLDRTOGuessInitialization(x0CLHat,uCLk):
    #######################################################
    # Building initial guess for the first DRTO iteration #
    #######################################################
    # Initial condition
    dx0, y0, u0, p0 = InitialConditionCL()
    
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # controller parameters
    ctrlPar = ControlTuning()
    
    w0 = []
    lam_w0 = []
    lam_g0 = []
    
    # Initial conditions (lifted)
    w0.append(x0CLHat)
    lam_w0.append(np.ones((len(dx0),1)))
    
    # Formulate the NLP
    for k in range(ecoPar['N']):
    
        # New NLP variable for the process noise
        w0.append(uCLk)
        lam_w0.append(np.ones((len(u0),1)))
        
        # State at collocation points
        for j in range(ecoPar['d']):
            w0.append(x0CLHat)
            lam_w0.append(np.ones((len(dx0),1)))
            
        # Loop over collocation points
        for j in range(1,ecoPar['d'] + 1):
           # Append collocation equations        
           lam_g0.append(np.ones((len(dx0),1)))
    
        # New NLP variable for state at end of interval
        w0.append(x0CLHat)
        lam_w0.append(np.ones((len(dx0),1)))
    
        # Add equality constraint
        lam_g0.append(np.ones((len(dx0),1)))
        
        # Add equality constraint
        lam_g0.append(np.ones((len(u0),1)))
        
        # Add top purity constraint
        lam_g0.append(np.ones((1,1)))
        
        # Add MV´s constraints
        lam_g0.append(np.ones((ctrlPar['crtlNum'],1)))
        
    # Concatenate vectors
    w0 = np.concatenate(w0)
    
    # Initializing multipliers            
    lam_w0 = np.concatenate(lam_w0)
    lam_g0 = np.concatenate(lam_g0)
    
    return w0, lam_w0, lam_g0   

#=======================================================================
# DYNAMIC ECONOMIC OPTIMIZATION (OPEN LOOP)
#=======================================================================
def DynamicEconomicOptimization(xvar,pvar,xdot):
    #=======================================================
    # Builds the MHE NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # System parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0)
    ntheta = 1 # len(p0)
    nbias = 4 # xD | xB | MD | MB =  xvar[par['NT'] - 1], xvar[0], xvar[par['NT']], xvar[2*par['NT'] - 1],
    # N.B. this is the bias from estimation not from the controller!
    # This bias is used only when the model update strategy 'bias update' is chosen
    # Otherwise it is set to zero
    
    ####################################
    # Building collocation polynomials #
    ####################################
    # Get collocation points
    tau_root = np.append(0, collocation_points(ecoPar['d'], 'legendre'))
    
    # Coefficients of the collocation equation
    C = np.zeros((ecoPar['d'] + 1,ecoPar['d'] + 1))
    
    # Coefficients of the continuity equation
    D = np.zeros(ecoPar['d'] + 1)
    
    # Coefficients of the quadrature function
    B = np.zeros(ecoPar['d'] + 1)
    
    # Construct polynomial basis
    for j in range(ecoPar['d'] + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(ecoPar['d'] + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
    
        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
    
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(ecoPar['d'] + 1):
            C[j,r] = pder(tau_root[r])
    
        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
        
    # Control discretization (size of the finite element)
    #h = par['T']*ecoPar['T']/ecoPar['N'] # --> not that the model sampling time par['T'] may not be 1! We need to take that into account
    h = ecoPar['simStep']*ecoPar['T']/ecoPar['N']
    
    ##########################################################################
    # preparing the system model and parameters for implementing collocation #
    ##########################################################################
    # bias (estimation) variable
    b_est_temp = MX.sym('bias_est_temp', nbias)
    
    # Objective term
    temp = vertcat((xvar[par['NT'] - 1] + b_est_temp[0])*pvar[2],(1 - (xvar[0]+ b_est_temp[1]))*pvar[3],pvar[1],(pvar[0] + pvar[2]),pvar[4])
    L =  dot(temp,ecoPar['alph'])
            
    # Define the ODE right hand side: continuous time dynamics 
    f = Function('f', [xvar, pvar,b_est_temp], [xdot, L])
    
    ##########################################################################
    # preparing the system model and parameters for implementing collocation #
    ##########################################################################
    # Estimated parameters
    thetaHat_var = MX.sym('thetaHat_var', ecoPar['N'])
    
    # Estimated parameters
    bias_est_var = MX.sym('bias_est_var', nbias)
     
    # Initial inputs
    Uk_1 = MX.sym('Uk_1', nu)
    
    # Start with an empty NLP
    # variables
    w = []
    
    # constraints
    g = []
    
    # save initial value of the input for calculating input movement constraint
    Um1 = Uk_1
    
    # Initial conditions
    Xk = MX.sym('X0', nx)
    w.append(Xk)

    # Build the objective
    obj = 0

    # Formulate the NLP
    for k in range(ecoPar['N']):
        
        # New NLP variable for the process noise
        Uk = MX.sym('U_' + str(k),nu)
        w.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(ecoPar['d']):
            Xkj = MX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
    
        # Loop over collocation points
        # extrapolating (only for Legendre)
        Xk_end = D[0]*Xk
        
        for j in range(1,ecoPar['d']+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*Xk
           for r in range(ecoPar['d']): xp = xp + C[r+1,j]*Xc[r]
    
           # Append collocation equations
           temp2 = vertcat(Uk,thetaHat_var[k],ecoPar['simStep'])
           fj, qj = f(Xc[j-1],temp2,bias_est_var)         
           g.append(h*fj - xp)
    
           # Add contribution to the end state
           Xk_end = Xk_end + D[j]*Xc[j-1]
           
           # Add contribution to quadrature function
           obj += B[j]*qj*h - mtimes([(Uk - Um1).T,ecoPar['Qu'],(Uk - Um1)])
         
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        w.append(Xk)
    
        # Add equality constraint
        g.append(Xk_end - Xk)
        
        # Add input movement constraints
        g.append(Uk - Um1)

        # Update past input
        Um1 = Uk

        # Add top purity constraint
        g.append(Xk_end[par['NT'] - 1] + bias_est_var[0])

    # Concatenate vectors
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    prob = {'f':-obj, 'x': w, 'g': g, 'p':vertcat(Uk_1,thetaHat_var,bias_est_var)}
    # N.B. Minimization
    
    # Create the solver
    opts = {'ipopt.print_level':5, 
            'print_time':0, 
            'ipopt.max_iter':500,  # 2000         
            'ipopt.warm_start_init_point':'yes', 
            'calc_lam_x':True,
            'ipopt.linear_solver':'mumps'}
            # 'ipopt.warm_start_init_point':'yes',
            #'ipopt.tol':1e-4, 
            #'ipopt.acceptable_tol':100*1e-4, 
            
    solver = nlpsol('solver', 'ipopt', prob, opts)
    
    return solver

#=======================================================================    
def CallDRTO(DRTOsolver,x0Hatk,Uk_1,thetaTraj,zTraj,biasHatk,xPurity,sysMeas,w_warm,lam_w_warm,lam_g_warm):
    #=======================================================
    # Initializes and solves the DRTO NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # System Parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # bounds
    lbw, ubw, lbg, ubg = SystemEcoDynOptBounds(x0Hatk,biasHatk,xPurity,sysMeas,zTraj)
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0)
    
    ###################
    # Solving the NLP #
    ###################
    sol = DRTOsolver(x0=w_warm, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(Uk_1,thetaTraj,biasHatk), lam_x0=lam_w_warm, lam_g0=lam_g_warm)
    
    # saving the solution
    # primal variables
    w_warm = sol['x'].full() 
    
    # dual variables
    lam_x_warm = sol['lam_x'].full() 
    lam_g_warm = sol['lam_g'].full() 
    
    # objective function
    of_sol = sol['f'].full() 
    
    if DRTOsolver.stats()['success']:
        solverSol = 1
    else:
        solverSol = 0 # for all other types of errors
    
    ################################
    # Recovering states and inputs #
    ################################
    # example order for d = 3 
    # 0:81    - x0   - (0*nx + 0*nu:1*nx - 1)
    
    # 82:85  - u0   -  (1*nx + 0*nu:1*nx + 1*nu - 1)
    # 86:167 - x00  -  (1*nx + 1*nu:2*nx + 1*nu - 1)
    # 168:249 - x01  - (2*nx + 1*nu:3*nx + 1*nu - 1) 
    # 250:331 - x02  - (3*nx + 1*nu:4*nx + 1*nu - 1)  
    # 332:413 - x1   - (4*nx + 1*nu:5*nx + 1*nu - 1)  per loop: (1 + d)*nx + nu
    
    # 492:573 - u1   - (5*nx + 1*nu:5*nx + 2*nu - 1)
    # 574:655 - x10  - (5*nx + 2*nu:6*nx + 2*nu - 1)
    # 574:655 - x11  - (6*nx + 2*nu:7*nx + 2*nu - 1)
    # 574:655 - x12  - (7*nx + 2*nu:8*nx + 2*nu - 1)
    # 656:737 - x2   - (8*nx + 2*nu:9*nx + 2*nu - 1)
    # ....                              total = N*((1 + d)*nx + nu) + nx
    
    # 
    
    # Preparing for saving the data
    uArrayk = []
    xArrayk = []
    xArrayCol = []
    mvArrayCol = []
    
    # size of the finite element 
    h = ecoPar['simStep']*ecoPar['T']/ecoPar['N']
    # Get collocation points
    tau = np.append(collocation_points(ecoPar['d'], 'legendre'),1)
    timeArrayCol = []
    
    # initial values
    xTemp = w_warm[0:nx]
    xArrayk.append(xTemp)
    xArrayCol.append(xTemp)
    timeArrayCol.append(0.0)
    #mvArrayCol.append(np.vstack((u0[0],u0[2],u0[3])))
    
    for ii in range(1,ecoPar['N'] + 1):
        helpvar2 = ecoPar['d'] + 1 
        #inputs
        uTemp = w_warm[(1 + (ii - 1)*helpvar2)*nx + (ii - 1)*nu:(1 + (ii - 1)*helpvar2)*nx + ii*nu]
        uArrayk.append(uTemp)
        
        #xk1
        xTemp2 = w_warm[(4 + (ii - 1)*helpvar2)*nx + ii*nu:(5 + (ii - 1)*helpvar2)*nx + ii*nu]
        xArrayk.append(xTemp2)
           
        #xk1
        for jj in range(4):
            # extracting the collocation point
            xTemp = w_warm[(1 + jj + (ii - 1)*helpvar2)*nx + ii*nu:(2 + jj + (ii - 1)*helpvar2)*nx + ii*nu]
            xArrayCol.append(xTemp)
            
            # reconstructing the MVS -- > repeating to match dimension
            mvArrayCol.append(np.vstack((uTemp[0],uTemp[2],uTemp[3])))
                        
            # t = t_{ii - 1} + h_i*tau
            timeArrayCol.append((ii - 1)*h + h*tau[jj])
                    
    uArrayk = np.hstack(uArrayk)
    xArrayk = np.hstack(xArrayk)
    xArrayCol = np.hstack(xArrayCol)
    mvArrayCol = np.hstack(mvArrayCol)
    timeArrayCol = np.hstack(timeArrayCol)
    
    return uArrayk, xArrayk,xArrayCol,mvArrayCol,timeArrayCol, of_sol, w_warm, lam_x_warm, lam_g_warm, solverSol

#=======================================================================
def SystemEcoDynOptBounds(x0Hat,bias0Hat,xPurity,sysMeas,zTraj):
    #============================#
    # specify bounds for MHE NLP # 
    #============================#
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
        
    # System parameters
    par = SystemParameters()
    
    # dimension of the system variables
    nx = len(dx0)
    nbias = len(bias0Hat)
    
    ##########
    # states #
    ##########
    # Lower bounds
    xLB = np.zeros((len(dx0),1))
    
    # upper bounds
    x_fraction_UB = np.ones((int(dx0.size/2),1))
    x_holdup_UB = 100*np.ones((int(dx0.size/2),1))
    xUB = np.concatenate((x_fraction_UB,x_holdup_UB))
    
    ##########
    # inputs #
    ##########
    # Lower bounds
    # Reflux
    LT_LB = 2.0  # [kmol/mim]
    # Boilup
    VB_LB = 2.0    # [kmol/min]
    # Distillate
    D_LB = 1e-6  # [kmol/mim]
    # Bottoms
    B_LB = 1e-6   # [kmol/min]
    
    uLB = np.array([LT_LB,VB_LB,D_LB,B_LB,1e-6,1e-6,1e-6], dtype = np.float64, ndmin=2).T
    
    # upper bounds
    # Reflux
    LT_UB = 6.0  # [kmol/mim] 4.5
    # Boilup
    VB_UB = 6.0    # [kmol/min] 4.5
    # Distillate
    D_UB = 3.0  # [kmol/mim]  sysMeas[0]
    # Bottoms
    B_UB = 3.0   # [kmol/min] sysMeas[0]
    
    uUB = np.array([LT_UB,VB_UB,D_UB,B_UB,3.0,1.0001,1.0001], dtype = np.float64, ndmin=2).T
    # N.B.: fix F, zF and qF by specifying Delta u = 0. Here the upper and lower bounds assume dummy values
    
    ###################
    # input movements #
    ###################
    # Lower bounds
    # Reflux
    D_LT_max = 0.3  # [kmol/mim] 0.3
    # Boilup
    D_VB_max = 0.3    # [kmol/min] 0.3 | 0,1
    # Distillate
    D_D_max = 0.1  # [kmol/mim] 0.3 | 0.1
    # Bottoms
    D_B_max = 0.1   # [kmol/min] 0.3 | 0.1
    
    D_max = np.array([D_LT_max,D_VB_max,D_D_max,D_B_max,0.0,10.0,0.0], dtype = np.float64, ndmin=2).T
    #D_max = np.array([D_LT_max,D_VB_max,D_D_max,D_B_max,0.0,0.0,0.0], dtype = np.float64, ndmin=2).T
    #N.B.: allowed input movement is zero! SysMeas doesn't change from values stored in Uk_1

    ###################
    # bias correction #
    ###################
    # mapping bias to states
    Hbias = np.zeros((nx,nbias))
    Hbias[par['NT'] - 1,0] = 1 # xD
    Hbias[0,1] = 1 # xB
    Hbias[2*par['NT'] - 1,2] = 1 # MD
    Hbias[par['NT'],3] = 1 # MB

    ################################
    # Building the bounds for DRTO #
    ################################
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # variables
    lbw = []
    ubw = []
    
    # constraints
    lbg = []
    ubg = []
    
    # "Lift" initial conditions
    lbw.append(x0Hat)
    ubw.append(x0Hat)
    
    for k in range(ecoPar['N']):
        
        # New NLP variable for the process noise
        temp1 = vertcat(uLB[0],uLB[1],uLB[2],uLB[3],uLB[4],zTraj[k],uLB[6])
        #temp1 = vertcat(uLB)
        lbw.append(temp1)
        temp2 = vertcat(uUB[0],uUB[1],uUB[2],uUB[3],uUB[4],zTraj[k],uUB[6])
        #temp2 = vertcat(uUB)
        ubw.append(temp2)
        
        # State at collocation points
        for j in range(ecoPar['d']):
            lbw.append(xLB - Hbias.dot(bias0Hat))
            ubw.append(xUB - Hbias.dot(bias0Hat))
    
        # Loop over collocation points
        for j in range(1,ecoPar['d'] + 1):
           # Append collocation equations        
           lbg.append(np.zeros((len(dx0),1)))
           ubg.append(np.zeros((len(dx0),1)))
        
        # New NLP variable for state at end of interval
        lbw.append(xLB - Hbias.dot(bias0Hat))
        ubw.append(xUB - Hbias.dot(bias0Hat))
    
        # Add equality constraint
        lbg.append(np.zeros((len(dx0),1)))
        ubg.append(np.zeros((len(dx0),1)))
        
        # Add input movement constraints
        lbg.append(-D_max)
        ubg.append(D_max)
        
        # Add top purity constraint 
        lbg.append(np.array(xPurity, dtype = np.float64, ndmin=2))
        ubg.append(np.array(1, dtype = np.float64, ndmin=2)) # no upper bound

    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)
    
    return lbw, ubw, lbg, ubg   

#=======================================================================
def DRTOGuessInitialization(x0Hat,uk):
    #######################################################
    # Building initial guess for the first DRTO iteration #
    #######################################################
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    w0 = []
    lam_w0 = []
    lam_g0 = []
    
    # Initial conditions (lifted)
    w0.append(x0Hat)
    lam_w0.append(np.ones((len(dx0),1)))
    
    # Formulate the NLP
    for k in range(ecoPar['N']):
    
        # New NLP variable for the process noise
        w0.append(uk)
        lam_w0.append(np.ones((len(u0),1)))
        
        # State at collocation points
        for j in range(ecoPar['d']):
            w0.append(x0Hat)
            lam_w0.append(np.ones((len(dx0),1)))
            
        # Loop over collocation points
        for j in range(1,ecoPar['d'] + 1):
           # Append collocation equations        
           lam_g0.append(np.ones((len(dx0),1)))
    
        # New NLP variable for state at end of interval
        w0.append(x0Hat)
        lam_w0.append(np.ones((len(dx0),1)))
    
        # Add equality constraint
        lam_g0.append(np.ones((len(dx0),1)))
        
        # Add equality constraint
        lam_g0.append(np.ones((len(u0),1)))
        
        # Add top purity constraint
        lam_g0.append(np.ones((1,1)))
        
    # Concatenate vectors
    w0 = np.concatenate(w0)
    
    # Initializing multipliers            
    lam_w0 = np.concatenate(lam_w0)
    lam_g0 = np.concatenate(lam_g0)
    
    return w0, lam_w0, lam_g0  

#=======================================================================
# STEADY-STATE ECONOMIC OPTIMIZATION (NOT USED HERE!)
#=======================================================================
def SSEconomicOptimization(xvar,pvar,xdot):
    #=======================================================
    # Builds the Economic Optimization NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################   
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
          
    # System Parameters
    par = SystemParameters()
    
    ########################
    # Optimization problem #   
    ########################
    # decision variables
    # w = vertcat(xvar,pvar[0:4])
    w = vertcat(xvar,pvar[0],pvar[1],pvar[2],pvar[3])
    # system parameters (feed + estimated parameters)
    sysPar = vertcat(pvar[4],pvar[5],pvar[6],pvar[7])  
    
    # constraints 
    g = []
    # system model 
    g.append(xdot)
    # add extra constraint (purity)
    g.append(xvar[par['NT'] - 1]) # top concentration min level
    # preparing constraints
    g = vertcat(*g)
    
    # objective function (max. profit)
    temp = vertcat(xvar[par['NT'] - 1]*pvar[2],(1 - xvar[0])*pvar[3],pvar[1],(pvar[0] + pvar[2]),pvar[4])
    
    J =  dot(temp,ecoPar['alph'])

    #formalize it into an NLP problem
    prob = {'f': -J, 'x': w, 'g': g, 'p':sysPar}

    # Create the solver
    opts = {'ipopt.print_level':5, 
            'print_time':0, 
            'ipopt.max_iter':500} 
            #'ipopt.linear_solver':'mumps',
            #'ipopt.tol':1e-4, 
            #'ipopt.acceptable_tol':100*1e-4, 
            
    solver = nlpsol('solver', 'ipopt', prob, opts)

    return solver

#=======================================================================    
def CallSSSolver(SSOptSolver,xGuess,uGuess,thetaHat,sysMeas,xPurity):
    #=======================================================
    # Solves the steady-state economic NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # System Parameters
    par = SystemParameters()
    
    # Econominc parameters
    ecoPar = EconomicSystemParameters()
    
    # bounds
    lbw, ubw, lbg, ubg = SystemEcoSSOptBounds(sysMeas,xPurity)

    ###########################
    # Initializing parameters #
    ###########################
    wk = np.concatenate((xGuess,uGuess))
    temp = np.array(thetaHat, ndmin=2)
    pk = np.concatenate((sysMeas, temp.T))
    
    # Solve
    sol = SSOptSolver(x0=wk, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=pk)
    
    # saving the solution
    temp = sol['x'].full() 
    # extract the solution
    xOptk = temp[0:par['NT']] 
    uOptk = temp[par['NT']:] 
    
    # objective function
    JOptk = -sol['f'].full()
    
    if SSOptSolver.stats()['success']:
        solverSol = 1
    else:
        solverSol = 0 # for all other types of errors
    
    return uOptk, xOptk, JOptk, solverSol  

#=======================================================================
def SystemEcoSSOptBounds(sysMeas,xPurity):
    # sysMeas = L, B, F, zF, qF
    #==========================================#
    # specify bounds for Economic Optimization # 
    #==========================================#
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    ##########
    # states #
    ##########
    # Lower bounds
    x_LB = np.zeros((int(dx0.size/2),1)) #x_fraction_LB
    
    # upper bounds
    x_UB = np.ones((int(dx0.size/2),1)) #x_fraction_UB
    
    ##########
    # inputs #
    ##########
    # Lower bounds
    # Reflux
    LT_LB = 2.0  # [kmol/mim]
    # Boilup
    VB_LB = 2.0    # [kmol/min]
    # Distillate
    D_LB = 1e-6  # [kmol/mim]
    # Bottoms
    B_LB = 1e-6   # [kmol/min]
    
    u_LB = np.array([LT_LB,VB_LB,D_LB,B_LB], dtype = np.float64, ndmin=2)
    
    # upper bounds
    # Reflux
    LT_UB = 6.0  # [kmol/mim]
    # Boilup
    VB_UB = 6.0    # [kmol/min]
    # Distillate
    D_UB = 3.0  # [kmol/mim] sysMeas[0] 
    # Bottoms
    B_UB = 3.0   # [kmol/min] 
    
    u_UB = np.array([LT_UB,VB_UB,D_UB,B_UB], dtype = np.float64, ndmin=2)
    
    ###############
    # constraints #
    ###############
    # Lower bounds
    # system model
    gODE_LB = np.zeros(len(x_LB) + 2) # model: dxdt = f(x) -SS-> 0 = f(x), upper and lower bounds are equal to zero

    # purity constraint
    gOp_LB = xPurity
    
    # Upper bounds
    # system model
    gODE_UB = np.zeros(len(x_UB) + 2)

    # purity constraint
    gOp_UB = 1
    
    lbw = np.concatenate((x_LB,u_LB.T))
    ubw = np.concatenate((x_UB,u_UB.T))
    lbg = np.append(gODE_LB, gOp_LB) 
    ubg = np.append(gODE_UB, gOp_UB)
    
    return lbw, ubw, lbg, ubg   

#######################
# NOTES FROM THE CODE #
#######################

#N.B.1: Warm-starting an interior-point algorithm is an important issue. One of the main difficulties arises
# from the fact that full-space variable information is required to generate the warm-starting point.
# When the user solves the opt problem, Ipopt will only return the optimal values of the primal variables x 
# and of the constraint multipliers corresponding to the active bounds of g(x). If this information is used to 
# solve the same problem again, you will notice that Ipopt will take some iterations in finding the same solution. 
# The reason for this is that we are missing the input information of the multipliers corresponding to the variable bounds
# If the user does not specify some of these values, Ipopt will set these multipliers to 1.0

# In order to make the warm-start effective, the user has control over the following options: 
# - warm start init point
# - warm start bound push
# - warm start mult bound push

# Note, that the use of this feature is far from solving the complicated issue of warm starting interiorpoint algorithms. 
# As a general advice, this feature will be useful if the user observes that the solution of subsequent problems 
# (i.e., for different data instances) preserves the same set of active inequalities and bounds (monitor the values of the z variable bounds). 
# In this case, initializing the bound multipliers and setting warm_start_init_point to `yes` and setting warm_start_bound_push, 
# warm_start_mult_bound_push and mu_init to a small value (10−6 or so) will reduce significantly the number of iterations. 
# This is particularly useful in setting up on-line applications and high-level optimization strategies. 
# If active-set changes are observed between subsequent solutions, then this strategy might not decrease the number of iterations 
# (in some cases, it might even tend to increase the number of iterations).

# You might also want to try the adaptive barrier update 
# (instead of the default monotone one where above we chose the initial value 10−6) when doing the warm start
# This can be activated by setting the mu_strategy option to `adaptive`. Also the option mu oracle gives some alternative choices. 
# In general, the adaptive choice often leads to less iterations, but the computational cost per iteration might be higher
# from: https://projects.coin-or.org/Ipopt/browser/stable/3.11/Ipopt/doc/documentation.pdf?format=raw 

#N.B.2: the calculation of the multipliers for the simple bounds (lam_x) in IPOPT is not always reliable
# If you set the option "calc_lam_x" to True for the Ipopt instance in CasADi, 
# the multipliers will instead be calculated by CasADi, from the KKT conditions (From Joel \ google groups)
# from: https://groups.google.com/g/casadi-users/c/fcjt-JX5BIc/m/wBImaJPcAQAJ
