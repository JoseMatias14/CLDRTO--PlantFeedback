#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# February 2022
#=======================================================

import numpy as np
import pickle

# importing  all the
# functions defined in casadi.py
from casadi import *

# import column model 
from ColumnModelRigorous import *

#=======================================================================    
def EstimationParameters(): 
    #===========================================#
    # specify estimator parameters (EKF and MHE #
    #===========================================#
    
    # System parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # number of parameters to be estimated
    npar = 2
    
    # States <-> outputs mapping matrix (selector)
    H = par['HOL']
    #######################
    # Bias Update  Tuning #
    #######################
    # first-order filter gain
    lambdaBias = 0.25 # 0.3 | 0.7
    
    #######################
    # EKF (states) Tuning #
    #######################
    # Estimate Covariance Guess
    P0 = np.zeros((len(dx0),len(dx0)))
    np.fill_diagonal(P0, 0.1) 

    # Process Noise Variance
    Q = np.zeros((len(dx0),len(dx0)))
    #np.fill_diagonal(Q, 0.1) # Perfect model  (NO Controllers) | 1
    np.fill_diagonal(Q, 0.1)
    
    # Measurement Noise Variance
    R = np.zeros((y0.size,y0.size)) # number of measurements
    np.fill_diagonal(R, 0.01) # no measurement noise  0.01
    
    ######################################
    # EKF (states and parameters) Tuning #
    ######################################   
    # Extended covariance guess
    Pe0 = np.zeros((len(dx0) + npar,len(dx0) + npar))
    np.fill_diagonal(Pe0, 0.1) 
    
    # Process Noise Variance (extended vector)
    Qe = np.zeros((len(dx0) + npar,len(dx0) + npar))
    np.fill_diagonal(Qe,0.1) # Perfect model  (NO Controllers)
    # changing parameter variance
    Qe[len(dx0),len(dx0)] = 0.1 # 0.1
    Qe[len(dx0) + 1,len(dx0) + 1] = 0.01 
    
    ####################################
    # Moving Horizon Estimatior Tuning #
    ####################################
    # Process Noise Variance
    Qmhe = np.zeros((len(dx0),len(dx0)))
    np.fill_diagonal(Qmhe, 1) # Perfect model  (NO Controllers)
    
    # Parameter Variance
    QthetaMhe = np.zeros((npar,npar))
    np.fill_diagonal(QthetaMhe, [1e-3, 1e-4])

    # Measurement Noise Variance
    Rmhe = np.zeros((y0.size,y0.size)) # number of measurements
    np.fill_diagonal(Rmhe, 0.01) # no measurement noise  0.01
    
    #######
    # time horizon (number of integration steps)
    # i.e. time [min] = T*par['T']
    T = 10 # [min] 
    
    # number of finite elements
    N = 10 
    
    # execution period: number of par['T']
    execPer = 12 # = 12*par['T'] = 1 [min]
    
    # integration step of the MHE 
    dT = execPer*par['T']
    
    # Degree of interpolating polynomial (orthogonal collocation)
    d = 3
    
    # weights for MHE objective function (only states)
    Q_1 = np.linalg.inv(Q)
    R_1 = np.linalg.inv(R)
    
    # weights for MHE objective function (states and parameters)
    Qe_1 = np.linalg.inv(Qmhe)
    Qthetae_1 = np.linalg.inv(QthetaMhe)
    Re_1 = np.linalg.inv(Rmhe)

    param = {'P0':P0,
           'Q':Q,
           'Pe0':Pe0,
           'Qe':Qe,
           'R':R,
           'Qmhe':Qmhe,
           'QthetaMhe':QthetaMhe, 
           'Rmhe':Rmhe,
           'd':d,
           'T':T,
           'N':N,
           'H':H,
           'dT':dT,
           'execPer':execPer,
           'invQ':Q_1,
           'invR':R_1,
           'invQe':Qe_1,
           'invQetheta':Qthetae_1,
           'invRe':Re_1,
           'lambdaBias':lambdaBias}

    return param

#=======================================================================
# MOVING HORIZON ESTIMATOR --> ONLY STATES
#=======================================================================
def MovingHorizonEstimatorStates(xvar,pvar,xdot):
    #=======================================================
    # Builds the MHE NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Estimators (EKF and MHE) parameters
    mhePar = EstimationParameters()
    
    # System parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0) + 1 + 1 # (+1 for p0 and + 1 for par['T'])
    
    ####################################
    # Building collocation polynomials #
    ####################################
    # Get collocation points
    tau_root = np.append(0, collocation_points(mhePar['d'], 'legendre'))
    
    # Coefficients of the collocation equation
    C = np.zeros((mhePar['d'] + 1,mhePar['d'] + 1))
    
    # Coefficients of the continuity equation
    D = np.zeros(mhePar['d'] + 1)
    
    # Coefficients of the quadrature function
    B = np.zeros(mhePar['d'] + 1)
    
    # Construct polynomial basis
    for j in range(mhePar['d'] + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(mhePar['d'] + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
    
        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
    
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(mhePar['d'] + 1):
            C[j,r] = pder(tau_root[r])
    
        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
        
    # Control discretization (size of the finite element)
    h = mhePar['dT']*mhePar['T']/mhePar['N'] # --> not that the model sampling time par['T'] may not be 1! We need to take that into account

    ##########################################################################
    # preparing the system model and parameters for implementing collocation #
    ##########################################################################
    # Measurements
    Y_in = MX.sym('Y_in', mhePar['N']*ny)
    Y = reshape(Y_in,mhePar['N'],ny)
     
    # Independent variables
    U_in = MX.sym('U_in', mhePar['N']*nu)
    U = reshape(U_in,mhePar['N'],nu)
    
    # Initial state
    X0_var = MX.sym('X0_var', nx)
    
    # Arrival Cost
    P_mat_in = MX.sym('P_mat',nx**2)
    P_mat = reshape(P_mat_in,nx,nx)
        
    # Define the ODE right hand side: continuous time dynamics 
    f = Function('f', [xvar, pvar], [xdot]) 
    
    # Start with an empty NLP
    # variables
    w = []
    
    # constraints
    g = []
    
    # Initial conditions
    Xk = MX.sym('X0', nx)
    w.append(Xk)

    # Build the objective
    obj = 0
    
    # Compute the arrival cost contribution
    obj += mtimes([(Xk - X0_var).T,P_mat,(Xk - X0_var)])
        
    # Formulate the NLP
    for k in range(mhePar['N']):
        
        # New NLP variable for the process noise
        Wk = MX.sym('W_' + str(k),nx)
        w.append(Wk)

        # State at collocation points
        Xc = []
        for j in range(mhePar['d']):
            Xkj = MX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
    
        # Loop over collocation points
        # extrapolating (only necessary for Legendre)
        Xk_end = D[0]*Xk
        
        for j in range(1,mhePar['d']+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*Xk
           for r in range(mhePar['d']): xp = xp + C[r+1,j]*Xc[r]
    
           # Append collocation equations
           fj = f(Xc[j-1],U[k,:].T)         
           g.append(h*fj - xp)
    
           # Add contribution to the end state
           Xk_end = Xk_end + D[j]*Xc[j-1];
        
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        w.append(Xk)
    
        # Add equality constraint
        g.append((Xk_end - Xk) + Wk)

        # add the process model contribution to the OF
        obj += mtimes([Wk.T,mhePar['invQ'],Wk])
        
        # add the measurement noise contribution to the OF
        yHat = mtimes([mhePar['H'],Xk])
        
        # calculate the residual
        vk = yHat - Y[k,:].T
        
        # add contribution of the residual to the OF
        obj += mtimes([vk.T,mhePar['invR'],vk])

    # Concatenate vectors
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    prob = {'f': obj, 'x': w, 'g': g, 'p':vertcat(Y_in, U_in, X0_var, P_mat_in)}
    
    # Create the solver
    opts = {'ipopt.print_level':5, 
            'print_time':0, 
            'ipopt.max_iter':1000, 
            'ipopt.warm_start_init_point':'yes', # see N.B.1(end of the file)
            'calc_lam_x':True, # see N.B.2
            'ipopt.linear_solver':'mumps'}
            #'ipopt.tol':1e-4, 
            #'ipopt.acceptable_tol':100*1e-4, 
            

    solver = nlpsol('solver', 'ipopt', prob, opts)
    
    return solver
 
#=======================================================================    
def CallMovingHorizonEstimatorStates(MHEsolver,xhat0,Y_k,P_k,Pi_k,w_warm,lam_w_warm,lam_g_warm):
    #=======================================================
    # Initializes and solves the MHE NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Estimators (EKF and MHE) parameters
    mhePar = EstimationParameters()
    
    # System Parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # bounds
    lbw, ubw, lbg, ubg = SystemBoundsMHEStates()
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0) + 1 + 1 # (+1 for p0 and + 1 for par['T'])
    
    #####################################
    # Reading current plant information #
    #####################################
    # FILTER PARAMETERS
    # Measurements
    Y_in = np.reshape(Y_k,(mhePar['N']*ny,1))
     
    # Independent variables
    U_in = np.reshape(P_k,(mhePar['N']*nu,1))
    
    # Initial state
    X0_in = xhat0
    
    # Arrival Cost
    Pi_k_in = np.linalg.inv(Pi_k)
    Pi_k_in = np.reshape(Pi_k_in,(nx**2,1))
    
    filter_parameters = np.concatenate((Y_in,U_in,X0_in,Pi_k_in))
        
    ###################
    # Solving the NLP #
    ###################
    sol = MHEsolver(x0=w_warm, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=filter_parameters, lam_x0=lam_w_warm, lam_g0=lam_g_warm)
    
    # saving the solution
    # primal variables
    w_warm = sol['x'].full() 
    # dual variables
    lam_x_warm = sol['lam_x'].full() 
    lam_g_warm = sol['lam_g'].full() 
    
    # objective function
    of_sol = sol['f'].full() 
    
    if MHEsolver.stats()['success']:
        solverSol = 1
    else:
        solverSol = 0 # for all other types of errors
    
    ########################
    # Organizing variables #
    ########################
    # example order for d = 3 
    # 0:81    - x0   - (0*nx:1*nx - 1)
    
    # 82:163  - w0   - (1*nx:2*nx - 1)
    # 164:245 - x00  - (2*nx:3*nx - 1)
    # 246:327 - x01  - (3*nx:4*nx - 1) 
    # 328:409 - x02  - (4*nx:5*nx - 1)  
    # 410:491 - x1   - (5*nx:6*nx - 1)  per loop: (2 + d)*nx
    
    # 492:573 - w1   - (6*nx:7*nx - 1) 
    # 574:655 - x10  - (7*nx:8*nx - 1)
    # 574:655 - x11  - (8*nx:9*nx - 1)
    # 574:655 - x12  - (9*nx:10*nx - 1)
    # 656:737 - x2   - (10*nx:11*nx - 1)
    # ....                              total = N*(2 + d)*nx + nx
    
    # 
    
    # building states (if want to plot in the future)
    wArrayk = []
    xArrayk = []
    
    # x0
    xArrayk.append(w_warm[0:nx])
    
    for ii in range(1,mhePar['N'] + 1):
        helpvar2 = 2 + mhePar['d']
        #wk
        wArrayk.append(w_warm[(1 + (ii - 1)*helpvar2)*nx:(2 + (ii - 1)*helpvar2)*nx])
                
        #xk1
        xArrayk.append(w_warm[(1 + (1 + mhePar['d']) + (ii - 1)*helpvar2)*nx:(2 + (1 + mhePar['d']) + (ii - 1)*helpvar2)*nx])
    
    wArrayk = np.hstack(wArrayk)
    xArrayk = np.hstack(xArrayk)
    xhat_k = np.reshape(xArrayk[:,-1],(nx,1))
    what_k = np.reshape(wArrayk[:,-1],(nx,1))
   
    
    return xhat_k, what_k, of_sol, xArrayk, wArrayk, w_warm, lam_x_warm, lam_g_warm, solverSol

#=======================================================================
def ExtendedKalmanFilterStates(yk,pk_1,xk_1k_1,PNk_1k_1,F,S_xx):
    #===================================================================================#
    # Solves the estimation problem using an EKF (also used for computing arrival cost) #
    #===================================================================================#
    # System parameters
    par = SystemParameters()
    
    # Estimators (EKF and MHE) parameters
    ekf = EstimationParameters()
    
    # ===================================
    #     Integrating results
    # ===================================
    # Evolving plant in time
    Ik = F(x0=xk_1k_1,p=pk_1)
    
    # extracting solution
    xkk_1 = Ik['xf'].full()
    
    # ===================================
    #     Sensitivities
    # ===================================
    # states
    Fk_1 = S_xx(x0=xk_1k_1, p=pk_1)['jac_xf_x0'].full()
    
    # outputs transition (Linear Output Function)
    Hk = ekf['H']
    
    # ===================================
    #     Filter
    # ===================================
    # predicted covariance estimate
    Pkk_1 = Fk_1.dot(PNk_1k_1).dot(Fk_1.transpose()) + ekf['Q']
       # dot or matmul?
       
    # Updating
    # Innovation covariance
    Sk = Hk.dot(Pkk_1).dot(Hk.transpose()) + ekf['R']
    
    # Kalman Gain
    Skinv = np.linalg.inv(Sk)
    Kk = Pkk_1.dot(Hk.transpose()).dot(Skinv)
    
    # Updating covariance estimate
    # Trick to guarantee that the matrix is symmetric
    temp = (np.identity(len(xk_1k_1)) - np.dot(Kk,Hk))
    Pkk = temp.dot(Pkk_1).dot(temp.transpose()) + Kk.dot(ekf['R']).dot(Kk.transpose()) # Joseph form (posteriori estimate of the covariance matrix)
    
    temp2 = Pkk
    for ii in range(np.shape(temp2)[0]):
        for jj in range(ii + 1, np.shape(temp2)[0]):
            temp2[jj,ii] = Pkk[ii,jj]
    
    Pkk = temp2
    
    # Evaluating model outputs at k|k-1
    yHat = Hk.dot(xkk_1)
    
    # Update state estimate
    xHatkk = xkk_1 + Kk.dot(yk - yHat)
    
    return xHatkk, Pkk

#=======================================================================
def MHEGuessInitializationStates(xArrayk,wArrayk):
    ######################################################
    # Building initial guess for the first MHE iteration #
    ######################################################
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # parameters
    mhePar = EstimationParameters()
    
    w0 = []
    lam_w0 = []
    lam_g0 = []
    
    # Initial conditions
    w0.append(xArrayk[:,0])
    lam_w0.append(np.ones((len(dx0),1)))
    
    # Formulate the NLP
    for k in range(mhePar['N']):
    
        # New NLP variable for the process noise
        w0.append(wArrayk[:,k])
        lam_w0.append(np.ones((len(dx0),1)))
        
        # State at collocation points
        for j in range(mhePar['d']):
            w0.append(xArrayk[:,k + 1])
            lam_w0.append(np.ones((len(dx0),1)))
            
        # Loop over collocation points
        for j in range(1,mhePar['d'] + 1):
           # Append collocation equations        
           lam_g0.append(np.ones((len(dx0),1)))
    
        # New NLP variable for state at end of interval
        w0.append(xArrayk[:,k + 1])
        lam_w0.append(np.ones((len(dx0),1)))
    
        # Add equality constraint
        lam_g0.append(np.ones((len(dx0),1)))
        
    # Concatenate vectors
    w0 = np.concatenate(w0)
    
    # Initializing multipliers            
    lam_w0 = np.concatenate(lam_w0)
    lam_g0 = np.concatenate(lam_g0)
    
    return w0, lam_w0, lam_g0   

#=======================================================================
def SystemBoundsMHEStates():
    #============================#
    # specify bounds for MHE NLP # 
    #============================#
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    ##########
    # states #
    ##########
    # Lower bounds
    xLB = np.zeros((len(dx0),1))
    
    # upper bounds
    x_fraction_UB = np.ones((int((dx0.size - 1)/2),1)) # - 1 from delayed states
    x_holdup_UB = 100*np.ones((int((dx0.size - 1)/2),1))
    x_delay_UB = np.ones((1,1))
    xUB = np.concatenate((x_fraction_UB,x_holdup_UB,x_delay_UB))
    
    #################
    # process noise #
    #################
    # Lower bounds
    wLB = -10*xUB
    
    # upper bounds
    wUB = 10*xUB
    
    ###############################
    # Building the bounds for MHE #
    ###############################
    # parameters
    mhePar = EstimationParameters()
    
    # variables
    lbw = []
    ubw = []
    
    # constraints
    lbg = []
    ubg = []
    
    # "Lift" initial conditions
    lbw.append(xLB)
    ubw.append(xUB)
    
    for k in range(mhePar['N']):
        
        # New NLP variable for the process noise
        lbw.append(wLB)
        ubw.append(wUB)
    
        # State at collocation points
        for j in range(mhePar['d']):
            lbw.append(xLB)
            ubw.append(xUB)
    
        # Loop over collocation points
        for j in range(1,mhePar['d'] + 1):
           # Append collocation equations        
           lbg.append(np.zeros(len(dx0)))
           ubg.append(np.zeros(len(dx0)))
        
        # New NLP variable for state at end of interval
        lbw.append(xLB)
        ubw.append(xUB)
    
        # Add equality constraint
        lbg.append(np.zeros(len(dx0)))
        ubg.append(np.zeros(len(dx0)))

    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)
    
    return lbw, ubw, lbg, ubg   

#=======================================================================
# MOVING HORIZON ESTIMATOR --> STATES + PARAMETERS
#=======================================================================
#=======================================================================
def MovingHorizonEstimatorStaPar(xvar,pvar,xdot):
    #=======================================================
    # Builds the MHE NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Estimators (EKF and MHE) parameters
    mhePar = EstimationParameters()
    
    # System parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0) + 1 + 1 # (+1 for p0 and + 1 for par['T'])
    npar = 2 # zF, alpha 
    
    ####################################
    # Building collocation polynomials #
    ####################################
    # Get collocation points
    tau_root = np.append(0, collocation_points(mhePar['d'], 'legendre'))
    
    # Coefficients of the collocation equation
    C = np.zeros((mhePar['d'] + 1,mhePar['d'] + 1))
    
    # Coefficients of the continuity equation
    D = np.zeros(mhePar['d'] + 1)
    
    # Coefficients of the quadrature function
    B = np.zeros(mhePar['d'] + 1)
    
    # Construct polynomial basis
    for j in range(mhePar['d'] + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(mhePar['d'] + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
    
        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
    
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(mhePar['d'] + 1):
            C[j,r] = pder(tau_root[r])
    
        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
        
    # Control discretization (size of the finite element)
    h = mhePar['dT']*mhePar['T']/mhePar['N'] # --> not that the model sampling time par['T'] may not be 1! We need to take that into account

    ##########################################################################
    # preparing the system model and parameters for implementing collocation #
    ##########################################################################
    # Measurements
    Y_in = MX.sym('Y_in', mhePar['N']*ny)
    Y = reshape(Y_in,mhePar['N'],ny)
     
    # Independent variables
    U_in = MX.sym('U_in', mhePar['N']*nu)
    U = reshape(U_in,mhePar['N'],nu)
    
    # Initial state
    X0_var = MX.sym('X0_var', nx)
    
    # Initial parameter
    theta0_var = MX.sym('theta0_var', npar) # 
    
    # Arrival Cost
    P_mat_in = MX.sym('P_mat',(nx + npar)**2)
    P_mat = reshape(P_mat_in,nx+npar,nx+npar)
        
    # Define the ODE right hand side: continuous time dynamics 
    f = Function('f', [xvar, pvar], [xdot]) 
    
    # Start with an empty NLP
    # variables
    w = []
    
    # constraints
    g = []
    
    # Initial conditions
    Xk = MX.sym('X_0', nx)
    w.append(Xk)
    
    thetak = MX.sym('theta_0', npar)
    w.append(thetak)
    
    # save previous value of the parameter for calculating parameter increment
    thetak_1 = thetak
    
    # Build the objective
    obj = 0
    
    # Compute the arrival cost contribution
    resk = vertcat(Xk,thetak) - vertcat(X0_var,theta0_var)
    obj += mtimes([resk.T,P_mat,resk])
        
    # Formulate the NLP
    for k in range(mhePar['N']):
        
        # State at collocation points
        Xc = []
        for j in range(mhePar['d']):
            Xkj = MX.sym('X_'+str(k)+'_'+str(j), nx)
            Xc.append(Xkj)
            w.append(Xkj)
    
        # Loop over collocation points
        # extrapolating (only necessary for Legendre)
        Xk_end = D[0]*Xk
        
        # including parameter estimates into input array
        inputsTemp = vertcat(U[k,0],U[k,1],U[k,2],U[k,3],U[k,4],thetak[0],U[k,6],thetak[1],U[k,8])
        
        for j in range(1,mhePar['d']+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*Xk
           for r in range(mhePar['d']): xp = xp + C[r+1,j]*Xc[r]
    
           # Append collocation equations
           fj = f(Xc[j-1],inputsTemp)         
           g.append(h*fj - xp)
    
           # Add contribution to the end state
           Xk_end = Xk_end + D[j]*Xc[j-1];
        
        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), nx)
        w.append(Xk)
        
        # Add equality constraint
        g.append(Xk_end - Xk)
        
        # at last time step we do not need a new parameter
        if k != mhePar['N'] - 1:
            # New NLP variable for the parameter
            thetak = MX.sym('theta_' + str(k+1),npar)
            w.append(thetak)
        
            # add the process model contribution to the OF
            theta_mov_k = thetak - thetak_1
            obj += mtimes([theta_mov_k.T,mhePar['invQetheta'],theta_mov_k])
        
            # Update past parameter
            thetak_1 = thetak
        
        # add the measurement noise contribution to the OF
        yHat = mtimes([mhePar['H'],Xk])
        
        # calculate the residual
        vk = yHat - Y[k,:].T
        
        # add contribution of the residual to the OF
        obj += mtimes([vk.T,mhePar['invRe'],vk])

    # Concatenate vectors
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    prob = {'f': obj, 'x': w, 'g': g, 'p':vertcat(Y_in, U_in, X0_var, theta0_var, P_mat_in)}
    
    # Create the solver
    opts = {'ipopt.print_level':5, 
            'print_time':0, 
            'ipopt.max_iter':500, 
            'ipopt.warm_start_init_point':'yes', # see N.B.1
            'calc_lam_x':True, # see N.B.2
            'ipopt.linear_solver':'mumps'}
            #'ipopt.tol':1e-4, 
            #'ipopt.acceptable_tol':100*1e-4, 
            

    solver = nlpsol('solver', 'ipopt', prob, opts)
    
    return solver
 
#=======================================================================    
def CallMovingHorizonEstimatorStaPar(MHEsolver,xhat0,thetaHat0,Y_k,P_k,Pi_k,w_warm,lam_w_warm,lam_g_warm):
    #=======================================================
    # Initializes and solves the MHE NLP
    #=======================================================
    ###########################
    # Initializing parameters #
    ###########################
    # Estimators (EKF and MHE) parameters
    mhePar = EstimationParameters()
    
    # System Parameters
    par = SystemParameters()
    
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    # bounds
    lbw, ubw, lbg, ubg = SystemBoundsMHEStaPar()
    
    # dimension of the system variables
    ny = len(y0)
    nx = len(dx0)
    nu = len(u0) + 1 + 1 # (+1 for p0 and + 1 for par['T'])
    npar = 2 # zF and alpha
    
    #####################################
    # Reading current plant information #
    #####################################
    # FILTER PARAMETERS
    # Measurements
    Y_in = np.reshape(Y_k,(mhePar['N']*ny,1))
     
    # Independent variables
    U_in = np.reshape(P_k,(mhePar['N']*nu,1))
    
    # Initial state
    X0_in = xhat0
    
    # Initial parameter
    Theta0_in = thetaHat0 
    
    # Arrival Cost
    Pi_k_in = np.linalg.inv(Pi_k)
    Pi_k_in = np.reshape(Pi_k_in,((nx + npar)**2,1))
    
    filter_parameters = np.concatenate((Y_in,U_in,X0_in,Theta0_in,Pi_k_in))
        
    ###################
    # Solving the NLP #
    ###################
    sol = MHEsolver(x0=w_warm, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=filter_parameters, lam_x0=lam_w_warm, lam_g0=lam_g_warm)
    
    # saving the solution
    # primal variables
    w_warm = sol['x'].full() 
    # dual variables
    lam_x_warm = sol['lam_x'].full() 
    lam_g_warm = sol['lam_g'].full() 
    
    # objective function
    of_sol = sol['f'].full() 
    
    if MHEsolver.stats()['success']:
        solverSol = 1
    else:
        solverSol = 0 # for all other types of errors
    
    ########################
    # Organizing variables #
    ########################
    # example order for d = 3 
    # 0:81    - x0      - (0*nx:1*nx - 1)
    # 82:83   - theta0  - (1*nx:1*nx + 1)
    
    # 84:165  - x00     - (1*nx + 2:2*nx + 1)
    # 166:247 - x01     - (2*nx + 2:3*nx + 1) 
    # 248:329 - x02     - (3*nx + 2:4*nx + 1)  
    # 330:411 - x1      - (4*nx + 2:5*nx + 1)  
    # 412:413 - theta1  - (5*nx + 2:5*nx + 3)  
    
    # 414:495 - x10     - (5*nx + 4:6*nx + 3)
    # 496:577 - x11     - (6*nx + 4:6*nx + 3) 
    # 578:659 - x12     - (7*nx + 4:8*nx + 3)  
    # 660:741 - x2      - (8*nx + 4:9*nx + 3)  
    # 742:743 - theta2  - (9*nx + 4:9*nx + 5)  per loop: (1 + d)*nx + npar
    
    # ... (N = 10) 
    
    # 3054:3135 - x9_0  - 37*nx + 20:38*nx + 19
    # 3136:3217 - x9_1  - 38*nx + 20:39*nx + 19
    # 3218:3299 - x9_2  - 40*nx + 20:41*nx + 19
    # 3300:3381 - x10    - 42*nx + 20:43*nx + 19
    # ....                              total = (N - 1)*((1 + d)*nx + npar) + (1 + d)*nx + nx + npar
    
    # xk0     - ((1 + d)*(ii - 1) + 1)*nx + ii*npar:((1 + d)*(ii - 1) + 2)*nx + ii*npar - 1
    # xk1     - ((1 + d)*(ii - 1) + 2)*nx + ii*npar:((1 + d)*(ii - 1) + 3)*nx + ii*npar - 1 
    # xk2     - ((1 + d)*(ii - 1) + 3)*nx + ii*npar:((1 + d)*(ii - 1) + 4)*nx + ii*npar - 1  
    # xk      - ((1 + d)*(ii - 1) + 4)*nx + ii*npar:((1 + d)*(ii - 1) + 5)*nx + ii*npar - 1 
    # thetak  - ((1 + d)*(ii - 1) + 5)*nx + ii*npar:((1 + d)*(ii - 1) + 5)*nx + ii*npar + 1 
    
    # building states (if want to plot in the future)
    thetaArrayk = []
    xArrayk = []
    
    # x0
    xArrayk.append(w_warm[0:nx])
    
    for ii in range(1,mhePar['N'] + 1):
        helpvar1 = 1 + mhePar['d']
        #xk
        xArrayk.append(w_warm[(helpvar1*(ii - 1) + 4)*nx + ii*npar:(helpvar1*(ii - 1) + 5)*nx + ii*npar - 1 + 1])        
       
        if ii != mhePar['N']:
            #thetak
            thetaArrayk.append(w_warm[(helpvar1*(ii - 1) + 5)*nx + ii*npar:(helpvar1*(ii - 1) + 5)*nx + ii*npar + 1 + 1])
                
    thetaArrayk = np.hstack(thetaArrayk)
    xArrayk = np.hstack(xArrayk)
    
    xhat_k = np.reshape(xArrayk[:,-1],(nx,1))
    theta_k = np.reshape(thetaArrayk[:,-1],(npar,1))
    
    return xhat_k, theta_k, of_sol, xArrayk, thetaArrayk, w_warm, lam_x_warm, lam_g_warm, solverSol

#=======================================================================
def ExtendedKalmanFilterStaPar(yk,pk_1,xk_1k_1,thetak_1k_1,PNk_1k_1,F,S_xx,S_xp):
    #===================================================================================#
    # Solves the estimation problem using an EKF (also used for computing arrival cost) #
    #===================================================================================#
    # System parameters
    par = SystemParameters()
    
    # Estimators (EKF and MHE) parameters
    ekf = EstimationParameters()
    
    # ===================================
    #     Integrating results
    # ===================================
    # L, V, D, B, F, ''zF'', dF, ''alpha'', T
    inputsTemp = vertcat(pk_1[0],pk_1[1],pk_1[2],pk_1[3],pk_1[4],thetak_1k_1[0],pk_1[6],thetak_1k_1[1],pk_1[8])
    # Evolving plant in time
    Ik = F(x0=xk_1k_1,p=inputsTemp)
    
    # extracting solution
    xkk_1 = Ik['xf'].full()
    
    # ===================================
    #     Sensitivities
    # ===================================
    # states
    Fx_k_1 = S_xx(x0=xk_1k_1, p=inputsTemp)['jac_xf_x0'].full()
    # states
    Fp_k_1_temp = S_xp(x0=xk_1k_1, p=inputsTemp)['jac_xf_p'].full()
    
    # extracting the values related to zF and alpha
    Fp_k_1 = np.vstack([Fp_k_1_temp[:,5], Fp_k_1_temp[:,7]]).T
    
    # ===================================
    #     Extended System
    # ===================================
    # arranging filter matrices 
    Fe_temp_1 = np.concatenate((Fx_k_1, Fp_k_1), axis=1)
    Fe_temp_2 = np.concatenate((np.zeros((len(thetak_1k_1),len(xk_1k_1))), np.identity(len(thetak_1k_1))), axis=1)
    Fek_1 = np.concatenate((Fe_temp_1, Fe_temp_2), axis=0) 
   
    # outputs transition original (Linear Output Function)
    Hk = ekf['H']
    
    # output transition (extended)
    Hek = np.concatenate((Hk,np.zeros((len(yk),len(thetak_1k_1)))), axis=1)

    # preparing extended vector
    xekk_1 = np.concatenate((xkk_1, thetak_1k_1), axis=0)

    # ===================================
    #     Filter
    # ===================================
    # predicted covariance estimate
    Pkk_1 = Fek_1.dot(PNk_1k_1).dot(Fek_1.transpose()) + ekf['Qe']
       # dot or matmul?
       
    # Updating
    # Innovation covariance
    Sk = Hek.dot(Pkk_1).dot(Hek.transpose()) + ekf['R']
    
    # Kalman Gain
    Skinv = np.linalg.inv(Sk)
    Kk = Pkk_1.dot(Hek.transpose()).dot(Skinv)
    
    # Updating covariance estimate
    # Trick to guarantee that the matrix is symmetric
    temp = (np.identity(len(xekk_1)) - np.dot(Kk,Hek))
    Pkk = temp.dot(Pkk_1).dot(temp.transpose()) + Kk.dot(ekf['R']).dot(Kk.transpose()) # Joseph form (posteriori estimate of the covariance matrix)
    
    # temp2 = Pkk
    # for ii in range(np.shape(temp2)[0]):
    #     for jj in range(ii + 1, np.shape(temp2)[0]):
    #         temp2[jj,ii] = Pkk[ii,jj]
    
    # Pkk = temp2
    
    # Evaluating model outputs at k|k-1
    yHat = Hek.dot(xekk_1)
    
    # Update state estimate
    xekk = xekk_1 + Kk.dot(yk - yHat)
    
    # ===================================
    #     Results
    # ===================================
    # dividing extended vector
    xHatkk = xekk[:len(xk_1k_1)] # states
    thetaHatkk = xekk[len(xk_1k_1):] # states
    
    return xHatkk, thetaHatkk, Pkk

#=======================================================================
def MHEGuessInitializationStaPar(xArrayk,thetaArrayk):
    ######################################################
    # Building initial guess for the first MHE iteration #
    ######################################################
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    npar = 2
    
    # parameters
    mhePar = EstimationParameters()
    
    w0 = []
    lam_w0 = []
    lam_g0 = []
    
    # Initial conditions
    w0.append(xArrayk[:,0])
    lam_w0.append(np.ones((len(dx0),1)))
    
    # parameters
    w0.append(thetaArrayk[:,0])
    lam_w0.append(np.ones((npar,1)))
    
    # Formulate the NLP
    for k in range(mhePar['N']):
    
        # State at collocation points
        for j in range(mhePar['d']):
            w0.append(xArrayk[:,k + 1])
            lam_w0.append(np.ones((len(dx0),1)))
            
        # Loop over collocation points
        for j in range(1,mhePar['d'] + 1):
           # Append collocation equations        
           lam_g0.append(np.ones((len(dx0),1)))
    
        # New NLP variable for state at end of interval
        w0.append(xArrayk[:,k + 1])
        lam_w0.append(np.ones((len(dx0),1)))
    
        # Add equality constraint
        lam_g0.append(np.ones((len(dx0),1)))
        
        # at last time step we do not need a new parameter
        if k != mhePar['N'] - 1:
            # parameter
            w0.append(thetaArrayk[:,k + 1])
            lam_w0.append(np.ones((npar,1)))
        
    # Concatenate vectors
    w0 = np.concatenate(w0)
    
    # Initializing multipliers            
    lam_w0 = np.concatenate(lam_w0)
    lam_g0 = np.concatenate(lam_g0)
    
    return w0, lam_w0, lam_g0   

#=======================================================================
def SystemBoundsMHEStaPar():
    #============================#
    # specify bounds for MHE NLP # 
    #============================#
    # Initial condition
    dx0, y0, u0, p0 = InitialCondition()
    
    ##########
    # states #
    ##########
    # Lower bounds
    xLB = np.zeros((len(dx0),1))
    
    # upper bounds
    x_fraction_UB = np.ones((int((dx0.size - 1)/2),1)) # -1 from delayed measurement
    x_holdup_UB = 100*np.ones((int((dx0.size - 1)/2),1))
    x_delay_UB = np.ones((1,1))
    xUB = np.concatenate((x_fraction_UB,x_holdup_UB,x_delay_UB))
    
    ####################
    # model parameters #
    ####################
    # Lower bounds
    thetaLB = np.array([0,1], ndmin=2).T
    
    # Upper bounds
    thetaUB = np.array([1,2], ndmin=2).T
    
    ###############################
    # Building the bounds for MHE #
    ###############################
    # parameters
    mhePar = EstimationParameters()
    
    # variables
    lbw = []
    ubw = []
    
    # constraints
    lbg = []
    ubg = []
    
    # "Lift" initial conditions
    # states
    lbw.append(xLB)
    ubw.append(xUB)

    # parameters
    lbw.append(thetaLB)
    ubw.append(thetaUB)
    
    for k in range(mhePar['N']):
            
        # State at collocation points
        for j in range(mhePar['d']):
            lbw.append(xLB)
            ubw.append(xUB)
    
        # Loop over collocation points
        for j in range(1,mhePar['d'] + 1):
           # Append collocation equations        
           lbg.append(np.zeros(len(dx0)))
           ubg.append(np.zeros(len(dx0)))
        
        # New NLP variable for state at end of interval
        lbw.append(xLB)
        ubw.append(xUB)
    
        # Add equality constraint
        lbg.append(np.zeros(len(dx0)))
        ubg.append(np.zeros(len(dx0)))

        # at last time step we do not need a new parameter
        if k != mhePar['N'] - 1:
            # parameter
            lbw.append(thetaLB)
            ubw.append(thetaUB)
      
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)
    
    return lbw, ubw, lbg, ubg   

#######################
# NOTES FROM THE CODE #
#######################
# SEE: EconomicOptimization.py