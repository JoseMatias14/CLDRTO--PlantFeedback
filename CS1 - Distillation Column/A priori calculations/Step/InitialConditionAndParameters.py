#=======================================================
# Author: Jose Otavio Matias
# email: assumpcj@macmaster.ca 
# March 2018 (in Matlab); Last revision: 07-Feb-2022
#=======================================================

import numpy as np

#======================================================================
def InitialCondition():
    #=======================================================
    # specify initial conditions
    #=======================================================
    
    # Inputs: par - system parameters
    par =  SystemParameters()
    
    # States
    # mole fractions of A [kmol A/kmol total]
    xA0 = [1.0000000000000e-02,
          1.4260912285191e-02,
          1.9723589240705e-02,
          2.6693264811825e-02,
          3.5531128682305e-02,
          4.6650909897754e-02,
          6.0505364325745e-02,
          7.7557807588356e-02,
          9.8234185152883e-02,
          0.12285384713922,
          0.15154333129394,
          0.18414705878182,
          0.22015928656995,
          0.25870700564480,
          0.29860668074760,
          0.33849626436079,
          0.37701507466589,
          0.41298329547589,
          0.44553302705302,
          0.47416387897878,
          0.49872493204438,
          0.52649475964320,
          0.55776390242974,
          0.59216060127435,
          0.62903916925421,
          0.66750673731202,
          0.70649840345461,
          0.74489017363107,
          0.78162559565357,
          0.81582677099606,
          0.84686636753928,
          0.87439107401281,
          0.89830161503826,
          0.91870393775007,
          0.93584846972424,
          0.95007115492539,
          0.96174449436761,
          0.97124167454496,
          0.97891332406394,
          0.98507462686567,
          0.99000000000000]
    
    xA0_arr = np.array(xA0, ndmin=2)
    
    # stage holdup [kmol]
    M0 = 0.5*np.ones((par['NT'],1)) 
    # for more realistic studies: M0_1=10 [kmol] (reboiler) 
    #                         and M0_NT=32.1 [kmol] (condenser)
    
    dx0 = np.concatenate((np.transpose(xA0_arr),M0))
    
    # Inputs
    # Reflux
    LT0 = 2.70629  # [kmol/mim]
    # Boilup
    VB0 = 3.20629    # [kmol/min]
    # Distillate
    D0 = 0.5          # [kmol/min]
    # Bottom flowrate
    B0 = 0.5         # [kmol/min]    

    # Feed flowrate                
    F0 = 1       # [kmol/min]  1.0 | 1.01                 
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
    y0 = np.vstack((dx0[0],dx0[6],dx0[23],dx0[34],dx0[40],dx0[41],dx0[81])) # molar fraction of internal stages + reboiler and condenser holdups

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
    
    # Boiling point of the substances (not being used)
    T_bL = 341.9
    T_bH = 355.4
        
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
    L0 = 2.70629  

    # Nominal liquid flow below feed [kmol/min]  	
    L0b = L0 + qF0*F0	
    
    # Effect of vapor flow on liquid flow ("K2-effect")
    lamb = 0
	
    # Nominal boilup flow [kmol/min] 
    V0 = 3.20629
    
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
           'T':T_samp}
    
    return par
