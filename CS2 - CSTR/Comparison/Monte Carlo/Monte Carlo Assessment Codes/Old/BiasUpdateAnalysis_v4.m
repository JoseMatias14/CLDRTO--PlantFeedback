clear 
clc
close all

import casadi.*

% for noise 
rng('default')

%% save results
nameFile = 'MC_parameters_CLv2_med.mat';


%% Preparing simulation
% Dynamic model 
[F,S_xx,S_xp] = CSTRDynamicModel();

% Initial condition, nominal parameters, and bounds
[X_0,U_0,theta_nom,sysBounds] = CSTRModelInitParBounds();

% sizes
dim_sys.nx = length(sysBounds.X_lb);
dim_sys.ntheta = length(sysBounds.theta_lb); % only estimable parameters (note: dim(theta_nom) != dim(theta_lb)
dim_sys.nu = length(sysBounds.U_lb);
                                                                                                                                                            
% simulation sampling time 
dt_sys = 2; %[h]

% Building DRTO solver
[solverDRTO,DRTObounds,Np] = DRTO(F,theta_nom,dt_sys,dim_sys,sysBounds);

%% Monte Carlo simulations
% simulation final value
tEnd = 100; %[h]
nEnd = tEnd/dt_sys;
Tpred = dt_sys*Np;

% exectution period of DRTO in system sampling time
dRTO_exec = 5; 

% Preparing bias updating simulation
% number of uncertain parameters
parIndex = [3, 5]; % position correspondence with theta0
nPar = 2; % dH, UA

% number of different parameters values drawn from the distribution
nSim = 20;

% mean is the nominal value, sigma is 0.001*nom
%Sig = [1; 1; 0.05; 1; 0.05; 1; 1];
Sig = [1; 1; 0.01; 1; 0.01; 1; 1];
sigma_theta_nom = Sig.*abs(theta_nom);

figure(1)
sgtitle('Parameter Distribution') 

thetaLab = {'F','V','\Delta H','\rho C_p','UA','\Psi','\Omega'};
for ii = 1:length(parIndex)
    % creating array to evaluate normal distribution
    x_temp = (theta_nom(parIndex(ii)) - 3*sigma_theta_nom(parIndex(ii))):(0.1*sigma_theta_nom(parIndex(ii))):(theta_nom(parIndex(ii)) + 3*sigma_theta_nom(parIndex(ii)));
    
    % evaluate normal dist. at points x_temp
    y_temp = normpdf(x_temp,theta_nom(parIndex(ii)),sigma_theta_nom(parIndex(ii)));

    % plotting
    subplot(2,1,ii)
        plot(x_temp,y_temp/sum(y_temp),'LineWidth',1.5), grid on, xlabel(thetaLab{parIndex(ii)}),ylabel('Probability Density')
        hold on 
        xline(theta_nom(parIndex(ii)),'k--','LineWidth',1.5)
        xline(theta_nom(parIndex(ii)) + sigma_theta_nom(parIndex(ii)),'k:','LineWidth',1.5)
        xline(theta_nom(parIndex(ii)) - sigma_theta_nom(parIndex(ii)),'k:','LineWidth',1.5)
end

% For saving the simulation results
for ii = 1:nPar
    for jj = 1:nSim
        % plant and model states
        XPlantPlot{ii,jj} = [];
        XModelPlot{ii,jj} = [];
        
        % optimal decisions implemented in the plant
        UPlot{ii,jj} = [];
    
        % plant and model states
        thetaModelPlot{ii,jj} = [];

        % instantaneous objective function
        OFPlot{ii,jj} = [];
    
        % bias model-plant
        biasPlot{ii,jj} = [];
    
        % checking if the solver converged
        SolFlagPlot{ii,jj} = [];
    end
end

% Drawing random numbers
thetaMC = zeros(nPar + 5,nSim + 3); %nominal, nominal + sigma, nominal - sigma, random

% nominal 
thetaMC(:,1) = theta_nom;
% nominal + sigma
thetaMC(:,2) = theta_nom + sigma_theta_nom;
% nominal - sigma
thetaMC(:,3) = theta_nom - sigma_theta_nom;

% drawing random numbers
for jj = 1:nSim
    % assign nominal parameters (from plant)
    thetaMC(:,3 + jj) = theta_nom;
    for ii = 1:nPar
        % draw parameter value
        parTemp = normrnd(theta_nom(parIndex(ii)),sigma_theta_nom(parIndex(ii)));
        % replace to create plant-model mismatch 
        thetaMC(parIndex(ii),3 + jj) = parTemp;
    end
end

% run MC simulation
for ii = 1:nPar
    for jj = 1:3 + nSim

        % counter used to implement DRTO's solution trajectory
        dRTO_exec_count = 0;

        % first order filter for bias
        filBiask = 0.0;
        
        % adding plant-model mismatch
        % plant parameter value
        theta_p = theta_nom;
        
        % assign parameter values from MC array
        theta_m = theta_nom;
        theta_m(parIndex(ii)) = thetaMC(parIndex(ii),jj);
        
        %initial condition
        Xk_p = X_0;
        Xk_m = X_0;
        biask = Xk_p(1:2) - Xk_m(1:2);
        Uk_p = U_0;

        % indicating that there is no feasible solution available
        solDRTO = 0;
            
        % for saving values
        XModelArray = Xk_m;
        XPlantArray = Xk_p;
        UPlantArray = Uk_p;
        OFPlantArray = 1 - Xk_p(1)/Uk_p(1);
        biasArray = biask;
        SolFlagArray = [];

        for kk = 1:nEnd
            fprintf('>>> Iteration: %d, Par: %d, Sim: %d \n',kk, ii, jj)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Apply optimal decision to the plant
            Fk = F('x0', Xk_p, 'p', [Uk_p;theta_p;dt_sys;zeros(dim_sys.nx - 1,1);0]);
            Xk_p = full(Fk.xf);
            OFk_p = full(Fk.qf);

            % Compute model state
            Fk = F('x0', Xk_m, 'p', [Uk_p;theta_m;dt_sys;zeros(dim_sys.nx - 1,1);0]);
            Xk_m = full(Fk.xf);

            % Computing bias (using 1st order filter)
            biask = (1 - filBiask)*(Xk_p(1:2) - Xk_m(1:2)) + filBiask*biask;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % saving values
            XModelArray = [XModelArray, Xk_m];
            XPlantArray = [XPlantArray, Xk_p];
            UPlantArray = [UPlantArray, Uk_p];
            OFPlantArray = [OFPlantArray, OFk_p];
            biasArray = [biasArray, biask];

            if kk > 10 && rem(kk,dRTO_exec) == 0
                if solDRTO == 0
                    w0_DRTO = repmat([Uk_p; Xk_p],[Np, 1]);
                else
                    % use previous solution
                    w0_DRTO = w_DRTO;
                end

                % bias update: nominal parameters, biask and Xk_m
                thetaDRTO = [theta_nom(1);theta_nom(2);theta_m(3);theta_nom(4);theta_m(5);theta_nom(6);theta_nom(7)];

                sol = solverDRTO('x0', w0_DRTO, 'lbx', DRTObounds.lbw, 'ubx', DRTObounds.ubw, ...
                    'lbg', DRTObounds.lbg, 'ubg', DRTObounds.ubg,'p',[Xk_m;Uk_p(1);thetaDRTO;biask]);
    
                % save solution
                w_DRTO = full(sol.x);
        
                % catch error
                if solverDRTO.stats.success ~=1
                    % solution failed
                    solDRTO = 0;
        
                    % uDRTOTrajectory and Uk_p are not updated
                    % re-starting counter --> use previous solution
                    Uk_p_array = repmat(Uk_p,[1, dRTO_exec]);
                    dRTO_exec_count = 1;
                else
                    % solution succeeded
                    solDRTO = 1;
        
                    % extracting trajectories
                    u_temp = [];
        
                    for k=1:Np
                        tempCounter = (dim_sys.nu + dim_sys.nx)*(k - 1);
                        u_temp = [u_temp, w_DRTO(tempCounter + 1:tempCounter + dim_sys.nu)];
                    end
       
                   
                    % updating
                    Uk_star = u_temp(:,1:dRTO_exec);
                    Uk_p_array = [Uk_star(1,:);Uk_star(2,:);Uk_star(3,:)]; % all are implemented
        
                    % re-starting counter
                    dRTO_exec_count = 1;

                end
                % saving flag
                SolFlagArray = [SolFlagArray, solDRTO];

            end % DRTO loop

            if dRTO_exec_count ~= 0  % avoiding loop before first DRTO execution
                Uk_p = Uk_p_array(:,dRTO_exec_count);
                dRTO_exec_count = dRTO_exec_count + 1;
            end

        end

        % For plotting
        XPlantPlot{ii,jj} = XPlantArray;
        XModelPlot{ii,jj} = XModelArray;
        UPlot{ii,jj} = UPlantArray;
        OFPlot{ii,jj} = OFPlantArray;
        biasPlot{ii,jj} = biasArray;
        SolFlagPlot{ii,jj} = SolFlagArray;
        thetaModelPlot{ii,jj} = theta_m;
    end
end

% saving results
save(nameFile,'tEnd','nEnd','Tpred','nPar','nSim','dRTO_exec','thetaMC','XPlantPlot','XModelPlot','UPlot','OFPlot','biasPlot','SolFlagPlot','thetaModelPlot')

%% Plotting results
BiasUpdateAnalysis_4_Plot

%%%%%%%%%%%%%%%
% CSTR MODELS %
%%%%%%%%%%%%%%%

function [F,S_xx,S_xp] = CSTRDynamicModel()
    % Builds CSTR Model from: https://www.mathworks.com/help/mpc/gs/cstr-model.html
    % The adiabatic continuous stirred tank reactor (CSTR)
    % a single first-order exothermic and irreversible reaction, A → B, takes place in the vessel, which is assumed to be always perfectly mixed. 
    % The inlet stream of reagent A enters the tank at a constant volumetric rate. 
    % The product stream B exits continuously at the same volumetric rate, and liquid density is constant. 
    % Thus, the volume of reacting liquid is constant.
    
    import casadi.*

    % Declare model states
    C_A = SX.sym('C_A');            % C_A: Concentration of A [kmol/m3] 
    T = SX.sym('T');                % T: Reactor temperature [K] 
    I_T = SX.sym('I_T');            % I_T: integral error [K] 
    x = [C_A; T; I_T];
    
    % Declare model inputs
    CAf = SX.sym('CAf');            % CAf: Concentration of A in the feed [kmol/m3]
    Tf = SX.sym('Tf');              % Tf: Feed temperature [K] 
    Tsp = SX.sym('Tsp');            % Tsp: Reactor temperature setpoint [K]     
    u = [CAf; Tf; Tsp];
    
    % Declare model disturbances and parameters
    dH = SX.sym('dH');              % dH: Heat of reaction per mole [1000 kcal/kmol]	
    rhoCp = SX.sym('rhoCp');        % rhoCp: Density multiplied by heat capacity [100 kcal/(m3·K)]
    UA = SX.sym('UA');              % UA: Overall heat transfer coefficient multiplied by tank area [100 kcal/(K·h)]
    Psi = SX.sym('Psi');            % Rearranged reaction parameters
    Omega = SX.sym('Omega');
    F = SX.sym('F');                 % F: Volumetric flow rate [m3/h]
    V = SX.sym('V');                 % V: Reactor volume [m3]
    % time transformation: CASADI integrates always from 0 to 1
    % and the USER specifies the time
    % scaling with step length
    h = SX.sym('h');                 % [h]
    theta = [F; V; dH; rhoCp; UA; Psi; Omega; h];
    
    %%%%%%%%%%%%%%
    % CONTROLLER %
    %%%%%%%%%%%%%%
    % Control action (pair: T <=> Tc )
    % Loop: temperature - coolant
    % Proportional Gain
    K_T = -2.050020500205002; % [-]
    % Integral Constant
    tau_T = 1.369675386933297; % [h]
    % Nominal volume 
    T_nom = 311.2639;   % [K]
    % Nominal flow
    Tc_nom = 292;     % [K]
    % SP deviation
    e_T = T - Tsp;      % [K} 
    % MV
    Tc = Tc_nom + K_T*(e_T + I_T/tau_T); 
    
    %%%%%%%%%%%%
    % REACTION %
    %%%%%%%%%%%%
    % Reaction rate constant (rearranged as in Statistical assessment of chemical kinetic models (1975) - D.J.Pritchard and D.W.Bacon)
    Tref = 373;          % TrefReference time for reparametrization [K]
    k = exp(Omega + (Tref/T - 1)*Psi);
    % Reaction rate per unit of volume, and it is described by the Arrhenius rate law, as follows.
    r = k*C_A;
    
    %%%%%%%%%%%%%%%%%
    % MASS BALANCES %
    %%%%%%%%%%%%%%%%%
    % ! assuming density is constant and perfect level control
    % Component balances
    dCAdt = (F/V)*(CAf - C_A) - r;
    % Energy Balance
    dTdt = (F/V)*(Tf - T) - ((dH*10^3)/(rhoCp*10^2))*r - (UA*10^2)/(V*rhoCp*10^2)*(T - Tc);
    % controller error integration
    dx_Edt = e_T;
    
    xdot = vertcat(dCAdt,dTdt,dx_Edt);
    
    %%%%%%%%%%%%%%%%%%
    % OBJECTIVE FUNC %
    %%%%%%%%%%%%%%%%%%
    % adding bias model
    bias_m = SX.sym('bias_m',2); % Ca, T

    % adding cost related to Caf usage
    DU_Caf = SX.sym('DU_Caf'); % Ca, T
    
    % Economic term
    % conversion: moles of A reacted vs. moles of A fed
    L = (1 - (C_A + bias_m{1})/CAf) - DU_Caf*(1.3*1e-2)*DU_Caf; 
    
    % Tracking 
    % using Tc as the control input and CA (around 2 (kg·mol)/m3)
    % L = (C_A + bias_m{1} - 2)^2 + 1e-6*(Tc + bias_m{2})^2;
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % CREATING INTEGRATOR %
    %%%%%%%%%%%%%%%%%%%%%%%
    ode = struct('x',x,'p',vertcat(u,theta,bias_m,DU_Caf),'ode',h*xdot,'quad',h*L); 
    F = integrator('F', 'cvodes', ode);

    %%%%%%%%%%%%%%%%%
    % SENSITIVITIES %
    %%%%%%%%%%%%%%%%%
    S_xx = F.factory('sensParStates',{'x0','p'},{'jac:xf:x0'});
    S_xp = F.factory('sensParStates',{'x0','p'},{'jac:xf:p'});

end

function [X_0,U_0,theta_nom,sysBounds] = CSTRModelInitParBounds()

    %%%%%%%%%%%%%%%%%%%%%%
    % INITIAL CONDITION  %
    %%%%%%%%%%%%%%%%%%%%%%
    % initial condition is given
    X_0 = [8.5691;    % C_A: Concentration of A [kmol/m3] 
          311.2740;     % T: Reactor temperature [K] 
          0];           % I_T: integral error [h] 
    
    U_0 = [8.5691;       % CAf: Concentration of A in the feed [kmol/m3] 10
          300;      % Tf: Feed temperature [K] 
          311.2740];      % Tsp: Reactor temperature setpoint [K] 

    %%%%%%%%%%%%%%%%%%%%%%
    % NOMINAL PARAMETERS %
    %%%%%%%%%%%%%%%%%%%%%%
    % parameters
    E_nom = 11843*1.1;	     % E: Activation energy per mole [kcal/kmol]	
    R_nom = 1.985875;    % R: Boltzmann's ideal gas constant [kcal/(kmol·K)]	
    
    k0_nom = 34930800;	 % Pre-exponential nonthermal factor [1/h]	
    % reparametrization
    Tref = 373;          % TrefReference time for reparametrization [K]
    Psi_nom = - E_nom/(R_nom*Tref);
    Omega_nom = log(k0_nom) + Psi_nom;
    
    theta_nom = [1;           % Fin: Volumetric flow rate [m3/h]	
              1;           % V: Reactor volume [m3]
              -5.960;      % dH: Heat of reaction per mole [1000 kcal/kmol]	
              5;	       % rhoCp: Density multiplied by heat capacity [100 kcal/(m3·K)]	
              1.6;	       % UA: Overall heat transfer coefficient multiplied by tank area [100 kcal/(K·h)]
              Psi_nom;
              Omega_nom];

    %%%%%%%%%% 
    % BOUNDS %
    %%%%%%%%%%
    % States Bounds
    sysBounds.X_lb = [1e-6;    % C_A [kmol/m3] 
            250.0;   % T [K] 
            -inf];   % integral error [h]
    
    sysBounds.X_ub = [14.0;    % C_A [kmol/m3] 
            400.0;   % T [K] 
            inf];    % integral error [h]

    % Inputs Bounds
    sysBounds.U_lb = [6.0;     % CAf [kmol/m3] 
            260.0;   % Tf [K] 
            250.0];   % Tsp [K] 
    
    sysBounds.U_ub = [14.0;     % CAf [kmol/m3] 
            340.0;    % Tf [K] 
            400.0];    % Tsp [K] 

    % Parameters bounds
    sysBounds.theta_lb = [-7;
                          1.2];   % Psi [rearranged activation energy]

    sysBounds.theta_ub = [-3;
                          1.8];   % Psi [rearranged activation energy]

end

function [solver,DRTObounds,N] = DRTO(F,theta_nom,dt_sys,dim_sys,sb)
    
    import casadi.*

    %%%%%%%%%%%%%%%%%%%%%
    % TUNING PARAMETERS %
    %%%%%%%%%%%%%%%%%%%%%
    dT_DRTO = dt_sys; % [h]
    N = 10; % number of control intervals 32 | 14
    
    %%%%%%%%%%%%%%%%%
    % Declaring NLP %
    %%%%%%%%%%%%%%%%%
    % Start with an empty NLP
    w = {};
    J = 0;
    g={};
    DRTObounds.lbw = [];
    DRTObounds.ubw = [];
    DRTObounds.lbg = [];
    DRTObounds.ubg = [];
    
    % Formulate the NLP
    % NLP parameters
    par = MX.sym('par',length(theta_nom));
    bias = MX.sym('bias',dim_sys.nx  - 1); % all states are measured (integral error does not count)
    
    % initial condition is consider an NLP parameter
    X0 = MX.sym('X0', dim_sys.nx);
    Xk = X0;

    % previous input is known
    Uprev = MX.sym('Uprev');
    U_1 = Uprev;

    for k=0:N-1
        % New NLP variable for the control
        Uk = MX.sym(['U_' num2str(k)],dim_sys.nu);
        w = {w{:}, Uk};
        DRTObounds.lbw = [DRTObounds.lbw; sb.U_lb];
        DRTObounds.ubw = [DRTObounds.ubw; sb.U_ub];
    
        % computing DU (only for Caf)
        DU = Uk{1} - U_1;

        % updating for loop
        U_1 = Uk{1};

        % Integrate till the end of the interval
        Fk = F('x0',Xk,'p', [Uk;par;dT_DRTO;bias;DU]);
        Xk_end = Fk.xf;
        J = J + Fk.qf;
    
        % New NLP variable for state at end of interval
        Xk = MX.sym(['X_' num2str(k+1)], dim_sys.nx);
        w = {w{:}, Xk};
        DRTObounds.lbw = [DRTObounds.lbw; -inf*ones(dim_sys.nx,1)];
        DRTObounds.ubw = [DRTObounds.ubw; inf*ones(dim_sys.nx,1)];
        
        % adding bias to the states constraints
        g = {g{:}, Xk + vertcat(bias{1},bias{2},0)}; % (integral error does not count)
        DRTObounds.lbg = [DRTObounds.lbg; sb.X_lb];
        DRTObounds.ubg = [DRTObounds.ubg; sb.X_ub];

        % Close continuinity gap
        g = {g{:}, Xk_end-Xk};
        DRTObounds.lbg = [DRTObounds.lbg; zeros(dim_sys.nx,1)];
        DRTObounds.ubg = [DRTObounds.ubg; zeros(dim_sys.nx,1)];

    end
    
    % Create an NLP solver
    opts = struct;
    opts.ipopt.max_iter = 1000;
    opts.ipopt.print_level = 0;
    opts.print_time = 0;
    
    prob = struct('f', -J, 'x', vertcat(w{:}),'g',vertcat(g{:}),'p',vertcat(X0,Uprev,par,bias));
    solver = nlpsol('solver', 'ipopt', prob,opts);
end


