clear 
clc
close all

import casadi.*

% for noise 
rng('default')

%PlotResults('MHE_2p.mat')

%% choosing model adaptation strategy
ma = [0,  % bias update
      1]; % MHE

%% save results
if ma(1) == 1
    nameFile = 'test2_BIAS_2p_un';
elseif ma(2) == 1
    nameFile = 'test2_MHE_2p_un';
end

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

% Building MHE solver
[solverMHE,MHEbounds,Ne,Qx,Qtheta,Qy] = MHE(F,theta_nom,dt_sys,dim_sys,sysBounds);

% Building DRTO solver
[solverDRTO,DRTObounds,Np] = DRTO(F,theta_nom,dt_sys,dim_sys,sysBounds);

%% Closed-loop simulation
% simulation final value
tEnd = 100; %[h]
nEnd = tEnd/dt_sys;

% exectution period of DRTO in system sampling time
dRTO_exec = 5; 

% counter used to implement DRTO's solution trajectory
dRTO_exec_count = 0;

% first order filter for bias
filBiask = 0.0;

% adding plant-model mismatch
% plant parameter value
theta_p = theta_nom;

% plant parameter value
theta_m = [theta_p(1);
            theta_p(2);
            -4;
            theta_p(4);
            1.5;
            theta_p(6);
            theta_p(7)];

%initial condition
Xk_p = X_0;
Xk_m = X_0;
biask = Xk_p(1:2) - Xk_m(1:2);
Uk_p = U_0;

%%%%%%%%%%%%%%
% Estimation %
%%%%%%%%%%%%%%
% Estimated states
Xhatk = Xk_p; % states are known
% Estimated values
thetaHatk = [theta_m(3);
             theta_m(5)];
% Arrival cost guess
% Pkk = diag([1,0.01,0.1,10]);
% Pkk = [0.51,  0.06,	0.02, 0.22;
%        0.06, 50.56, 0.63, 1.14;
%        0.02,  0.63,	8.14, 1.28;
%        0.22,  1.14, 1.28, 6.04];
Pkk = diag([1,100,1000,1,0.1]);

% indicating that there is no feasible solution available
solMHE = 0;
solDRTO = 0;

%%%%%%%%%%%%%%%%%
% Saving Values %
%%%%%%%%%%%%%%%%%
% Plant
XPlantArray = [Xk_p;292]; % nominal temperature
XMeasArray = Xk_p;
UPlantArray = Uk_p;
OFPlantArray = 1 - Xk_p(1)/Uk_p(1);
thetaPlantArray = [theta_p(3);theta_p(5)];

% Nominal Model
XModelArray = Xk_m;
biasModelArray = biask;

% Estimation 
XHatArray = Xhatk;
thetaHatArray = thetaHatk;
SolMHEFlag = [];

execTimeArray = [];

for kk = Ne + 1:nEnd
    xHatTrajectory{kk} = [];
    thetaHatTrajectory{kk} = [];
end

% Optimization
SolDRTOFlag = [];
for kk = Ne + 1:nEnd
    uDRTOTrajectory{kk} = [];
end

% beginning simulation
for kk = 1:nEnd
    fprintf('>>> Iteration: %d \n',kk)

    if dRTO_exec_count ~= 0  % avoiding loop before first DRTO execution
        Uk_p = Uk_p_array(:,dRTO_exec_count);
        dRTO_exec_count = dRTO_exec_count + 1;
    end

    % Apply optimal decision to the plant
    Fk = F('x0', Xk_p, 'p', [Uk_p;theta_p;dt_sys;zeros(dim_sys.nx - 1,1);0]);
    Xk_p = full(Fk.xf);
    OFk_p = full(Fk.qf);

    % Recomputing jacket temperature
    % Proportional Gain
    K_T = -2.050020500205002; % [-]
    % Integral Constant
    tau_T = 1.369675386933297; % [h]
    % Nominal flow
    Tc_nom = 292; % [K]
    % SP deviation
    e_T = Xk_p(2) - Uk_p(3);      % [K} 
    % recomputing Tc
    Tc = Tc_nom + K_T*(e_T + Xk_p(3)/tau_T);

    % Compute model state
    Fk = F('x0', Xk_m, 'p', [Uk_p;theta_m;dt_sys;zeros(dim_sys.nx - 1,1);0]);
    Xk_m = full(Fk.xf);

    % Computing bias (using 1st order filter)
    biask = (1 - filBiask)*(Xk_p(1:2) - Xk_m(1:2)) + filBiask*biask;

    % Update arrival cost
    Pkk = ArrivalCost(Xhatk,thetaHatk,Pkk,Uk_p,theta_p,dt_sys,dim_sys,Qx,Qtheta,Qy,F,S_xx,S_xp);
    
    % saving values
    XPlantArray = [XPlantArray, [Xk_p;Tc]];
    XMeasArray = [XMeasArray, Xk_p + 1e-6*randn(3,1)];
    UPlantArray = [UPlantArray, Uk_p + 1e-6*randn(3,1)];
    OFPlantArray = [OFPlantArray, OFk_p];
    thetaPlantArray = [thetaPlantArray, [theta_p(3);theta_p(5)]];
    XModelArray = [XModelArray, Xk_m];
    biasModelArray = [biasModelArray, biask];
    XHatArray = [XHatArray, Xhatk];
    thetaHatArray = [thetaHatArray, thetaHatk];

    %% calling MHE
    if kk > Ne && rem(kk,dRTO_exec) == 0
        if ma(1) == 1

            SolMHEFlag = [SolMHEFlag, 0];
            tMHE = 0;

            % message
            fprintf('>>>>>> Bias Update \n')

        elseif ma(2) == 1
            % Preparing guess for MHE (other bounds were previously declared)
            if solMHE == 0
                w0_MHE = repmat([XMeasArray(:,end); thetaHatk],[Ne+1, 1]);
            else
                 % use previous solution
                 w0_MHE = w_MHE;
            end
            
            
            % solve MHE problem
            MHEparameters = [reshape(XMeasArray(:,kk - Ne + 1:kk)',[Ne*dim_sys.nx,1]);    %Y_in
                             reshape(UPlantArray(:,kk - Ne + 1:kk)',[Ne*dim_sys.nu,1]);         %U_in
                             thetaHatArray(:,kk - Ne);     %theta0_in
                             XHatArray(:,kk - Ne);         %X0_in
                             reshape(Pkk,[(dim_sys.nx+dim_sys.ntheta)^2,1])]; %P_mat_in
    
            tic
            sol = solverMHE('x0', w0_MHE, 'lbx', MHEbounds.lbw, 'ubx', MHEbounds.ubw,...
                'lbg', MHEbounds.lbg, 'ubg', MHEbounds.ubg,'p',MHEparameters);
        
            tMHE = toc;
    
            % catch error
            if solverMHE.stats.success ~=1
                % solution failed
                solMHE = 0;
                text_temp = 'failed';
    
                % xHatTrajectory, thetaHatTrajectory, Xhatk and thetaHatk 
            else
                % solution succeeded
                solMHE = 1;
                text_temp = 'conv.';
    
                % save estimates
                w_MHE = full(sol.x);
    
                % extracting trajectories
                x_temp = [];
                theta_temp = [];
    
                for k=1:Ne+1
                    tempCounter = (dim_sys.nx + dim_sys.ntheta)*(k - 1);
                    x_temp = [x_temp, w_MHE(tempCounter + 1:tempCounter + dim_sys.nx)];
                    theta_temp = [theta_temp, w_MHE(tempCounter + dim_sys.nx + 1:tempCounter + dim_sys.nx + dim_sys.ntheta)];
                end
    
                % saving data
                xHatTrajectory{kk} = x_temp;
                thetaHatTrajectory{kk} = theta_temp;
    
                % updating
                Xhatk = x_temp(:,end);
                thetaHatk = theta_temp(:,end);
            end
            SolMHEFlag = [SolMHEFlag, solMHE];
    
            % overriding integral state
            Xhatk(3) = Xk_p(3);
    
            % message
            fprintf('>>>>>> MHE: %s , time: %d \n',text_temp,tMHE)
        end
        %% calling DRTO
        % Preparing initial guess
        if solDRTO == 0
            w0_DRTO = repmat([Uk_p; Xk_p],[Np, 1]);
        else
            % use previous solution
            w0_DRTO = w_DRTO;        
        end
      
        tic
        if ma(1) == 1
            % bias update: nominal parameters, biask and Xk_m
            thetaDRTO = [theta_nom(1);theta_nom(2);theta_m(3);theta_nom(4);theta_m(5);theta_nom(6);theta_nom(7)];

            sol = solverDRTO('x0', w0_DRTO, 'lbx', DRTObounds.lbw, 'ubx', DRTObounds.ubw, ...
                'lbg', DRTObounds.lbg, 'ubg', DRTObounds.ubg,'p',[Xk_m;Uk_p(1);thetaDRTO;biask]);

        elseif ma(2) == 1
            % bias update: xHatk, thetaHatk, and do not use biask
            thetaDRTO = [theta_nom(1);theta_nom(2);thetaHatk(1);theta_nom(4);thetaHatk(2);theta_nom(6);theta_nom(7)];

            sol = solverDRTO('x0', w0_DRTO, 'lbx', DRTObounds.lbw, 'ubx', DRTObounds.ubw, ...
                'lbg', DRTObounds.lbg, 'ubg', DRTObounds.ubg,'p',[Xhatk;Uk_p(1);thetaDRTO;zeros(2,1)]);
        end

        tDRTO = toc;

        % save solution
        w_DRTO = full(sol.x);

        % catch error
        if solverDRTO.stats.success ~=1
            % solution failed
            solDRTO = 0;
            text_temp = 'failed';

            % uDRTOTrajectory and Uk_p are not updated
            % re-starting counter --> use previous solution
            Uk_p_array = repmat(Uk_p,[1, dRTO_exec]);
            dRTO_exec_count = 1;
        else
            % solution succeeded
            solDRTO = 1;
            text_temp = 'conv.';

            % extracting trajectories
            u_temp = [];

            for k=1:Np
                tempCounter = (dim_sys.nu + dim_sys.nx)*(k - 1);
                u_temp = [u_temp, w_DRTO(tempCounter + 1:tempCounter + dim_sys.nu)];
            end

            % saving data
            uDRTOTrajectory{kk} = u_temp;

            % updating
            Uk_star = u_temp(:,1:dRTO_exec);
            Uk_p_array = [Uk_star(1,:);Uk_star(2,:);Uk_star(3,:)]; % all are implemented

            % re-starting counter
            dRTO_exec_count = 1;

        end
        SolDRTOFlag = [SolDRTOFlag, solDRTO];
    
        % message
        fprintf('>>>>>> DRTO: %s , time: %d \n',text_temp,tDRTO)

        % saving execution time
        execTimeArray = [execTimeArray, [tMHE;tDRTO]];
    end


end
save(nameFile,'dt_sys','dRTO_exec','Ne','Np','tEnd','nEnd','XPlantArray','XMeasArray','UPlantArray','OFPlantArray','thetaPlantArray',...
    'XModelArray','biasModelArray','XHatArray','thetaHatArray',...
    'xHatTrajectory','thetaHatTrajectory','SolMHEFlag',...
    'uDRTOTrajectory','SolDRTOFlag','execTimeArray');

PlotResults(nameFile)
   
%% FUNCTIONS 

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

function [solver,MHEbounds,N,Qx,Qtheta,Qy] = MHE(F,theta_nom,dt_sys,dim_sys,sb)
    
    import casadi.*

    %%%%%%%%%%%%%%%%%%%%%
    % TUNING PARAMETERS %
    %%%%%%%%%%%%%%%%%%%%%
    dT_MHE = dt_sys; % [h]
    N = 10; % number of sampling intervals within the estimation window
    
    % Objective function weights
    %Qy = diag([0.01, 1, 0.1]);
    %Qy = diag([1, 100, 10]);
    Qy = diag([1, 100, 0.001]);
    QyInv = inv(Qy);
    % overriding for integral error state
    %QyInv(3,3) = 0;
    
    Qx = diag([1, 100, 0.001]);
    QxInv = inv(Qx);
    
    Qtheta = [1, 0;
          0, 0.1];
    QthetaInv = inv(Qtheta);
    
    %%%%%%%%%%%%%%%%%
    % Declaring NLP %
    %%%%%%%%%%%%%%%%%
    % Start with an empty NLP
    w = {};
    g={};

    % Preparing NLP bounds
    MHEbounds.lbw = [];
    MHEbounds.ubw = [];
    MHEbounds.lbg = [];
    MHEbounds.ubg = [];

    % NLP parameters
    % plant measuremments
    X_in = MX.sym('X_in', N*dim_sys.nx);
    X = reshape(X_in,[N,dim_sys.nx]);
    
    % plant inputs
    U_in = MX.sym('U_in', N*dim_sys.nu);
    U = reshape(U_in,[N,dim_sys.nu]);
    
    % parameter estimates @ beginning of the window
    theta0_in = MX.sym('theta0_in',dim_sys.ntheta);
    
    % state estimates @ beginning of the window
    X0_in = MX.sym('X0_in',dim_sys.nx);
    
    % arrival cost
    P_mat_in = MX.sym('P_mat_in',(dim_sys.nx+dim_sys.ntheta)^2);
    P_mat = reshape(P_mat_in,[dim_sys.nx+dim_sys.ntheta,dim_sys.nx+dim_sys.ntheta]);
    
    % initial condition
    Xk = MX.sym('X0', dim_sys.nx);
    Pk = MX.sym('P0', dim_sys.ntheta);
    w = {w{:}, Xk, Pk};
    MHEbounds.lbw = [MHEbounds.lbw; sb.X_lb; sb.theta_lb];
    MHEbounds.ubw = [MHEbounds.ubw; sb.X_ub; sb.theta_ub];
    
    % arrival cost
    J = ([Xk - X0_in; Pk - theta0_in])'*P_mat*([Xk - X0_in; Pk - theta0_in]);
    
    for k=0:N-1
        
        % recovering measurements and parameter (estimable + fixed)
        Yk = X(k+1, :)';
        Uk = U(k+1, :)';
        thetak = [theta_nom(1);theta_nom(2);Pk{1};theta_nom(4);Pk{2};theta_nom(6);theta_nom(7)];
    
        % Integrate till the end of the interval
        Fk = F('x0',Xk,'p', [Uk;thetak;dT_MHE;zeros(2,1);0]); % last two inputs are dummy here (no bias and DU is used only in the economic layer)
        Xk_end = Fk.xf;
        Pk_end = MX.sym(['P_' num2str(k+1)],dim_sys.ntheta); % assuming random walk model
    
        % New NLP variable for the estimation
        Xk = MX.sym(['X_' num2str(k+1)],dim_sys.nx);
        w = {w{:}, Xk, Pk_end};
        MHEbounds.lbw = [MHEbounds.lbw; sb.X_lb; sb.theta_lb];
        MHEbounds.ubw = [MHEbounds.ubw; sb.X_ub; sb.theta_ub];
    
        % Add continuinity gap constraints
        g = {g{:}, Xk_end-Xk};
        MHEbounds.lbg = [MHEbounds.lbg; zeros(dim_sys.nx,1)];
        MHEbounds.ubg = [MHEbounds.ubg; zeros(dim_sys.nx,1)];    
        
        % Adding terms to the objective function
        J = J + (Yk - Xk_end)'*QyInv*(Yk - Xk_end) + (Pk_end - Pk)'*QthetaInv*(Pk_end - Pk);
    
        % looping parameters
        Pk = Pk_end;
    
    end
    
    % Create an NLP solver
    opts = struct;
    opts.ipopt.max_iter = 1000;
    opts.ipopt.print_level = 0;
    opts.print_time = 0;
    
    prob = struct('f', J, 'x', vertcat(w{:}),'g',vertcat(g{:}),'p',vertcat(X_in,U_in,theta0_in,X0_in,P_mat_in));
    solver = nlpsol('solver', 'ipopt', prob,opts);
end

function Pkk = ArrivalCost(Xhatk,thetaHatk,Pkk,Ukp,theta_p,dt_sys,dim_sys,Qx,Qtheta,Qy,F,S_xx,S_xp)
    
    %%%%%%%%%%%%%%%%%%%%%
    % TUNING PARAMETERS %
    %%%%%%%%%%%%%%%%%%%%%
    Qe = [Qx, zeros(dim_sys.nx,dim_sys.ntheta);
        zeros(dim_sys.ntheta,dim_sys.nx), Qtheta];

    %%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATING ARRIVAL COST %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % plant parameter value + estimates
    thetaHatLong = [theta_p(1);
        theta_p(2);
        thetaHatk(1);
        theta_p(4);
        thetaHatk(2);
        theta_p(6);
        theta_p(7)];
    
    % evolving system in time
    Fk = F('x0', Xhatk, 'p', [Ukp;thetaHatLong;dt_sys;zeros(dim_sys.nx - 1,1);0]);
    Xhatkk = full(Fk.xf);

    % last two inputs are dummy (only for computing economic OF)
    F_xx = full(S_xx(Xhatkk,[Ukp;thetaHatLong;dt_sys;zeros(dim_sys.nx - 1,1);0]));
    F_xp = full(S_xp(Xhatkk,[Ukp;thetaHatLong;dt_sys;zeros(dim_sys.nx - 1,1);0]));
    F_xp(:,10:14) = [];%excluding derivative in relation to par.T and bias
    F_xp(:,1:7) = [];%excluding derivative in relation to inputs
    
    %Arranging filter matrices - extended state vector
    Fk_1 = [F_xx, F_xp; zeros(dim_sys.ntheta,dim_sys.nx), eye(dim_sys.ntheta)];
    
    %output transition (all states are measured)
    Hk = [eye(dim_sys.nx),zeros(dim_sys.nx,dim_sys.ntheta)];
    
    %predicted covariance estimate
    Pk_1k_1 = Pkk; % loop
    Pkk_1 = Fk_1*Pk_1k_1*Fk_1'+ Qe;
    
    %Updating
    %Innovation covariance
    Sk = Hk*Pkk_1*Hk' + Qy;
    
    %Kalman Gain
    Kk = Pkk_1*Hk'*pinv(Sk);
    
    %Updating covariance estimate
    %Galo's upload system! Reduced Matrix is used only in the gain calculations
    Pkk = (eye(dim_sys.nx + dim_sys.ntheta) - Kk*Hk)*Pkk_1*((eye(dim_sys.nx + dim_sys.ntheta) - Kk*Hk))' + Kk*Qy*Kk';
    
    %guaranteeing symmetric matrix
    temp = Pkk;
    
    for i = 1:size(temp,1)
        for j = (i + 1):size(temp,2)
            temp(j,i) = Pkk(i,j);
        end
    end
    
    Pkk = temp;
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

function PlotResults(nameFile)
    
    load(nameFile)

    % time array
    tsimgrid = linspace(0, tEnd, nEnd+1);
    
    % labels
    xLab = {'C_A','T'};
    uLab = {'C_{A,f}','T_f','T_{SP}'};
    thetaLab = {'\Delta H','\Psi'};
    
    %%%%%%%%%%%%%%%%
    % Optimization %
    %%%%%%%%%%%%%%%%
    % SS Opt. (previously computed)
    uStarSS = [12.037;306.69;400];
    xStarSS = [0.94761;400];
    OF_SS = dt_sys*0.921277660957634;

    %%%%%%%%%%
    % STATES %
    %%%%%%%%%%
    figure(1)
    sgtitle('States') 
    for ii = 1:2
        subplot(3,1,ii)
        plot(tsimgrid,XPlantArray(ii,:),'r--','LineWidth',1.5)
        hold on
        plot(tsimgrid,XHatArray(ii,:),'kx-','LineWidth',1.5)
        yline(xStarSS(ii),'k:','LineWidth',1.0)
    
        xlim([0,tEnd])
    
        xlabel('t [h]')
        ylabel(xLab{ii})
    
        if ii == 1
            legend({'true','est.'},'Location','best')
        end
        grid on
    
    end

    subplot(3,1,3)
        plot(tsimgrid,XPlantArray(4,:),'r--','LineWidth',1.5)
    
        xlim([0,tEnd])
    
        xlabel('t [h]')
        ylabel('T_c')

        grid on
    
    %%%%%%%%%%
    % STATES %
    %%%%%%%%%%
    figure(2)
    sgtitle('Inputs') 
    for ii = 1:3
        subplot(3,1,ii)
        stairs(tsimgrid,UPlantArray(ii,:),'r','LineWidth',1.5)
        yline(uStarSS(ii),'k:','LineWidth',1.0)
        
        xlim([0,tEnd])
    
        xlabel('t [h]')
        ylabel(uLab{ii})
    
        grid on
    end
    
    %%%%%%%%%%%%%%
    % PARAMETERS %
    %%%%%%%%%%%%%%
    figure(3)
    sgtitle('Parameters') 
    for ii = 1:2
        subplot(2,1,ii)
        plot(tsimgrid,thetaPlantArray(ii,:),'r--','LineWidth',1.5)
        hold on
        plot(tsimgrid,thetaHatArray(ii,:),'kx','LineWidth',1.5)
    
        xlim([0,tEnd])
    
        xlabel('t [h]')
        ylabel('\Psi')
    
        if ii == 1
            legend({'true','est.'},'Location','best')
        end
    
        grid on
    end
    
    %%%%%%%%%%%%%%%%%%%
    % CONVERSION (OF) %
    %%%%%%%%%%%%%%%%%%%
    figure(4)
    
    plot(tsimgrid,OFPlantArray/dt_sys,'k')
    hold on
    yline(OF_SS/dt_sys,'k--','LineWidth',1.5)
    
    xlim([0,tEnd])
    
    xlabel('t [h]')
    ylabel('OF dyn')
    
    grid on
    
    %%%%%%%%%%%%%%%%
    % MHE SOLUTION %
    %%%%%%%%%%%%%%%%
    figure(5)
    sgtitle('Parameters trajectories') 
    for ii = 1:2
        subplot(2,1,ii)
        plot(tsimgrid,thetaPlantArray(ii,:),'r--','LineWidth',1.5)
        hold on
        for kk = Ne + dRTO_exec:dRTO_exec:nEnd
            if SolMHEFlag((kk - (Ne + dRTO_exec))/dRTO_exec + 1) == 1
                timeTemp = dt_sys*(kk - Ne):dt_sys:dt_sys*kk;
                plot(timeTemp,thetaHatTrajectory{kk}(ii,:),'Color',[0, 0, 0.01])
            end
        end
    
        xlim([0,tEnd])
    
        xlabel('t [h]')
        ylabel(thetaLab{ii})
    
        grid on
    end
    
    figure(6)
    sgtitle('States trajectories') 
    for ii = 1:2
        subplot(2,1,ii)
        plot(tsimgrid,XPlantArray(ii,:),'r--','LineWidth',1.5)
        hold on
        for kk = Ne + dRTO_exec:dRTO_exec:nEnd
            if SolMHEFlag((kk - (Ne + dRTO_exec))/dRTO_exec + 1) == 1
                timeTemp = dt_sys*(kk - Ne):dt_sys:dt_sys*kk;
                plot(timeTemp,xHatTrajectory{kk}(ii,:),'Color',[0, 0, 0.01])
            end
        end
        
        xlim([0,tEnd])
    
        xlabel('t [h]')
        ylabel(xLab{ii})
    
        grid on
    end
    
    %%%%%%%%%%%%%%%%%
    % DRTO SOLUTION %
    %%%%%%%%%%%%%%%%%
    figure(7)
    sgtitle('Input trajectories') 
    for ii = 1:3
        subplot(3,1,ii)
        plot(tsimgrid,UPlantArray(ii,:),'r--','LineWidth',1.5)
        hold on
        for kk = Ne + dRTO_exec:dRTO_exec:nEnd
            if SolDRTOFlag((kk - (Ne + dRTO_exec))/dRTO_exec + 1) == 1
                timeTemp = dt_sys*kk:dt_sys:dt_sys*(kk + Np - 1);
                plot(timeTemp,uDRTOTrajectory{kk}(ii,:),'Color',[0, 0, 0.01])
            end
        end
    
        xlim([0,tEnd + Np])
    
        xlabel('t [h]')
        ylabel(uLab{ii})
    
        grid on
    end
    
    %%%%%%%%%%%%%%
    % CONVERGED? %
    %%%%%%%%%%%%%%
    figure(8)
    subplot(2,1,1)    
        plot(tsimgrid(Ne + dRTO_exec:dRTO_exec:nEnd),SolMHEFlag,'kx','markersize',8)
    
        yticks([0, 1])
        yticklabels({'No','Yes'})
        
        xlim([0,tEnd])
        ylim([-0.1, 1.1])
        title('MHE Solution')
        grid on
    
    subplot(2,1,2)    
        plot(tsimgrid(Ne + dRTO_exec:dRTO_exec:nEnd),SolDRTOFlag,'kx','markersize',8)
    
        yticks([0, 1])
        yticklabels({'No','Yes'})
        
        xlim([0,tEnd])
        ylim([-0.1, 1.1])
        title('DRTO Solution')
        grid on
    
    %%%%%%%%%%%%%%%%%%
    % EXECUTION TIME %
    %%%%%%%%%%%%%%%%%%
    figure(9)
    subplot(3,1,1)    
        plot(tsimgrid(Ne + dRTO_exec:dRTO_exec:nEnd),execTimeArray(1,:),'kx-','markersize',5)
    
        title('MHE Solution Time')
        grid on
    
        xlim([0,tEnd])
    
    subplot(3,1,2)    
        plot(tsimgrid(Ne + dRTO_exec:dRTO_exec:nEnd),execTimeArray(2,:),'kx-','markersize',5)
    
        title('DRTO Solution Time')
        grid on
    
        xlim([0,tEnd + Np])
    
    subplot(3,1,3)    
        plot(tsimgrid(Ne + dRTO_exec:dRTO_exec:nEnd),100*execTimeArray(2,:)./(execTimeArray(1,:) + execTimeArray(2,:)),'kx-','markersize',5)
    
        title('DRTO Solution Time (%)')
        grid on
    
        xlim([0,tEnd])

end
