clear 
clc
close all

import casadi.*

%% Building dynamic model
% Declare model states
C_A = SX.sym('C_A');
T = SX.sym('T'); 
I_T = SX.sym('I_T');
x = [C_A; T; I_T];

% Declare model inputs
CAf = SX.sym('CAf'); 
Tf = SX.sym('Tf');
Tsp = SX.sym('Tsp');
u = [CAf; Tf; Tsp];

% Declare model disturbances and parameters
dH = SX.sym('dH'); 
rhoCp = SX.sym('rhoCp');
UA = SX.sym('UA'); 
Psi = SX.sym('Psi');
Omega = SX.sym('Omega');

% Feedrate
F = SX.sym('F'); % [m3/h]

% Reactor volume
V = SX.sym('V'); % [m3]

% time transformation: CASADI integrates always from 0 to 1
% and the USER specifies the time
% scaling with step length
h = SX.sym('h'); % [h]

theta = [F; V; dH; rhoCp; UA; Psi; Omega; h];

% Reaction rate constant (rearranged as in Statistical assessment of chemical kinetic models (1975) - D.J.Pritchard and D.W.Bacon)
Tref = 373;          % TrefReference time for reparametrization [K]
k = exp(Omega + (Tref/T - 1)*Psi);

% Reaction rate per unit of volume, and it is described by the Arrhenius rate law, as follows.
r = k*C_A;

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

e_T = T - Tsp; % error: deviation from SP
% MV
Tc = Tc_nom + K_T*(e_T + I_T/tau_T); 

% ! Mass balances (assuming density is constant and perfect control)
% Component balances
dCAdt = (F/V)*(CAf - C_A) - r;
% Energy Balance
dTdt = (F/V)*(Tf - T) - ((dH*10^3)/(rhoCp*10^2))*r - (UA*10^2)/(V*rhoCp*10^2)*(T - Tc);
% controller error integration
dx_Edt = e_T;

xdot = vertcat(dCAdt,dTdt,dx_Edt);

% Objective term
% adding bias model
bias_m = SX.sym('bias_m',2); % Ca, T

% adding cost related to Caf usage
DU_Caf = SX.sym('DU_Caf'); % Ca, T

% Economic term
% conversion: moles of A reacted vs. moles of A fed
L = (1 - (C_A + bias_m{1})/CAf) - DU_Caf*(1.3*1e-2)*DU_Caf; %5*1e-3

% ! end modeling

%%%%%%%%%%%%%%%%%%%%%%%
% Creating integrator %
%%%%%%%%%%%%%%%%%%%%%%%%
ode = struct('x',x,'p',vertcat(u,theta,bias_m,DU_Caf),'ode',h*xdot,'quad',h*L); 
F = integrator('F', 'cvodes', ode);

%% Building DRTO
Tpred = 20; % Time horizon 4 | 16 | 16
N = 10; % number of control intervals 10 | 40 | 20

% States Bounds
X_lb = [1e-6;    % C_A [kmol/m3] 
        250.0;   % T [K] 
        -inf];   % integral error [h]

X_ub = [14.0;    % C_A [kmol/m3] 
        400.0;   % T [K] 
        inf];    % integral error [h]
nx = length(X_lb);

% Inputs Bounds
U_lb = [6.0;     % CAf [kmol/m3] 
        260.0;   % Tf [K] 
        250.0];   % Tsp [K] 

U_ub = [14.0;     % CAf [kmol/m3] 
        340.0;    % Tf [K] 
        400.0];    % Tsp [K] 
nu = length(U_lb);

% parameters
E_nom = 11843*1.1;	     % E: Activation energy per mole [kcal/kmol]	
R_nom = 1.985875;    % R: Boltzmann's ideal gas constant [kcal/(kmol·K)]	

k0_nom = 34930800;	 % Pre-exponential nonthermal factor [1/h]	
% reparametrization
Tref = 373;          % TrefReference time for reparametrization [K]
Psi_nom = - E_nom/(R_nom*Tref);
Omega_nom = log(k0_nom) + Psi_nom;

theta0 = [1;           % Fin: Volumetric flow rate [m3/h]	
          1;           % V: Reactor volume [m3]
          -5.960;      % dH: Heat of reaction per mole [1000 kcal/kmol]	
          5;	       % rhoCp: Density multiplied by heat capacity [100 kcal/(m3·K)]	
          1.5;	       % UA: Overall heat transfer coefficient multiplied by tank area [100 kcal/(K·h)]
          Psi_nom;
          Omega_nom];          
ntheta = length(theta0);

% Declaring NLP
% Start with an empty NLP
w = {};
J = 0;
g={};

% Formulate the NLP

% NLP parameters
par = MX.sym('par',ntheta);
bias = MX.sym('bias',nx  - 1); % all states are measured (integral error does not count)

%  initial condition is consider an NLP parameter
X0 = MX.sym('X0', nx);
Xk = X0;

% previous input is known
Uprev = MX.sym('Uprev');
U_1 = Uprev;

for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)],nu);
    w = {w{:}, Uk};

    % computing DU (only for Caf)
    DU = Uk{1} - U_1;

    % Integrate till the end of the interval
    Fk = F('x0',Xk,'p', [Uk;par;Tpred/N;bias;DU]);
    Xk_end = Fk.xf;
    J = J+Fk.qf;

    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], nx);
    w = {w{:}, Xk};
    
    % adding bias to the states constraints
    g = {g{:}, Xk + vertcat(bias{1},bias{2},0)}; % (integral error does not count)

    % Add equality constraint
    g = {g{:}, Xk_end-Xk};

end

% Create an NLP solver
opts = struct;
opts.ipopt.max_iter = 500;
opts.ipopt.print_level = 0;
opts.print_time = 0;

prob = struct('f', -J, 'x', vertcat(w{:}),'g',vertcat(g{:}),'p',vertcat(X0,Uprev,par,bias));
solver = nlpsol('solver', 'ipopt', prob,opts);

%% Monte Carlo simulations
% Controller sampling time
dT_CL = Tpred/N;

% simulation final value
tEnd = 100; %[h]
nEnd = tEnd/dT_CL;

% initial condition is given
X_nom = [8.5691;    % C_A: Concentration of A [kmol/m3] 
      311.2740;     % T: Reactor temperature [K] 
      0];           % I_T: integral error [h] 

U_nom = [8.5691;       % CAf: Concentration of A in the feed [kmol/m3] 10
      300;      % Tf: Feed temperature [K] 
      311.2740];      % Tsp: Reactor temperature setpoint [K] 

% plant parameter value
theta_p = theta0;

% SS Opt. (previously computed)
uStarSS = [12.037;306.69;270.41];
xStarSS = [0.94761;400];
OFStarSS = Tpred/N*0.921277660957634;

% Preparing bias updating simulation
% number of uncertain parameters
parIndex = [3, 5, 6, 7]; % position correspondence with theta0
nPar = 4; % dH, UA, Psi, Omega

% number of different parameters values drawn from the distribution
nSim = 20;

% mean is the nominal value, sigma is 0.001*nom
%Sig = [1; 1; 0.001; 1; 0.001; 0.001; 0.001];
%Sig = [1; 1; 0.01; 1; 0.01; 0.01; 0.01];
Sig = [1; 1; 0.05; 1; 0.05; 0.05; 0.05];
sigma_theta0 = Sig.*abs(theta0);
figure(1)
sgtitle('Parameter Distribution') 

thetaLab = {'F','V','\Delta H','\rho C_p','UA','\Psi','\Omega'};
for ii = 1:length(parIndex)
    % creating array to evaluate normal distribution
    x_temp = (theta0(parIndex(ii)) - 3*sigma_theta0(parIndex(ii))):(0.1*sigma_theta0(parIndex(ii))):(theta0(parIndex(ii)) + 3*sigma_theta0(parIndex(ii)));
    
    % evaluate normal dist. at points x_temp
    y_temp = normpdf(x_temp,theta0(parIndex(ii)),sigma_theta0(parIndex(ii)));

    % plotting
    subplot(2,3,ii)
        plot(x_temp,y_temp/sum(y_temp),'LineWidth',1.5), grid on, xlabel(thetaLab{parIndex(ii)}),ylabel('Probability Density')
        hold on 
        xline(theta0(parIndex(ii)),'k--','LineWidth',1.5)
        xline(theta0(parIndex(ii)) + sigma_theta0(parIndex(ii)),'k:','LineWidth',1.5)
        xline(theta0(parIndex(ii)) - sigma_theta0(parIndex(ii)),'k:','LineWidth',1.5)
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
thetaMC = zeros(nPar + 3,nSim + 3); %nominal, nominal + sigma, nominal - sigma, random

% nominal 
thetaMC(:,1) = theta0;
% nominal + sigma
thetaMC(:,2) = theta0 + sigma_theta0;
% nominal - sigma
thetaMC(:,3) = theta0 - sigma_theta0;

% drawing random numbers
for jj = 1:nSim
    % assign nominal parameters (from plant)
    thetaMC(:,3 + jj) = theta0;
    for ii = 1:nPar
        % draw parameter value
        parTemp = normrnd(theta0(parIndex(ii)),sigma_theta0(parIndex(ii)));
        % replace to create plant-model mismatch 
        thetaMC(parIndex(ii),3 + jj) = parTemp;
    end
end

% run MC simulation
for ii = 1:nPar
    for jj = 1:3 + nSim

        % assign parameter values from MC array
        theta_m = theta0;
        theta_m(parIndex(ii)) = thetaMC(parIndex(ii),jj);

        % initial value of the inputs
        Uk_p = U_nom;

        %initial condition is known
        Xk_p = X_nom;
        Xk_m = X_nom;
        biask = Xk_p(1:2) - Xk_m(1:2); 

        % first order filter for bias
        filBiask = 0.0;
            
        % for saving values
        XModelArray = Xk_m;
        XPlantArray = Xk_p;
        UPlantArray = Uk_p;
        OFPlantArray = 1 - Xk_p(1)/Uk_p(1);
        biasArray = biask;
        SolFlagArray = [];

        for kk = 1:nEnd
            fprintf('>>> Iteration: %d, Par: %d, Sim: %d \n',kk, ii, jj)
            % Preparing NLP bounds and initial guess
            w0 = [];
            lbw = [];
            ubw = [];
            lbg = [];
            ubg = [];

            for k=0:N-1
                % Control
                lbw = [lbw; U_lb];
                ubw = [ubw; U_ub];
                w0 = [w0;  Uk_p];

                % new NLP variable
                lbw = [lbw; -inf*ones(nx,1)];
                ubw = [ubw; inf*ones(nx,1)];
                w0 = [w0; Xk_m];

                % bias constraints
                lbg = [lbg; X_lb];
                ubg = [ubg; X_ub];

                % Equality constraints
                lbg = [lbg; zeros(nx,1)];
                ubg = [ubg; zeros(nx,1)];
            end

            % solve OCP with the wrong model
            sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, ...
                'lbg', lbg, 'ubg', ubg,'p',[Xk_m;Uk_p(1);theta_m;biask]);

            % catch error
            if solver.stats.success ~=1
                % solution failed
                solFlag = 0;
            else
                % solution succeeded
                solFlag = 1;
            end
            SolFlagArray = [SolFlagArray, solFlag];

            % save optimal inputs
            w_opt = full(sol.x);

            % receding horizon implementation
            Uk_star = w_opt(nx + 1:nx + nu);
            Uk_p = [Uk_star(1);Uk_star(2);Uk_star(3)];

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Apply optimal decision to the plant
            Fk = F('x0', Xk_p, 'p', [Uk_p;theta_p;dT_CL;zeros(nx - 1,1);0]);
            Xk_p = full(Fk.xf);
            OFk_p = full(Fk.qf);

            % Compute model state
            Fk = F('x0', Xk_m, 'p', [Uk_p;theta_m;dT_CL;zeros(nx - 1,1);0]);
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
save('MC_parameters_CL_highHigh.mat','thetaMC','XPlantPlot','XModelPlot','UPlot','OFPlot','biasPlot','SolFlagPlot','thetaModelPlot')

%% Plotting results
BiasUpdateAnalysis_3_Plot

