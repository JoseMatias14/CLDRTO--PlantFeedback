clear 
clc
close all

import casadi.*

%% Building SS model
% Declare model states
C_A = SX.sym('C_A'); 
T = SX.sym('T'); 
x = [C_A; T];

% Declare model inputs
CAf = SX.sym('CAf'); 
Tf = SX.sym('Tf');
Tc = SX.sym('Tc'); 
u = [CAf; Tf; Tc];

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

theta = [F; V; dH; rhoCp; UA; Psi; Omega];

% Reaction rate constant (rearranged as in Statistical assessment of chemical kinetic models (1975) - D.J.Pritchard and D.W.Bacon)
Tref = 373;          % TrefReference time for reparametrization [K]
k = exp(Omega + (Tref/T - 1)*Psi);

% Reaction rate per unit of volume, and it is described by the Arrhenius rate law, as follows.
r = k*C_A;

% ! Mass balances (assuming density is constant and perfect control)
% Component balances
dCAdt = (F/V)*(CAf - C_A) - r;
% Energy Balance
dTdt = (F/V)*(Tf - T) - ((dH*10^3)/(rhoCp*10^2))*r - (UA*10^2)/(V*rhoCp*10^2)*(T - Tc);

xdot = vertcat(dCAdt,dTdt);

% Objective term
% conversion
% moles of A reacted vs. moles of A fed
% and a penalty for the reactor temperature
L = (1 - C_A/CAf);

%% Steady-state optimization

% Declaring parameters
% parameters
E_nom = 11843;	     % E: Activation energy per mole [kcal/kmol]	
R_nom = 1.985875;    % R: Boltzmann's ideal gas constant [kcal/(kmol·K)]	
k0_nom = 34930800;	 % Pre-exponential nonthermal factor [1/h]	
% reparametrization
Psi_nom = - E_nom/(R_nom*Tref);
Omega_nom = log(k0_nom) + Psi_nom;

theta0 = [1;           % F: Volumetric flow rate [m3/h]
          1;           % V: Reactor Volume [m3] 
          -5.960;      % dH: Heat of reaction per mole [1000 kcal/kmol]	
          5;	       % rhoCp: Density multiplied by heat capacity: [100 kcal/(m3·K)]	
          1.5;	       % UA: Overall heat transfer coefficient multiplied by tank area: [100 kcal/(K·h)]
          Psi_nom;
          Omega_nom];

% Declaring bounds
% States Bounds
X_lb = [1e-6;    % C_A [kmol/m3] 
        250.0];   % T [K] 

X_ub = [14.0;    % C_A [kmol/m3] 
        400.0];   % T [K] 

% Inputs Bounds
U_lb = [6.0;     % CAf [kmol/m3] 
        260.0;   % Tf [K] 
        200.0];   % Tc [K] 

U_ub = [15.0;     % CAf [kmol/m3] 
        340.0;    % Tf [K] 
        330.0];    % Tc [K] 

% number of different input values
nSim = 500;

% initial condition is given
X_nom = [8.5691;    % C_A: Concentration of A [kmol/m3] 
      311.2740];  % T: Reactor temperature [K] 

U_nom = [10;       % CAf: Concentration of A in the feed [kmol/m3] 
      300;      % Tf: Feed temperature [K] 
      292];      % Tc: Coolant temperature[K] | STEP: - 10


% SS Opt and Dynamic Opt
temp1 = [[0.94761;400],[1.157338191402323;400]];
temp2 = [[12.037;306.69;270.41],[14.635849122816744;2.840706169358707e+02;2.507257430324447e+02]];

% drawing values from uniform distribution
temp3 = U_lb + (U_ub - U_lb).*rand(3,nSim);

% initial guesses array
W0Array = [[X_nom;U_nom],[temp1;temp2],[repmat(X_nom,[1, nSim]);temp3]];
% Solution array
WStarArray = [];
JStarArray = [];


for kk = 1:nSim + 1

    % preparing symbolic variable
    w = {};
    % preparing bounds
    lbw = [];
    ubw = [];

    % declaring symbolic variables
    w = {w{:},x,u};
    % specifying bounds
    lbw = [lbw;X_lb;W0Array(3:5,kk)];
    ubw = [ubw;X_ub;W0Array(3:5,kk)];

    % preparing symbolic constraints
    g = {};
    % preparing bounds
    lbg = [];
    ubg = [];

    %declaring constraints
    g = {g{:},xdot};
    % specifying bounds
    lbg = [lbg;0;0];
    ubg = [ubg;0;0];
    % Note: Equality constraint is defined by setting ub = lb with the same
    % value

    %Optimization objective function
    J = 0; %feasibility

    % formalize it into an NLP problem
    nlp = struct('x',vertcat(w{:}),'f',J,'g',vertcat(g{:}),'p',theta);

    % Assign solver
    solver = nlpsol('solver','ipopt',nlp);

    % Solve
    sol = solver('x0',W0Array(:,kk),'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,'p',theta0);

    % Extract Solution
    wopt = full(sol.x);
    % catch error
    % solution succeeded
    if solver.stats.success == 1
         WStarArray = [WStarArray, wopt];
         JStarArray = [JStarArray, 1 - wopt(1)/wopt(3)];
    end

end

%% Showing nominal results
t = table(lbw, WStarArray(:,1), ubw, 'VariableNames', {'LB', 'Opt.', 'UB'})


%% Plotting
figure(1)

plot(WStarArray(1,1),WStarArray(2,1),'ro','MarkerSize',5)
hold on 
plot(WStarArray(1,2:end),WStarArray(2,2:end),'kx','MarkerSize',5)
grid on

xlim([X_lb(1),X_ub(1)])
ylim([X_lb(2),X_ub(2)])

xlabel('Ca [kmol/m3] ')
ylabel('T [K] ')


figure(2)

scatter3(WStarArray(3,:),WStarArray(4,:),WStarArray(5,:),40,JStarArray)    % draw the scatter plot

hold on
count = 0;
for ii = 1:length(JStarArray)
    if JStarArray(ii) >= 0.9212 %% SS Opt (previously computed)
        scatter3(WStarArray(3,ii),WStarArray(4,ii),WStarArray(5,ii),'red','x')    % draw the scatter plot
        count = count + 1;
    end
end
count


grid on

xlim([U_lb(1),U_ub(1)])
ylim([U_lb(2),U_ub(2)])
zlim([U_lb(3),U_ub(3)])

xlabel('Caf [kmol/m3] ')
ylabel('Tf [K] ')
zlabel('Tc [K] ')
title('All Points')
view(-31,14)
cb = colorbar;                                  
cb.Label.String = 'Conversion';

















