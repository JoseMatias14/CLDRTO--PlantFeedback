clear 
clc
%close all

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
E_nom = 11843*1.1;	     % E: Activation energy per mole [kcal/kmol]	
R_nom = 1.985875;    % R: Boltzmann's ideal gas constant [kcal/(kmol·K)]	
k0_nom = 34930800;	 % Pre-exponential nonthermal factor [1/h]	
% reparametrization
Psi_nom = - E_nom/(R_nom*Tref);
Omega_nom = log(k0_nom) + Psi_nom;

theta0 = [1;           % F: Volumetric flow rate [m3/h]
          1;           % V: Reactor Volume [m3] 
          -5.960;      % dH: Heat of reaction per mole [1000 kcal/kmol]	
          5;	       % rhoCp: Density multiplied by heat capacity: [100 kcal/(m3·K)]	
          1.6;	       % UA: Overall heat transfer coefficient multiplied by tank area: [100 kcal/(K·h)]
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
        250.0];   % Tc [K] 

U_ub = [14.0;     % CAf [kmol/m3] 
        340.0;    % Tf [K] 
        400.0];    % Tc [K] 

% preparing symbolic variable
w = {};
% preparing bounds
lbw = [];
ubw = [];

% declaring symbolic variables
w = {w{:},x,u};
% specifying bounds
lbw = [lbw;X_lb;U_lb];
ubw = [ubw;X_ub;U_ub];

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
J = -L; %max

% formalize it into an NLP problem
nlp = struct('x',vertcat(w{:}),'f',J,'g',vertcat(g{:}),'p',theta);

% Assign solver
solver = nlpsol('solver','ipopt',nlp);

%% Solving

% number of different initial guesses used
nSim = 100;

% initial condition is given
X_nom = [8.5691;    % C_A: Concentration of A [kmol/m3] 
      311.2740];  % T: Reactor temperature [K] 

U_nom = [8.5691;       % CAf: Concentration of A in the feed [kmol/m3] 
      300;      % Tf: Feed temperature [K] 
      292];      % Tc: Coolant temperature[K] | STEP: - 10

% drawing values from uniform distribution
temp1 = [X_lb;U_lb] + ([X_ub;U_ub] - [X_lb;U_lb]).*rand(5,nSim);

% initial guesses array
W0Array = [[X_nom;U_nom],temp1];
% Solution array
WStarArray = [];
JStarArray = [];


for kk = 1:nSim + 1

    % Solve
    sol = solver('x0',W0Array(:,kk),'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,'p',theta0);

    % Extract Solution  
    WStarArray = [WStarArray, full(sol.x)];
    JStarArray = [JStarArray, full(sol.f)];


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

scatter3(WStarArray(3,1),WStarArray(4,1),WStarArray(5,1),'red')
hold on 
scatter3(WStarArray(3,2:end),WStarArray(4,2:end),WStarArray(5,2:end),nSim,[0 0 0],'x')
grid on

xlim([U_lb(1),U_ub(1)])
ylim([U_lb(2),U_ub(2)])
zlim([U_lb(3),U_ub(3)])

xlabel('Caf [kmol/m3] ')
ylabel('Tf [K] ')
zlabel('Tc [K] ')













