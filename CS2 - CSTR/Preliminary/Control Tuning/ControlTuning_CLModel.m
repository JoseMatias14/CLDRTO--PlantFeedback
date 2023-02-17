clear 
clc
close all

import casadi.*

Tsim = 20; % Time horizon
N = 40; % number of control intervals 5 | 10 | 40

% initial condition is given
X0 = [8.5691;    % C_A: Concentration of A [kmol/m3]
      311.2740;  % T: Reactor temperature [K] 
      0]; 	       % e_T: SP deviation 

U0 = [10;    % CAf: Concentration of A in the feed [kmol/m3] 
      300;       % Tf: Feed temperature [K] 
      400];  % TSP: Reactor temperature setpoint [K] 
      
% parameters
E_nom = 11843;	     % E: Activation energy per mole [kcal/kmol]	
R_nom = 1.985875;    % R: Boltzmann's ideal gas constant [kcal/(kmol·K)]	
Tref = 373;          % TrefReference time for reparametrization [K]
k0_nom = 34930800;	 % Pre-exponential nonthermal factor [1/h]	
% reparametrization
Psi_nom = - E_nom/(R_nom*Tref);
Omega_nom = log(k0_nom) + Psi_nom;

theta0 = [1;           % F: Volumetric flow rate [m3/h]	
          1;           % V: Reactor volume [m3]
          -5.960;      % dH: Heat of reaction per mole [1000 kcal/kmol]	
          5;	       % rhoCp: Density multiplied by heat capacity [100 kcal/(m3·K)]	
          1.5;	       % UA: Overall heat transfer coefficient multiplied by tank area [100 kcal/(K·h)]
          Psi_nom;
          Omega_nom];

% declaring controller parameters (from SIMC rules)
% Estimated model parameters using sysid toolbox 
% FROM ControlTuning.m file --> model with 4 states
num1 = 0.1626/0.7301;
den1 = [1/-0.7301 1];

% k' = k/tau_1
% Kc = 1/k'*1/(theta + tau_c)
% tau_I = min(tau_1,4*(tau_c + theta))
% tuning parameter
tau_c = 3;
kprime = num1/den1(1);
KcT = 1/kprime*1/(0 + tau_c);
tauT = -min(den1(1),4*(tau_c + 0));

% Loop: temperature - coolant
% Proportional Gain
% K_T = -1.537515375153752; % [-]
K_T = KcT; % [-]
% Integral Constant
% tau_T = 1.369675386933297; % [h]
tau_T = tauT; % [h]
% Nominal volume 
T_nom = 311.2639;   % [K]
% Nominal flow
Tc_nom = 292;     % [K]

%% Building plant model
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
k = exp(Omega + (Tref/T - 1)*Psi);

% Reaction rate per unit of volume, and it is described by the Arrhenius rate law, as follows.
r = k*C_A;

% Control action (pair: T <=> Tc )
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
% economic (maximize B)
L = 1 - C_A/CAf;

% ! end modeling

%%%%%%%%%%%%%%%%%%%%%%%
% Creating integrator %
%%%%%%%%%%%%%%%%%%%%%%%%
ode = struct('x',x,'p',vertcat(u,theta),'ode',h*xdot,'quad',h*L);
F = integrator('F', 'cvodes', ode);

%% Simulation closed-loop with PI controller

% finding controller sampling time
NCL = 100; 
dT_Cont = Tsim/NCL;

% for saving values
XCLArray = X0;
UCLArray = [];

% initializing states
Xk = X0;
Uk = U0;

for kk = 0:NCL-1

    % Apply optimal decision to the plant
    Fk = F('x0', Xk, 'p', [Uk;theta0;Tsim/NCL]);
    Xk = full(Fk.xf);

	% saving values
    XCLArray = [XCLArray, Xk];
    UCLArray = [UCLArray, U0];

	% adding disturbance
	if kk > NCL/2
		theta0(5) = 7.5;
	end

end

%% Plotting results
tsimgrid = linspace(0, Tsim, NCL+1);
xLab = {'C_A','T','x_E'};
uLab = {'C_{A,f}','T_f','T_{SP}'};

% rebuilding coolant temperature
TcArray = Tc_nom + K_T*((XCLArray(2,:) - 400) + XCLArray(3,:)/tau_T); 

figure(1)
sgtitle('States') 
for ii = 1:3
    subplot(5,1,ii)
        hold on 
        plot(tsimgrid,XCLArray(ii,:),'k','LineWidth',1.5)
        
        if ii == 2
            yline(400,'k--','LineWidth',1.5)
        end
        
        xlim([0,Tsim])
    
        xlabel('t [h]')
        ylabel(xLab{ii})
        
        grid on
end

subplot(5,1,4)
        plot(tsimgrid,XCLArray(2,:) - 400,'k','LineWidth',1.5)
    
        xlim([0,Tsim])
    
        xlabel('t [h]')
        ylabel('T - T_{SP}')
	    title('Error')
        
        grid on

subplot(5,1,5)
        stairs(tsimgrid,TcArray,'k','LineWidth',1.5)
    
        xlim([0,Tsim])
    
        xlabel('t [h]')
        ylabel('Tc')
	   title('MV')
        
        grid on

