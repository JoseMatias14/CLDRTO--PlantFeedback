clear 
clc
close all

import casadi.*

Tsim = 20; % Time horizon
N = 20; % number of control intervals 5 | 10 | 40

%% Initilizing model and specifying initial condition (STEP 1)
% initial condition is given
X0 = [8.5691;    % C_A: Concentration of A [kmol/m3] 
      311.2740];  % T: Reactor temperature [K] 
nx = length(X0);

% States Bounds
X_lb = [1e-6;    % C_A [kmol/m3] 
        250.0];   % T [K] 

X_ub = [20.0;    % C_A [kmol/m3] 
        400.0];   % T [K] 

% System Inputs
U0 = [10;       % CAf: Concentration of A in the feed [kmol/m3] 
      300;      % Tf: Feed temperature [K] 
      292];      % Tc: Coolant temperature[K] | STEP: - 10
nu = length(U0);

% Inputs Bounds
U_lb = [5.0;     % CAf [kmol/m3] 
        260.0;   % Tf [K] 
        200.0];  % Tc [K] 

U_ub = [15.0;     % CAf [kmol/m3] 
        375.0;    % Tf [K] 
        355.0];    % Tc [K] 

% parameters
E_nom = 11843;	     % E: Activation energy per mole [kcal/kmol]	
R_nom = 1.985875;    % R: Boltzmann's ideal gas constant [kcal/(kmol·K)]	
Tref = 373;          % TrefReference time for reparametrization [K]
k0_nom = 34930800;	 % Pre-exponential nonthermal factor [1/h]	
% reparametrization
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

% Building plant model
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

% time transformation: CASADI integrates always from 0 to 1
% and the USER specifies the time
% scaling with step length
h = SX.sym('h'); % [h]

theta = [F; V; dH; rhoCp; UA; Psi; Omega; h];

% Reaction rate constant (rearranged as in Statistical assessment of chemical kinetic models (1975) - D.J.Pritchard and D.W.Bacon)
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
L = 1 - C_A/CAf;

% ! end modeling

%%%%%%%%%%%%%%%%%%%%%%%
% Creating integrator %
%%%%%%%%%%%%%%%%%%%%%%%%
ode = struct('x',x,'p',vertcat(u,theta),'ode',h*xdot,'quad',h*L);
F = integrator('F', 'cvodes', ode);

% sensitivities
S_xp = F.factory('sensParStates',{'x0','p'},{'jac:xf:p'});
S_Lp = F.factory('sensParStates',{'x0','p'},{'jac:qf:p'});

%% Computing sensitivities functions (STEP 2)
% for saving values
for jj = 1:nx
    SX_Array{jj} = zeros(ntheta - 2,1); % excluding Fin, V 
end
SL_rray = zeros(ntheta - 2,1); % excluding Fin, V 

% computing initial value of the OF
L0 =  1 - X0(1)/U0(1);

for kk = 1:N
    % states
    S_xpk = S_xp('x0', X0, 'p', [U0;theta0;kk*Tsim/N]);
    s_x_opt_temp = full(S_xpk.jac_xf_p);
    for jj = 1:nx
        %normalizing using initial/nominal value
        SX_Array{jj} = [SX_Array{jj}, (theta0(3:end)/X0(jj)).*(s_x_opt_temp(jj,nu+3:end - 1))']; % excluding Fin, V, and h
    end

    % objective function
    S_xpk = S_Lp('x0', X0, 'p', [U0;theta0;kk*Tsim/N]);
    s_q_opt_temp = full(S_xpk.jac_qf_p);
    %normalizing using initial/nominal value
    SL_rray = [SL_rray, (theta0(3:end)/L0).*(s_q_opt_temp(nu+3:end - 1))']; % excluding Fin, V, and h
end

% Plotting sensitivities
tsimgrid = linspace(0, Tsim, N+1);
xLab = {'C_A','T'};
thetaLab = {'\Delta H','\rho C_p','UA','\Psi','\Omega'};

% plotting sensitivities for each state
for jj = 1:nx 
    figure(jj)
    sgtitle(['State: ',xLab{jj}]) 
    for ii = 1:5
        subplot(2,3,ii)
            hold on 
            plot(tsimgrid,SX_Array{jj}(ii,:),'kx','LineWidth',1.5)
        
            xlim([0,Tsim])
        
            xlabel('t [h]')
            ylabel(thetaLab{ii})
            
            grid on
    end
end

%% Rank parameter significance (Step3)

% creating variables for saving
% delta mean squared value
dmsqr = zeros(ntheta - 2,nx); % excluding Fin, V 
% normalized sensitivity
for jj = 1:nx
    snorm{jj} = zeros(N,ntheta - 2); % excluding Fin, V 
end

for ii = 1:ntheta - 2 % for each parameter
    for jj = 1:nx % for each state
        % extracting normalized sensitivity (excluding initial point)
        temp1 = SX_Array{jj}(ii,2:end)';

        % delta mean squared value
        dmsqr(ii,jj) = sqrt((temp1'*temp1)/length(temp1)); 

        % normalized non-dimensional sensitivity function
        snorm{jj}(:,ii) = temp1/norm(temp1); 
    end
end

% plotting delta mean squared value 
figure(5)

thetaLab = {'\Delta H','\rho C_p','UA','\Psi','\Omega'};
xLab = {'C_A','T'};

sgtitle('Delta Mean-square Measure')
% a plot for each state
for jj = 1:nx
    subplot(nx,1,jj)
        %plotting the values for each parameter
        bar(dmsqr(:,jj),'w')
        yline(0.1,'k:')
    
    xticklabels(thetaLab)
    ylabel('\delta^{msqr}')
    
    title(xLab{jj})
    %ylim([0 10])
end

save('deltaMeanSqrData','dmsqr')

%% Compute collinearity index (step 4) 
% excluding volume and C_B
snormy = [snorm{1}; snorm{2}];

%total number of parameter combinations
comb_theta = [];
for r = 2:ntheta - 2 % all possible pairing of parameters with different sizes (2,3,4,...,ntheta) 
    comb_theta = [comb_theta, factorial(ntheta)/(factorial(ntheta - r)*factorial(r))];
end

% adding to get the total all possible combinations
count = sum(comb_theta) ;  

% saving values
countTable = 0; % index counter for building table with all possible combinations
SubsetK = []; % saving counter value
SubsetSize = []; % saving the number of parameters combined
SubsetCol = []; % saving colinearity index
%SubsetLabels = {}; % text labels for the subset

for cc = 2:ntheta % excluding Fin

    % all possible pairing of parameters with sets of size cc
    combos = combnk([1,2,3,4,5],cc); 
    
    % n = number of combinations, mm = size of the combinations 
    [ncomb, mm] = size(combos); 
    
    for ll = 1:ncomb
        % increasing counter (new line in the table used for showing the
        % results)
        countTable = countTable + 1;
        
        % find parameters that will be used in the combination
        pix=combos(ll,:);

        % select columns that represent those parameters
        tempn = snormy(:,pix) ; 

        % normalized sensitivity matrix
        nsm = tempn'*tempn;  
        
        % collinearity index
        SubsetCol = [SubsetCol; real(1/sqrt(min(eig(nsm))))]; 

        % current counter 
        SubsetK = [SubsetK; countTable]; 

        % subset size
        SubsetSize = [SubsetSize; cc]; 

        % labels
        SubsetLabels{countTable,:} = thetaLab(combos(ll,:));

    end
end
% tabulate the results of collinearity index. See Table5.8
TableColIndex = table(SubsetK,SubsetSize,SubsetLabels,SubsetCol,'VariableNames',{'SubsetK','SubsetSize','SubsetCombnts','GammaK'})


