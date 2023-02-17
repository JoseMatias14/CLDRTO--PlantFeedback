clear 
close all
clc

%% orginal process
s = tf('s');
% higher-order function
g = (-0.3*s+1)*(0.08*s + 1)/((2*s + 1)*(s + 1)*(0.4*s + 1)*(0.2*s + 1)*(0.05*s + 1)^3);

%% Half-rule Approximation
% approximated as a first order delay process with
k = 1;
tau1 = 2.5; tau2 = 0; theta = 1.47; 
% approximated as a second order delay process 
kSO = 1;
tau1SO = 2; tau2SO = 1.2; thetaSO = 0.77; 

%N.B.: for tuning a PI controller, you want to approximate as a first-order
% plus time delay model

%% SIMC PI and PID tunings
% for SIMC
%%%%%%
% PI %
%%%%%%
% Desired closed-loop tuning constant tauc >= theta
% The smallest value for fast control is the lower bound
tauc = theta;
% tuning proportional gain 
Kc = (1/k)*tau1/(tauc + theta); %PI: 0.85, PID: 1.30
% tuning integral time
% for IMC; taui == tau1
taui = min(tau1,4*(tauc + theta)); %PI: 2.50, PID:2
% tuning derivative time
taud = tau2;

% Building the controler
% P + I
cpi = Kc*((taui*s + 1)*(taud*s + 1)/(taui*s));

%%%%%%%
% PID %
%%%%%%%
taucSO = thetaSO;
KcSO = (1/kSO)*tau1SO/(taucSO + thetaSO); %PI: 0.85, PID: 1.30
% tuning integral time
tauiSO = min(tau1SO,4*(taucSO + thetaSO)); %PI: 2.50, PID:2
% tuning derivative time
taudSO = tau2SO;

% Building the controler
% P + I + D
cpiSO = KcSO*(1 + 1/(tauiSO*s));
cpidSO = cpiSO*(taudSO*s + 1)/(0.1*taudSO*s + 1);
%N.B.: with added filter for smoother control alpha = 0.1

%% Closed-loop function (building transfer functions)
% 1 - without D action
L = cpi*g; % loop function
S = inv(1 + L); % sensitivity function
%setpoint responses -
% minreal --> Minimal realization and pole-zero cancellation.
% y response for changes in ysp
Ty = g*cpi*S; Ty = minreal(Ty);
% u response for changes in ysp
Tuy = cpi*S; Tuy = minreal(Tuy); 
% PI response
Typi = Ty; 
Tuypi = Tuy;
%input disturbance
gd = g;
% y response for changes in d
Td = gd*S; Td = minreal(Td);
% u response for changes in d
Tud = gd*cpi*S; Tud = minreal(Tud);
% PI response
Tdpi=Td; 
Tudpi=Tud;

% 2 - with D action
L = cpidSO*g; % without D-action on setpoint
S = inv(1 + L);
%setpoint responses -
Ty = g*cpidSO*S; Ty = minreal(Ty);
Tuy = cpidSO*S; Tuy = minreal(Tuy); 
% PID response
Typid = Ty;
Tuypid = Tuy;
%input disturbance
gd = g;
Td = gd*S; Td = minreal(Td);
Tud = gd*cpidSO*S; Tud = minreal(Tud);
% PID response
Tdpid=Td;
Tudpid=Tud;

% responses overlap
% figure(1),step(Typi,'blue',Tuypi,'red',15)
% figure(2),step(Tdpi,'blue',Tudpi,'red',15)
% figure(1),step(Typid,'blue',Tuypid,'red',15)
% figure(2),step(Tdpid,'blue',Tudpid,'red',15)
figure(1),step(Typi,'blue',Typid,'blue--',Tuypi,'red',Tuypid,'red--',15),xlim([0.1, 15])
figure(2),step(Tdpi,'blue',Tdpid,'blue--',Tudpi,'red',Tudpid,'red--',15),xlim([0.1, 15])

