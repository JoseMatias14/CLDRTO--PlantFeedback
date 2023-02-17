clear 
close all
clc
s = tf('s');
% transfer functions from Modeling.m

%N.B.:
% SP ---| K |- u,d -| G |--- y
%     |                 |
%     ------------------

% for sp step
% H = feedback(G*K,1)
% H2 = G*K/(1+G*K)

% for input disturbance 
% H = feedback(G,K)
% H2 = G/(1+G*K)

%% L - xD
num1 = 4.016800074033301e-05/6.190579996267742e-04;
den1 = [1/6.190579996267742e-04 1];
g = tf(num1,den1,'InputDelay',145);

%figure(1)
%step(g,100)

% extracting system parameters
k = 3.591031902266622e-05/(5.490428584768014e-04);
tau1 = 1/(5.490428584768014e-04); tau2 = 0; theta = 145*5; 
% SIMC
% Desired closed-loop tuning constant tauc >= theta
% The smallest value for fast control is the lower bound
tauc = theta %  1.5*theta | 10*theta | 20*theta
% tuning proportional gain 
Kc = (1/k)*tau1/(tauc + theta)
% tuning integral time
% for IMC; taui == tau1
taui = min(tau1,4*(tauc + theta))
taud = 0;

% Building the controler 
% P + I 
cpi = Kc*((taui*s + 1)*(taud*s + 1)/(taui*s));
T_pi_simc = feedback(g*cpi, 1);

% tuning with matlab function
[C_pi,info] = pidtune(g,'PI');
T_pi = feedback(g*C_pi, 1);

figure(2)%setpoint change 0.001
opt = stepDataOptions('StepAmplitude',0.001);
step(T_pi,'r--',T_pi_simc,'b.',200*60,opt)

% Input Disturbance rejection
DR_pi = feedback(g,C_pi);
DR_pi_simc = feedback(g,cpi);

% Compare responses
figure(3)
opt = stepDataOptions('StepAmplitude',0.5);
step(DR_pi,'r--',DR_pi_simc,'b.',200*60,opt)

%% level (either D - MD or B - MB
num2 = -1.005;
den2 = [1 0];
g2 = tf(num2,den2);

figure(4)
step(0.025*g2,10)

% extracting system parameters
kLevel = -1.005;
thetaLevel = 0;
% N.B.: the integrating process is a special case of 1st order process
% where tau1 tends to infinity
% SIMC
taucLevel = 0.6; % 0.6
% tuning proportional gain 
KcLevel = (1/kLevel)*1/(taucLevel + thetaLevel) 
% tuning integral time
tauiLevel = 4*(taucLevel + thetaLevel)
taudLevel = 0;

% Building the controler 
% P + I 
cpiLevel = KcLevel*((tauiLevel*s + 1)*(taudLevel*s + 1)/(tauiLevel*s));
T_pi_level_simc = feedback(g2*cpiLevel, 1);

% tuning with matlab function
[C_pi_level,~] = pidtune(g2,'PI');
T_pi_Level = feedback(g2*C_pi_level, 1);

figure(5)%setpoint change 0.001
opt = stepDataOptions('StepAmplitude',0.025);
step(T_pi_Level,'r--',T_pi_level_simc,'b.',100,opt)

% Input Disturbance rejection
DR_pi_level = feedback(g2,C_pi_level);
DR_pi_simc_level = feedback(g2,cpiLevel);

% Compare responses
figure(6)
opt = stepDataOptions('StepAmplitude',0.1);
step(DR_pi_level,'r--',DR_pi_simc_level,'b.',100,opt)
