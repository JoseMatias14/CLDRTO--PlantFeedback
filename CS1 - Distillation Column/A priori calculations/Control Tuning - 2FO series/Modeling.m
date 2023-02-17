% Author: Jose Otavio Matias
% Work address
% email: jose.o.a.matias@ntnu.no
% May 2020; Last revision: 2020-05-28
clear
close all
clc

nd = 8;
data_B_MB = readtable('data_B_MB.csv');
train_dat{1} = data_B_MB.Var1; %B
train_dat{2} = data_B_MB.Var2; %MB
data_D_MD = readtable('data_D_MD.csv');
train_dat{3} = data_D_MD.Var1; %D
train_dat{4} = data_D_MD.Var2; %MD
data_L_xD = readtable('data_L_xD.csv');
train_dat{5} = data_L_xD.Var1; %L
train_dat{6} = data_L_xD.Var2; %xD
train_dat{7} = data_L_xD.Var3; %xD (with delay)
train_dat{8} = data_L_xD.Var4; %xD (with more delay)

dt = 5; %[s]

%%  pre-processing the data
for ii = 1:nd
    data_norm{ii} = train_dat{ii} - train_dat{ii}(1);
end

plot(0:dt:(length(data_norm{6}) - 1)*dt,data_norm{6},'k:')
hold on 
plot(0:dt:(length(data_norm{6}) - 1)*dt,data_norm{7},'k--')
hold on
plot(0:dt:(length(data_norm{6}) - 1)*dt,data_norm{8},'k')
hold on
%% Estimating the model parameters L - xD
%%%%% NO DELAY
% num1 = 3.804157484859962e-05/5.841416817344758e-04;
% den1 = [1/5.841416817344758e-04 1];
% P1 = tf(num1,den1,'InputDelay',0);

%%%%% DELAY 1
% num1 = 4.016800074033301e-05/6.190579996267742e-04;
% den1 = [1/6.190579996267742e-04 1];
% P1 = tf(num1,den1,'InputDelay',145);

%%%%% DELAY 2
num1 = 3.591031902266622e-05/(5.490428584768014e-04);
den1 = [1/(5.490428584768014e-04) 1];
P1 = tf(num1,den1,'InputDelay',145*dt);

figure(1)
step(P1*0.1283) % scaled step size
hold on
% figure()
%plot(0:(length(data_norm{7}(61:end)) - 1),data_norm{7}(61:end),'r:')
plot(0:5:(length(data_norm{8}(61:end)) - 1)*5,data_norm{8}(61:end),'r:')

%% Estimating the model parameters D - MD
num2 = -1.005;
den2 = [1 0];
P2 = tf(num2,den2);
figure(2)
step(P2*0.0248)
hold on
% figure()
plot(0:1/12:(length(data_norm{4}(61:end)) - 1)/12,data_norm{4}(61:end),'r:')
xlim([0, (length(data_norm{4}(61:end)) - 1)/12])

%% Estimating the model parameters B - MB
num3 = -510;
den3 = [5*1e2 1];
P3 = tf(num3,den3);
figure(3)
step(P3*0.0252)
hold on
% figure()
plot(0:1/12:(length(data_norm{2}(61:end)) - 1)/12,data_norm{2}(61:end),'r:')
xlim([0, (length(data_norm{2}(61:end)) - 1)/12])

%% Control Example
