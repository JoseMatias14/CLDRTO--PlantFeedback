clear 
clc
close all


    
    a = load("test2_MHE_2p_un.mat");
    b = load("test2_BIAS_2p_un.mat");

    % time array
    tsimgrid = linspace(0, a.tEnd, a.nEnd+1);
    
    % labels
    xLab = {'C_A','T'};
    uLab = {'C_{A,f}','T_f','T_{SP}'};
    thetaLab = {'\Delta H','\rho C_p','UA','\Psi','\Omega'};
    
    %%%%%%%%%%%%%%%%
    % Optimization %
    %%%%%%%%%%%%%%%%
    % SS Opt. (previously computed)
    uStarSS = [12.037;306.69;400];
    xStarSS = [0.94761;400];
    OF_SS = a.dt_sys*0.921277660957634;

    %%%%%%%%%%
    % STATES %
    %%%%%%%%%%
    figure(1)
    sgtitle('Plant States') 
    for ii = 1:2
        subplot(2,1,ii)
        plot(tsimgrid,a.XPlantArray(ii,:),'ro-','LineWidth',1.5)
        hold on
        plot(tsimgrid,b.XPlantArray(ii,:),'kx-','LineWidth',1.5)
    
        xlim([0,a.tEnd])
    
        xlabel('t [h]')
        ylabel(xLab{ii})
    
        if ii == 1
            legend({'MHE','bias'},'Location','best')
        end
        grid on
    
    end
    
    figure(2)
    sgtitle('Plant States') 
    for ii = 1:2
        subplot(2,1,ii)
        plot(tsimgrid,a.XPlantArray(ii,:),'ro-','LineWidth',1.5)
        hold on
        plot(tsimgrid,b.XPlantArray(ii,:),'kx-','LineWidth',1.5)

        xlim([0,a.tEnd])
        
        xlabel('t [h]')
        ylabel(xLab{ii})
    
        if ii == 1
            legend({'MHE','bias'},'Location','best')
            %ylim([0.5, 1.5])
        end
        grid on
    
    end

    %%%%%%%%%%
    % INPUTS %
    %%%%%%%%%%
    figure(3)
    sgtitle('Inputs') 
    for ii = 1:3
        subplot(3,1,ii)
        stairs(tsimgrid,a.UPlantArray(ii,:),'ro-','LineWidth',1.5)
        hold on
        stairs(tsimgrid,b.UPlantArray(ii,:),'kx-','LineWidth',1.5)

        xlim([0,a.tEnd])
    
        xlabel('t [h]')
        ylabel(uLab{ii})
    
        grid on
    end
    
 
    
    %%%%%%%%%%%%%%%%%%%
    % CONVERSION (OF) %
    %%%%%%%%%%%%%%%%%%%
    figure(4)
    subplot(1,2,1)
    plot(tsimgrid,a.OFPlantArray/a.dt_sys,'ro-')
    hold on
    plot(tsimgrid,b.OFPlantArray/b.dt_sys,'kx-')
    grid on
    xlim([0,b.tEnd])
    %ylim([0.91,0.93])
    
    xlabel('t [h]')
    ylabel('OF dyn')
    
    subplot(1,2,2)
    plot(tsimgrid,cumsum(a.OFPlantArray/a.dt_sys - b.OFPlantArray/b.dt_sys),'kx-')
  
    xlim([0,b.tEnd])
    %ylim([0.91,0.93])
    
    xlabel('t [h]')
    ylabel('OF dyn')

    grid on
    
   