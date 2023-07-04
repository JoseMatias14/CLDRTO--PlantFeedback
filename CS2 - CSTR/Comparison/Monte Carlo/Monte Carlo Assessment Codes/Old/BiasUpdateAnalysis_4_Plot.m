clear 
clc
close all

% saving results
load('MC_parameters_CLv2_med.mat')

%% Plotting results
tsimgrid = linspace(0, tEnd, nEnd+1);

% labels
xLab = {'C_A','T'};
uLab = {'C_{A,f}','T_f','T_c'};

for ii = 1:nPar
    %%%%%%%%%%
    % STATES %
    %%%%%%%%%%
    figure((ii - 1)*5 + 1)
    sgtitle('States - Plant') 
    
    for cc = 1:2
        subplot(2,1,cc)
        % nominal
        plot(tsimgrid,XPlantPlot{ii,1}(cc,:),'k','LineWidth',1.5)
        hold on 
        % nominal + sigma
        plot(tsimgrid,XPlantPlot{ii,2}(cc,:),'k:','LineWidth',1)
        % nominal - sigma
        plot(tsimgrid,XPlantPlot{ii,3}(cc,:),'k:','LineWidth',1)
        % random
        for jj = 3:3 + nSim
             plot(tsimgrid,XPlantPlot{ii,jj}(cc,:),'Color',[0,0,0,0.2])
        end
        % SS Opt
        xlim([0,tEnd])

        xlabel('t [h]')
        ylabel(xLab{cc})

        grid on

    end

    figure((ii - 1)*5 + 2)
    sgtitle('Bias') 
    for cc = 1:2
        subplot(2,1,cc)
        % nominal
        plot(tsimgrid,biasPlot{ii,1}(cc,:),'k','LineWidth',1.5)
        hold on 
        % nominal + sigma
        plot(tsimgrid,biasPlot{ii,2}(cc,:),'k:','LineWidth',1)
        % nominal - sigma
        plot(tsimgrid,biasPlot{ii,3}(cc,:),'k:','LineWidth',1)
        % random 
        for jj = 3:3 + nSim
             plot(tsimgrid,biasPlot{ii,jj}(cc,:),'Color',[0,0,0,0.2])
        end

        xlim([0,tEnd])

        xlabel('t [h]')
        ylabel(xLab{cc})

        grid on

    end

    figure((ii - 1)*5 + 3)
    sgtitle('Inputs')
    for cc = 1:3
        subplot(3,1,cc)
            % nominal
            stairs(tsimgrid,UPlot{ii,1}(cc,:),'k:','LineWidth',1.5)
            hold on
            % nominal + sigma
            plot(tsimgrid,UPlot{ii,2}(cc,:),'k:','LineWidth',1)
            % nominal - sigma
            plot(tsimgrid,UPlot{ii,3}(cc,:),'k:','LineWidth',1)
            % randomly picked
            for jj = 1:3 + nSim
                plot(tsimgrid,UPlot{ii,jj}(cc,:),'Color',[0,0,0,0.2])
            end

            % SS Opt
            %yline(uStarSS(cc),'k--','LineWidth',1.5)
            
            xlim([0,tEnd])
    
            xlabel('t [h]')
            ylabel(uLab{cc})
    
            grid on
    end
    
    figure((ii - 1)*5 + 4)
    title('OF') 
        % nominal
        plot(tsimgrid,OFPlot{ii,1},'k:','LineWidth',1.5)
        hold on
        % nominal + sigma
        plot(tsimgrid,OFPlot{ii,2},'k:','LineWidth',1)
        % nominal - sigma
        plot(tsimgrid,OFPlot{ii,3},'k:','LineWidth',1)
        % randomly picked
        for jj = 1:3 + nSim
            plot(tsimgrid,OFPlot{ii,jj},'Color',[0,0,0,0.2])
        end
        grid on

        xlim([0,tEnd])

        xlabel('t [h]')
        ylabel('OF [$]')
        % SS Opt
        %yline(OFStarSS,'k--','LineWidth',1.5)
    
    figure((ii - 1)*5 + 5)
    title('Converged?')
    hold on 
    for jj = 1:3 + nSim
        for kk = dRTO_exec:dRTO_exec:nEnd
            if SolFlagPlot{ii,jj}((kk - dRTO_exec)/dRTO_exec + 1) == 1
                scatter(kk,jj,'ko')
            else
                scatter(kk,jj,'rx')
            end
        end  
    end
    grid on

end

