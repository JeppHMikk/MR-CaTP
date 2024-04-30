clc
close all
clear all

h1 = openfig('figs/pos_wo_com_serv.fig','reuse'); % open figure
ax1 = gca; % get handle to axes of figure
h2 = openfig('figs/pos_w_com_serv.fig','reuse');
ax2 = gca;
% test1.fig and test2.fig are the names of the figure files which you would % like to copy into multiple subplots
h3 = figure; %create new figure
s1 = subplot(1,2,1); %create and get handle to the subplot axes
axis equal
box on
xlim([-200 200])
ylim([-200 200])
xticks(-200:200:200)
yticks(-200:200:200)
xlabel('x (m)','Interpreter','latex')
ylabel('y (m)','Interpreter','latex')
title('Robots wo. CIS','Interpreter','latex')

s2 = subplot(1,2,2);
axis equal
box on
xlim([-200 200])
ylim([-200 200])
xticks(-200:200:200)
yticks(-200:200:200)
xlabel('x (m)','Interpreter','latex')
ylabel('y (m)','Interpreter','latex')
title('Robots w. CIS','Interpreter','latex')

fig1 = get(ax1,'children'); %get handle to all the children in the figure
fig2 = get(ax2,'children');
copyobj(fig1,s1); %copy children to new parent axes i.e. the subplot axes
copyobj(fig2,s2);

sgtitle({'Robots w. and wo. Communication Insurance Service'},'Interpreter','latex')

set(h3,'Position',[0,0,500,275])

exportgraphics(h3,'figs/pos_w_wo_com_serv.eps')

