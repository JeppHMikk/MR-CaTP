clc
clear all
close all

% Seeds:
% Show solved problem: 17,23,25,28,30,40
% Show infeasible problem: 20,24,29,37

addpath functions/

rng(41)
set(0, 'DefaultFigureRenderer', 'painters');

%%

options = optimoptions("quadprog",'Display','none','Algorithm','interior-point-convex');

N = 10; % number of robots
T = 1000; %length(t);
width = 50; % environment width
height = 50; % environment height
r = 0.1; % robot radii
epsilon = 10; % minimum clearance
alpha = 0.1; % signal attenuation
d50 = 50; % 50% signal attenuation distance
K = 4; % number of waypoints
N_pois = 5;
l2_min_hard = 0.25; % Fiedler value hard constraint
l2_min_soft = 1; % Fiedler value soft constraint
reach_time = nan;
reached = false;

% Generate random initial positions
p = zeros(2,T,N);
for j = 2:N
    while(true)
        pj = [unifrnd(-width,width,1);unifrnd(-height,height,1)];
        if(all(vecnorm(pj - p(:,1,1:j-1),2) >= 2*r + 2*epsilon))
            p(:,1,j) = pj;
            break
        end
    end
end

v = zeros(2,T,N);
p(:,1,1) = zeros(2,1);

A = zeros(N,N,T); % adjacency matrix

l2 = zeros(1,T); % Fiedler value
dl2 = zeros(1,T); % Fiedler value derivative

dldp = zeros(N,1);

vs_prev = zeros(K*2*N + K,1);
vs = zeros(K*2*N,1);
v_prev = zeros(2,1,N);
slack = ones(T,1);

u_opt_prev = zeros(2*N*K,1);

nsf_cnt = 0;

v_ref = 0.1*randn(2*N,1);

%%

for k = 1:T

    % Calculate current adjacency matrix and eigenvalue
    for i = 1:N
        for j = i+1:N
            Aij = arrprob(p(:,k,i),p(:,k,j),alpha,d50);
            A(i,j,k) = Aij;
            A(j,i,k) = Aij;
        end
    end
    D = diag(sum(A(:,:,k),2));
    L = D - A(:,:,k);
    [V,E] = eig(L);
    v2 = V(:,2);
    l2(k) = E(2,2);
    if(k == 1)
        l2_est(k) = l2(k);
    end

    % Calculate Voronoi constraints
    [C,d] = voronoiConstraints(reshape(p(:,k,:),2,N),r,epsilon);
    B = kron(tril(ones(K)),eye(2*N));
    d = kron(ones(K,1),d - C*reshape(p(:,k,:),2*N,1));
    C = kron(eye(K),C)*B;

    % Calculate constraint for preserving communication
    [DLdp,dldp] = communicationGradient(p(:,k,:),A(:,:,k),v2,K,alpha);
    
    H = blkdiag(eye(2*N*K),diag(0.1*ones(K,1)));
    f = -[kron(ones(K,1),v_ref);zeros(K,1)];
    A_hard = [-DLdp,zeros(size(DLdp,1),K)];
    b_hard = repmat(l2(k) - l2_min_hard,K,1);
    A_soft = [-DLdp,-eye(K)];
    b_soft = repmat(l2(k) - l2_min_soft,K,1);
    % Ac = [A_hard;A_soft;[C,zeros(size(C,1),size(A_hard,2)-size(C,2))]]; %...
    % bc = [b_hard;b_soft;d];
    Ac = [C,zeros(size(C,1),size(A_hard,2)-size(C,2))]; %...
    bc = d;
    lb = [repmat([0;0;-0.5*ones(2*(N-1),1)],K,1);zeros(K,1)];
    ub = [repmat([0;0;0.5*ones(2*(N-1),1)],K,1);inf(K,1)];
    hot_start = [vs_prev(1:2*N*(K-1));vs_prev(2*N*(K-2)+1:2*N*(K-1));vs_prev(2*N*K+1:end)];
    vs_new = quadprog(H,f,Ac,bc,[],[],lb,ub,hot_start,options);

    if(~isempty(vs_new))
        vs_prev = vs_new;
        vs = vs_new(1:K*2*N);
    else
        disp('NSF')
        vs = zeros(size(vs));
    end

    l_pred = l2(k) + DLdp*vs;
    v(:,k,:) = reshape(vs(1:2*N),2,N);

    % update positions of robots
    for i = 1:N
        p(:,k+1,i) = p(:,k,i) + v(:,k,i);
    end

    v_ref = reshape(v(:,k,:),2*N,1) + 0.01*randn(2*N,1);

    if(mod(k,10) == 1)
        clf('reset')
        subplot(2,1,1)
        hold on

        set(gca,'ColorOrderIndex',1)
        for i = 1:N
            plot(p(1,k,i),p(2,k,i),'.','LineWidth',2,'MarkerSize',20)
        end
        hold off
        axis equal
        box on
        xlim([-200 200])
        ylim([-200 200])
    
        subplot(2,1,2)
        hold on
        yline(l2_min_hard,'k-.','LineWidth',2)
        yline(l2_min_soft,'k--','LineWidth',2)
        plot(l2(1:k),'k','LineWidth',2)
        plot(k:(k+K),[l2(k);l_pred],'r','LineWidth',2)
        xline(reach_time,'g-','LineWidth',2)
        hold off
        box on
        xlim([1,T])
        ylim([0,1.1*max(l2)])
    
        drawnow
    end


end

%%

close all

fig2 = figure(2);
k = T;
hold on
set(gca,'ColorOrderIndex',1)
for i = 1:N
    plot(p(1,1:k,i),p(2,1:k,i),'-','LineWidth',0.5)
end
set(gca,'ColorOrderIndex',1)
for i = 1:N
    plot(p(1,k,i),p(2,k,i),'.','LineWidth',2,'MarkerSize',15)
end
hold off
axis equal
box on
xlim([-200 200])
ylim([-200 200])
xlabel('x (m)','Interpreter','latex')
ylabel('y (m)','Interpreter','latex')
hold off
title('Robots wo. Communication Insurance Service','Fontsize',12,'Interpreter','latex')
set(fig2,'Position',[0,0,500,475])

% exportgraphics(fig2,'pos_wo_com_ser.eps')

fig3 = figure(3);
hold on
yline(l2_min_hard,'r-.','LineWidth',2)
yline(l2_min_soft,'r--','LineWidth',2)
plot(l2,'k','LineWidth',2)
hold off
box on
legend('$\underline{\lambda_2}$','$\stackrel{\lambda_2}{\sim}$','$\lambda_2$','Interpreter','latex','Location','northeast')
xlim([1,T])
ylim([0,1.1*max(l2)])
xlabel('k ()','Interpreter','latex')
ylabel('$\lambda_2 ()$','Interpreter','latex')
title('Fiedler Value wo. Communication Insurance Service','Fontsize',12,'Interpreter','latex')

set(fig3,'Position',[0,0,500,200])

exportgraphics(fig3,'figs/fiedler_wo_com_ser.eps')


