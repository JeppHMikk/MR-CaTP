clc
clear all
close all

% Seeds:
% Show solved problem: 17,23,25,28,30,40
% Show infeasible problem: 20,24,29,37

rng(17)
set(0, 'DefaultFigureRenderer', 'painters');

%%

options = optimoptions("fmincon","Display","off");

N = 10; % number of robots
T = 500;
width = 50; % environment width
height = 50; % environment height
r = 0.1; % robot radii
epsilon = 10; % minimum clearance
alpha = 0.1; % signal attenuation
d50 = 50; % 50% signal attenuation distance
K = 5; % prediction horizon
reach_time = nan;
reached = false;
N_pois = 5; % Number of POIs

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
l2_min = 0.1; % Fiedler value lower bound
dldp = zeros(N,1);

vs_prev = zeros(K*2*N,1);
vs = zeros(K*2*N,1);
v_prev = zeros(2,1,N);
slack = ones(T,1);

% Sample random POI position
pois = zeros(2,N_pois);
pois(:,2:end) = [unifrnd(-200,200,[1,N_pois-1]);unifrnd(-200,200,[1,N_pois-1])];

u_opt_prev = zeros(2*N*K,1);

nsf_cnt = 0;

%%

% calculate distance between each robot and pois for Hungarian
% algorithm
C = zeros(N_pois-1,N-1);
for i = 2:N_pois
    for j = 2:N
        C(i-1,j-1) = norm(pois(:,i) - p(:,1,j),2);
    end
end
[ass,cost] = munkres(C);
ass = ass + 1; % we ignored the groundstation
ass = [1,ass];

s = zeros(N_pois,N);
iter = 1;
for i = ass
    s(iter,i) = 1;
    iter = iter + 1;
end
S = kron(s,eye(2));

dir_grad = zeros(2*N*K,1);

rt = zeros(T,1); % run time for each time step

for k = 1:T

    tic;

    % Check if POIs have been reached
    if(all(vecnorm(reshape(p(:,k,ass),2,N_pois) - pois,2,1) <= 1) && ~reached)
        reach_time = t(k);
        reached = true;
    end

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

    % Calculate Voronoi constraints
    [C,d] = voronoiConstraints(reshape(p(:,k,:),2,N),r,epsilon);
    B = kron(tril(ones(K)),eye(2*N));
    d = kron(ones(K,1),d - C*reshape(p(:,k,:),2*N,1));
    C = kron(eye(K),C)*B;

    % Calculate constraint for preserving communication
    [DLdp,dldp] = communicationGradient(p(:,k,:),A(:,:,k),v2,K,alpha);
    
    S_tilde = kron(eye(K),S)*B;
    H = (S_tilde'*S_tilde) + 5*(eye(2*N*K));
    f = (-kron(ones(K,1),reshape(pois,2*N_pois,1) - S*reshape(p(:,k,:),2*N,1))'*S_tilde)';
    f = f - (1000*DLdp(end,:).*kron(ones(1,K),1-ones(1,2*N_pois)*S))';
    lb = [repmat([0;0;-0.5*ones(2*(N-1),1)],K,1)];
    ub = [repmat([0;0;0.5*ones(2*(N-1),1)],K,1)];
    costfun = @(x)(x'*H*x + f'*x);
    constfun = @(x)nonlincon(x,reshape(p(:,k,:),2*N,1),K,B,N,alpha,d50,l2_min,r,epsilon);

    vs_new = fmincon(costfun,vs_prev,[],[],[],[],lb,ub,constfun,options);

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

    rt(k) = toc;

    if(mod(k,10) == 1)
        % plot robots
        clf('reset')
        subplot(2,1,1)
        hold on

        set(gca,'ColorOrderIndex',1)
        for i = 1:N
            plot(p(1,1,i),p(2,1,i),'x','LineWidth',2,'MarkerSize',10)
        end
        set(gca,'ColorOrderIndex',1)
        for i = 1:N
            plot(p(1,1:k,i),p(2,1:k,i),'--','LineWidth',0.5)
        end
        set(gca,'ColorOrderIndex',1)
        for i = 1:N
            plot(p(1,k,i),p(2,k,i),'.','LineWidth',2,'MarkerSize',20)
        end
        plot(pois(1,:),pois(2,:),'ko','LineWidth',2,'MarkerSize',10);
        hold off
        axis equal
        box on
        xlim([-200 200])
        ylim([-200 200])
    
        subplot(2,1,2)
        hold on
        yline(l2_min,'k-.','LineWidth',2)
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

k_snap = [0,50,200,500];
k_snap(1) = 1;

for j = 1:4
    k = k_snap(j);
    subplot(2,2,j)
    hold on
    set(gca,'ColorOrderIndex',1)
    for i = 2:N
        plot(p(1,1:k,i),p(2,1:k,i),'-','LineWidth',0.5)
    end
    plot(p(1,k,1),p(2,k,1),'rx','LineWidth',2,'MarkerSize',10)
    set(gca,'ColorOrderIndex',1)
    for i = 2:N
        plot(p(1,k,i),p(2,k,i),'.','LineWidth',2,'MarkerSize',15)
    end
    plot(pois(1,2:end),pois(2,2:end),'ko','LineWidth',2,'MarkerSize',5);
    hold off
    axis equal
    box on
    xlim([-200 200])
    ylim([-200 200])
    xticks(-200:200:200)
    yticks(-200:200:200)
    xlabel('x (m)','Interpreter','latex')
    ylabel('y (m)','Interpreter','latex')
    title("k="+k_snap(j)+" ()",'Interpreter','latex')
    hold off
end
sgtitle('Robots Trajectories','Fontsize',12,'Interpreter','latex')
set(fig2,'Position',[0,0,500,475])

fig3 = figure(3);
hold on
yline(l2_min,'r--','LineWidth',2)
plot(l2(1:T),'k','LineWidth',2)
xline(reach_time,'g-','LineWidth',2)
hold off
box on
legend('$\underline{\lambda_2}$','$\lambda_2$','$k_{reach}$','Interpreter','latex','Location','northeast')
ylim([0,1.1*max(l2)])
xlabel('k ()','Interpreter','latex')
ylabel('$\lambda_2 ()$','Interpreter','latex')
title('Fiedler Value','Fontsize',12,'Interpreter','latex')

set(fig3,'Position',[0,0,500,200])

% exportgraphics(fig2,'poi_pos_exp.eps')
% exportgraphics(fig3,'poi_fiedler_exp.eps')

%%

function [c,ceq] = nonlincon(u,p,K,B,N,alpha,d50,l2_min,r,epsilon)

    p_pred = reshape(kron(ones(K,1),p) + B*u,2,N,K);

    c_comms = [];
    for k = 1:K
        A = zeros(N,N);
        for i = 1:N
            for j = i+1:N
                Aij = arrprob(p_pred(:,i,k),p_pred(:,j,k),alpha,d50);
                A(i,j) = Aij;
                A(j,i) = Aij;
            end
        end
        D = diag(sum(A,2));
        L = D - A;
        l = sort(eig(L));
        l2 = l(2);
        c_comms = [c_comms;l2_min-l2];
    end

    c_coll = [];
    for k = 1:K
        for i = 1:N
            for j = i+1:N
                dij = norm(p_pred(:,i,k)-p_pred(:,j,k),2);
                c_coll = [c_coll;2*r + epsilon - dij];
            end
        end
    end

    c = [c_comms;c_coll];
    ceq = [];

end

function alpha = arrprob(p1,p2,alpha,d50)

    d = norm(p1-p2,2);
    alpha = 1 - 1/(1 + exp(-alpha*(d - d50)));
    
end

function [DLdp,dldp] = communicationGradient(p,A,v2,K,alpha)

    N = size(p,3);
    
    dAdt2 = zeros(N,N,2,N);
    dDdt2 = zeros(N,N,2,N);
    dldp = [];
    for i = 1:N
        for j = 1:N
            if(i ~= j)
                dij = norm(p(:,i) - p(:,j),2);
                dAdt2(i,j,1,i) = -alpha*(1 - A(i,j))*A(i,j)*((p(1,i) - p(1,j))/dij);
                dAdt2(i,j,2,i) = -alpha*(1 - A(i,j))*A(i,j)*((p(2,i) - p(2,j))/dij);
                dAdt2(j,i,1,i) = dAdt2(i,j,1,i);
                dAdt2(j,i,2,i) = dAdt2(i,j,2,i);
            end
        end
        %dAdt2(:,:,1,i) = dAdt2(:,:,1,i);
        %dAdt2(:,:,2,i) = dAdt2(:,:,2,i);
        dDdt2(:,:,1,i) = diag(sum(dAdt2(:,:,1,i),2));
        dDdt2(:,:,2,i) = diag(sum(dAdt2(:,:,2,i),2));
        dldp(end+1) = v2'*(dDdt2(:,:,1,i) - dAdt2(:,:,1,i))*v2;
        dldp(end+1) = v2'*(dDdt2(:,:,2,i) - dAdt2(:,:,2,i))*v2;
    end

    DLdp = kron(tril(ones(K)),dldp);

end

function [C_blk,d_cat] = voronoiConstraints(p,r,epsilon)
    
    N = size(p,2);

    % Calculate Voronoi constraints
    D = delaunay(p(1,:)',p(2,:)');
    D = [[D(:,1),D(:,2)];[D(:,1),D(:,3)];[D(:,2),D(:,3)]];
    D = unique(D,'rows');
    C = cell(N,1);
    d = cell(N,1);
    for h = 1:size(D,1)
        i = D(h,1);
        j = D(h,2);
    
        cij = (p(:,j) - p(:,i))/norm((p(:,j) - p(:,i)),2);
        dij = 1/2*cij'*(p(:,j) + p(:,i));
    
        C{i}(end+1,:) = cij;
        d{i}(end+1,:) = dij - (r + epsilon/2);
        C{j}(end+1,:) = -cij;
        d{j}(end+1,:) = -dij - (r + epsilon/2);
    
    end
    
    C_blk = [];
    d_cat = [];
    
    for i = 1:N
        
        C_blk = blkdiag(C_blk,C{i});
        d_cat = [d_cat;d{i}];
    
    end

end

function [c,ceq] = nlc(v,n)

    v = reshape(v(1:2*n),2,n);

    c = [];

    for i = 1:n

        c = norm(v(:,i),2) - 1;

    end

    ceq = [];


end


