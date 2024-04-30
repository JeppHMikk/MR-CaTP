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